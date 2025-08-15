import argparse
import collections

import wandb
import time
import numpy as np
import os
import cv2

from PIL import Image

import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader


from models.retina.retinanet import model, coco_eval
from models.retina.retinanet.dataloader import CocoDataset, CocoSubDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, \
    Normalizer
from models.moe_layer.soft_gating_networks import SimpleGate

def main(args=None):
    # Parameters
    coco_path = 'data/coco'
    test_data_path = 'data/test'
    num_epochs = 20
    batch_size = 4
    enable_logging = True
    initial_lr = 1e-5

    class_file = 'data/coco/coco.names'
    classes = [line.strip() for line in open(class_file, 'r')]
    dataset_train = CocoDataset(coco_path, set_name='train2017',
                                transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
    dataset_val = CocoDataset(coco_path, set_name='val2017',
                                transform=transforms.Compose([Normalizer(), Resizer()]))

    sampler = AspectRatioBasedSampler(dataset_train, batch_size=batch_size, drop_last=False)
    dataloader_train = DataLoader(dataset_train, num_workers=2, collate_fn=collater, batch_sampler=sampler)

    if dataset_val is not None:
        sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
        dataloader_val = DataLoader(dataset_val, num_workers=2, collate_fn=collater, batch_sampler=sampler_val)

    # Create the model
    retinanet = model.resnet50(num_classes=len(classes), pretrained=True)

    retinanet = retinanet.cuda()
    print('Training on Cuda')

    retinanet.training = True

    def schedule_function(epoch):
        if epoch <= 12:
            return 1
        if epoch <= 17:
            return 0.1
        return 0.01

    optimizer = optim.Adam(retinanet.parameters(), lr=initial_lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule_function)

    loss_hist = collections.deque(maxlen=500)

    retinanet.train()
    retinanet.freeze_bn()

    print('Num training images: {}'.format(len(dataset_train)))

    # Initialize new W&B run and declare hyperparameters
    if enable_logging:
        wandb.init(project='RetinaNet', name='Baseline',
            reinit=True,
            config={
                'Num epochs': num_epochs,
                'Batch size': batch_size,
                'Initial lr': optimizer.param_groups[0]['lr'],
                'Optimizer': optimizer,
                'Num training data': (len(dataset_train)),
                'Num validation data': len(dataset_val)
            })
        wandb.watch(retinanet)

    start_time = time.time()

    total_iterations = 0
    for epoch_num in range(0, num_epochs):

        retinanet.train()
        retinanet.freeze_bn()

        epoch_loss = []

        for iter_num, data in enumerate(dataloader_train):
            total_iterations += 1
            try:
                retinanet.train()
                retinanet.freeze_bn()

                optimizer.zero_grad()

                classification_loss, regression_loss, aux_loss = retinanet([data['img'].cuda().float(), data['annot'].cuda()])
                    
                classification_loss = classification_loss.mean()
                regression_loss = regression_loss.mean()

                loss = classification_loss + regression_loss

                if bool(loss == 0):
                    continue

                loss.backward()

                torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)

                optimizer.step()

                loss_hist.append(float(loss))

                epoch_loss.append(float(loss))

                print(
                    'Epoch: {} | Iteration: {} | Cls loss: {:1.5f} | Reg loss: {:1.5f} | Running loss: {:1.5f}'.format(
                        epoch_num, iter_num, float(classification_loss), float(regression_loss), np.mean(loss_hist)))

                if total_iterations % 2000 == 0 and enable_logging:
                    wandb.log({
                        'Classification loss ': float(classification_loss),
                        'Regression loss ': float(regression_loss),
                        'Running loss ': np.mean(loss_hist),
                        'Learning rate': optimizer.param_groups[0]['lr']
                    })

                    for folder in os.scandir(test_data_path):
                        for img_name in os.listdir(folder.path):
                            test_data = os.path.join(folder, img_name)
                            box_image = log_bounding_boxes(test_data, dataset_val.labels, retinanet)
                            wandb.log({os.path.join(folder.name, img_name): box_image})

                del classification_loss
                del regression_loss
            except Exception as e:
                print(e)
                continue

        print('Evaluating dataset')
        coco_eval_result = coco_eval.evaluate_coco(dataset_val, retinanet)

        try:
            metric_names = [
                'AP@50:95', 'AP50', 'AP75', 'AP_small', 'AP_medium', 'AP_large', 'AR_maxDets=1', 'AR_maxDets=10',
                'AR-maxDets=100', 'AR@50:95_small', 'AR@50:95_medium', 'AR@50:95_large'
            ]
            eval_results = {
                'metrics/' + str(metric_names[i]): coco_eval_result['coco_eval'].stats[i]
                for i in range(len(metric_names))
            }
            wandb.log(eval_results)
            wandb.log({'epoch': epoch_num})
        except:
            print('Exception during evaluation ocurred')
            

        if enable_logging:
            if epoch_num % 5 == 0 and epoch_num > 0:
                filename = 'retinanet_baseline' + '_epoch_' + str(epoch_num) + '.tar'
                torch.save({
                    'epoch': epoch_num,
                    'model_state_dict': retinanet.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, filename)
                wandb.save(filename)

        scheduler.step()
        torch.save(retinanet.state_dict(), '{}_retinanet_{}.pt'.format('coco', epoch_num))

    retinanet.eval()

    torch.save(retinanet.state_dict(), 'model_final.pt')

    time_elapsed = time.time() - start_time
    time_elapsed_string = time.strftime("%H:%M:%S", time.gmtime(time_elapsed))

    if enable_logging:
        wandb.config.update({
        'total_training_time': time_elapsed_string
        })

        filename = 'retinanet' + '_final' + '.tar'
        torch.save({
            'model_state_dict': retinanet.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, filename)

        wandb.save(filename)
        wandb.join()


def log_bounding_boxes(image_path, class_labels, retinanet):
    retinanet.eval()
    # load images
    image = cv2.imread(image_path)
    
    if image is None:
        return None

    rows, cols, cns = image.shape

    smallest_side = min(rows, cols)

    # rescale the image so the smallest side is min_side
    min_side = 608
    max_side = 1024
    scale = min_side / smallest_side

    # check if the largest side is now greater than max_side, which can happen
    # when images have a large aspect ratio
    largest_side = max(rows, cols)

    if largest_side * scale > max_side:
        scale = max_side / largest_side

    # resize the image with the computed scale
    image = cv2.resize(image, (int(round(cols * scale)), int(round((rows * scale)))))
    rows, cols, cns = image.shape

    pad_w = 32 - rows % 32
    pad_h = 32 - cols % 32

    new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
    new_image[:rows, :cols, :] = image.astype(np.float32)
    image = new_image.astype(np.float32)
    image /= 255
    image -= [0.485, 0.456, 0.406]
    image /= [0.229, 0.224, 0.225]
    image = np.expand_dims(image, 0)
    image = np.transpose(image, (0, 3, 1, 2))

    with torch.no_grad():
        image = torch.from_numpy(image)
        if torch.cuda.is_available():
            image = image.cuda()

        scores, classification, transformed_anchors = retinanet(image.cuda().float())
        idxs = np.where(scores.cpu() > 0.5)
        all_boxes = []

        for j in range(idxs[0].shape[0]):
            bbox = transformed_anchors[idxs[0][j], :]

            x1 = int(bbox[0]) / cols
            y1 = int(bbox[1]) / rows
            x2 = int(bbox[2]) / cols
            y2 = int(bbox[3]) / rows
            label_name = class_labels[int(classification[idxs[0][j]])]
            score = scores[j].cpu().item()
            box_data = {
                'position':{
                    'minX': x1,
                    'maxX': x2,
                    'minY': y1,
                    'maxY': y2
                    },
                'class_id': classification[idxs[0][j]].cpu().item(),
                'box_caption': '{} {:.3f}'.format(label_name, score),
                'scores': {'score': score} }

            all_boxes.append(box_data)
    image_orig = Image.open(image_path)
    box_image = wandb.Image(image_orig, boxes = {"predictions": {"box_data": all_boxes, "class_labels" : class_labels}})
    print('logging image: ', image_path)
    return box_image


if __name__ == '__main__':
    main()
