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

from retinanet import model
from retinanet.dataloader import CocoDataset, CocoSubDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, \
    Normalizer
from torch.utils.data import DataLoader

from retinanet import coco_eval

assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--dataset', default='coco', help='Dataset type, must be one of csv or coco.')
    parser.add_argument('--coco_path', default='data/coco',  help='Path to COCO directory')

    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=16)
    parser.add_argument('--enable_logging', help='Enable wanb logging', default=True)
    parser.add_argument('--image_path', default='data/test', help='folder with test images for logging')

    parser = parser.parse_args(args)

    # Create the data loaders
    if parser.dataset == 'coco':

        if parser.coco_path is None:
            raise ValueError('Must provide --coco_path when training on COCO,')
        
        class_file = 'data/coco/subset.names'
        classes = [line.strip() for line in open(class_file, 'r')]

        dataset_train = CocoSubDataset(parser.coco_path, classes=classes, set_name='train2017',
                                    transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
        dataset_val = CocoSubDataset(parser.coco_path, classes=classes, set_name='val2017',
                                  transform=transforms.Compose([Normalizer(), Resizer()]))
        
    sampler = AspectRatioBasedSampler(dataset_train, batch_size=4, drop_last=False)
    dataloader_train = DataLoader(dataset_train, num_workers=2, collate_fn=collater, batch_sampler=sampler)

    if dataset_val is not None:
        sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
        dataloader_val = DataLoader(dataset_val, num_workers=2, collate_fn=collater, batch_sampler=sampler_val)

    # Create the model
    if parser.depth == 18:
        retinanet = model.resnet18(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 34:
        retinanet = model.resnet34(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 50:
        retinanet = model.resnet50(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 101:
        retinanet = model.resnet101(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 152:
        retinanet = model.resnet152(num_classes=dataset_train.num_classes(), pretrained=True)
    else:
        raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')

    use_gpu = True

    if use_gpu:
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()
            print('Training on Cuda')

    # if torch.cuda.is_available():
    #     retinanet = torch.nn.DataParallel(retinanet).cuda()
    # else:
    #     retinanet = torch.nn.DataParallel(retinanet)

    retinanet.training = True

    optimizer = optim.Adam(retinanet.parameters(), lr=1e-5)

    def schedule_function(epoch):
        if epoch <= 10:
            return 1
        if epoch <= 13:
            return 0.1
        return 0.01

    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
    loss_hist = collections.deque(maxlen=500)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule_function)

    retinanet.train()
    retinanet.freeze_bn()

    print('Num training images: {}'.format(len(dataset_train)))

            # Initialize new W&B run and declare hyperparameters
    if parser.enable_logging:
        wandb.init(project='RetinaNet', name='Init_test_reduced_continue',
                    reinit=True,
                    config={
                        'Num epochs': parser.epochs,
                        'Batch size': sampler.batch_size,
                        'Initial lr': optimizer.param_groups[0]['lr'],
                        'Optimizer': optimizer,
                        'Num training data': (len(dataset_train)),
                        'Num validation data': len(dataset_val)
                    })
        wandb.watch(retinanet)

    start_time = time.time()

    total_iterations = 0
    ############################################
    for epoch_num in range(10, parser.epochs):

        retinanet.train()
        retinanet.freeze_bn()

        epoch_loss = []

        for iter_num, data in enumerate(dataloader_train):
            total_iterations += 1
            try:
                optimizer.zero_grad()

                classification_loss, regression_loss = retinanet([data['img'].cuda().float(), data['annot'].cuda()])
                    
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
                    'Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(
                        epoch_num, iter_num, float(classification_loss), float(regression_loss), np.mean(loss_hist)))


                if total_iterations % 1000 == 0 and parser.enable_logging:
                    wandb.log({
                        'Classification loss ': float(classification_loss),
                        'Regression loss ': float(regression_loss),
                        'Running loss ': np.mean(loss_hist),
                        'Learning rate': optimizer.param_groups[0]['lr']
                    }, step=int(total_iterations/1000))
 
                del classification_loss
                del regression_loss
            except Exception as e:
                print(e)
                continue


        print('Evaluating dataset')
        result = coco_eval.evaluate_coco(dataset_val, retinanet)

        try:
            metric_names = [
                'AP@50:95', 'AP50', 'AP75', 'AP_small', 'AP_medium', 'AP_large', 'AR_maxDets=1', 'AR_maxDets=10',
                'AR-maxDets=100', 'AR@50:95_small', 'AR@50:95_medium', 'AR@50:95_large'
            ]
            eval_results = {
                'metrics/' + str(metric_names[i]): result.stats[i]
                for i in range(len(metric_names))
            }
            wandb.log(eval_results)
            wandb.log({'epoch': epoch_num})
        except:
            print('Exception during evaluation ocurred')
            


        if parser.enable_logging:
            if epoch_num % 5 == 0 and epoch_num > 0:
                filename = 'retinanet' + '_epoch_' + str(epoch_num) + '.tar'
                torch.save({
                    'epoch': epoch_num,
                    'model_state_dict': retinanet.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, filename)
                wandb.save(filename)
            for img_name in os.listdir(parser.image_path):
                box_image = log_bounding_boxes(os.path.join(parser.image_path, img_name), dataset_val.labels, retinanet)
                wandb.log({img_name: box_image})

        scheduler.step()
        torch.save(retinanet.state_dict(), '{}_retinanet_{}.pt'.format(parser.dataset, epoch_num))

    retinanet.eval()

    torch.save(retinanet.state_dict(), 'model_final.pt')

    time_elapsed = time.time() - start_time
    time_elapsed_string = time.strftime("%H:%M:%S", time.gmtime(time_elapsed))

    if parser.enable_logging:
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
            print(x1, x2, y1, y2)
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
