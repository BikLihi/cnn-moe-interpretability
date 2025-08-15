import argparse
import torch
from torchvision import transforms
import wandb
from retinanet import model
from retinanet.dataloader import CocoDataset, CocoSubDataset, Resizer, Normalizer
from retinanet import coco_eval
from models.moe_layer.soft_gating_networks import FCGate
import json


assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))

 
def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--coco_path', default='data/coco', help='Path to COCO directory')

    parser = parser.parse_args(args)

    dataset = CocoDataset(parser.coco_path, set_name='test2017',
                              transform=transforms.Compose([Normalizer(), Resizer()]))

    # Create the model
    retinanet = model.resnet50(num_classes=80,)
    retinanet.load_state_dict(torch.load('baseline_state_dict.pt'))

    # Freeze all layers
    for param in retinanet.parameters():
        param.requires_grad = False

    # Add MoE Predictors
    gate1 = FCGate(in_channels=256, 
                        num_experts=4,
                        top_k=2,
                        use_noise=True,
                        name='FCGate',
                        loss_fkt='kl_divergence',
                        w_aux_loss=0.25
                        )
    
    gate2 = FCGate(in_channels=256, 
                    num_experts=4,
                    top_k=2,
                    use_noise=True,
                    name='FCGate',
                    loss_fkt='kl_divergence',
                    w_aux_loss=0.25
                    )

    retinanet.classificationModel = model.ClassificationModelMoE(num_features_in=256, num_experts=4, top_k=2, gating_network=gate1, num_classes=80)
    retinanet.regressionModel = model.RegressionModelMoE(num_features_in=256, num_experts=4, top_k=2, gating_network=gate2)

    file_model = wandb.restore('retinanet_final.tar', run_path='lukas-struppek/RetinaNet/16q9vpvj')
    retinanet.load_state_dict(torch.load(file_model.name)['model_state_dict'])


    use_gpu = True

    if use_gpu:
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()

    retinanet.training = False
    retinanet.eval()
   
    with torch.no_grad():

        # start collecting results
        results = []
        image_ids = []

        for index in range(len(dataset)):
            data = dataset[index]
            scale = data['scale']

            # run network
            if torch.cuda.is_available():
                scores, labels, boxes = retinanet(data['img'].permute(2, 0, 1).cuda().float().unsqueeze(dim=0))
            else:
                scores, labels, boxes = retinanet(data['img'].permute(2, 0, 1).float().unsqueeze(dim=0))
            scores = scores.cpu()
            labels = labels.cpu()
            boxes  = boxes.cpu()

            # correct boxes for image scale
            boxes /= scale

            if boxes.shape[0] > 0:
                # change to (x, y, w, h) (MS COCO standard)
                boxes[:, 2] -= boxes[:, 0]
                boxes[:, 3] -= boxes[:, 1]

                # compute predicted labels and scores
                #for box, score, label in zip(boxes[0], scores[0], labels[0]):
                for box_id in range(boxes.shape[0]):
                    score = float(scores[box_id])
                    label = int(labels[box_id])
                    box = boxes[box_id, :]

                    # scores are sorted, so we can break
                    if score < 0.05:
                        break

                    # append detection for each positively labeled class
                    image_result = {
                        'image_id'    : dataset.image_ids[index],
                        'category_id' : dataset.label_to_coco_label(label),
                        'score'       : float(score),
                        'bbox'        : box.tolist(),
                    }

                    # append detection to results
                    results.append(image_result)

            # append image to list of processed images
            image_ids.append(dataset.image_ids[index])

            # print progress
            print('{}/{}'.format(index, len(dataset)), end='\r')

        if not len(results):
            return

        # write output
        json.dump(results, open('detections_test-dev2017_EmbeddedMoE_results.json', 'w'), indent=4)

if __name__ == '__main__':
    main()
