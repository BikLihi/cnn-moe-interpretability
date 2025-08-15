from utils.visualize_activations import CNNActivationVisualization
import collections

import wandb
import time
import numpy as np
import pandas as pd
import os
import cv2

from PIL import Image

import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader


from models.retina.retinanet import model_single_gating_no_fpn, coco_eval, model
from models.retina.retinanet.dataloader import CocoDataset, CocoSubDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, \
    Normalizer
from models.moe_layer.soft_gating_networks import FCGate, SimpleGate

from utils.dataset_utils import load_retinanet_image
from utils.visualize_features import CNNLayerVisualization

# Create the model
retinanet = model_single_gating_no_fpn.resnet50(num_classes=80, pretrained=False)
retinanet.freeze_bn()

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

retinanet.classificationModel = model_single_gating_no_fpn.ClassificationModelMoE(num_features_in=256, num_experts=4, top_k=2, gating_network=gate1, num_classes=80)
retinanet.regressionModel = model_single_gating_no_fpn.RegressionModelMoE(num_features_in=256, num_experts=4, top_k=2, gating_network=gate2)

file_model = wandb.restore('retinanet_final.tar', run_path='lukas-struppek/RetinaNet/30x8izsh')
retinanet.load_state_dict(torch.load('retinanet_final.tar')['model_state_dict'])

retinanet.cuda()

model = retinanet
layer = retinanet.regressionModel.gate.conv1

visualizer = CNNActivationVisualization(model, layer)

# print(visualizer2.compute_activations(image, 256, feature_level=0, save_path='.', top_k=3))

# for i, image_path in enumerate(os.listdir('/home/lb4653/mixture-of-experts-thesis/data/coco/images/val2017')):
#     image_path = os.path.join('/home/lb4653/mixture-of-experts-thesis/data/coco/images/val2017', image_path)
#     print('Image ' + str(i))
#     image = load_retinanet_image(image_path)
#     activations.append([visualizer1.compute_activations(image, 256, feature_level=0, save_path='.', top_k=3), image_path])


# act_df = pd.DataFrame(activations)
# act_df.to_csv('/home/lb4653/mixture-of-experts-thesis/analysis/retinanet/activations.csv')

image = load_retinanet_image('/home/lb4653/mixture-of-experts-thesis/analysis/retinanet/test2/hund.jpg')
top_activation = visualizer.compute_activations(image, 256, feature_level=0, save_path='.', top_k=3)
for i in top_activation:
    feature_vis = CNNLayerVisualization(model, layer, i, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], save_path='/home/lb4653/mixture-of-experts-thesis/activations/')
    feature_vis.visualise_layer_with_hooks(init_size=8, scaling_factor=1.2, scaling_steps=30, iterations_per_level=20, file_name='layer', lr=0.05, verbose=True)
