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
from models.retina.analyze_regression_moe import analyze_regression
from models.retina.retinanet.dataloader import CocoDataset, CocoSubDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, \
    Normalizer
from models.moe_layer.soft_gating_networks import SimpleGate
from models.moe_layer.static_gating_networks import SingleWeightingGatingNetwork

class_file = '/home/lb4653/mixture-of-experts-thesis/data/coco/subset.names'
classes = [line.strip() for line in open(class_file, 'r')]

# Create the model
retinanet = model.resnet50(num_classes=len(classes), pretrained=False)
retinanet.freeze_bn()

# Freeze all layers
for param in retinanet.parameters():
    param.requires_grad = False

# Add MoE Regressor
gate = SimpleGate(in_channels=256, 
                    num_experts=4,
                    top_k=2,
                    use_noise=True,
                    name='SimpleGate',
                    loss_fkt='kl_divergence',
                    w_aux_loss=0.5
                    )

retinanet.regressionModel = model.RegressionModelMoE(num_features_in=256, num_experts=4, top_k=2, gating_network=gate)
file_model = wandb.restore('retinanet_final.tar', run_path='lukas-struppek/RetinaNet/b1468is2')
retinanet.load_state_dict(torch.load('retinanet_final.tar')['model_state_dict'])

retinanet = retinanet.cuda()

analyze_regression('/home/lb4653/mixture-of-experts-thesis/data/test/predictions/', retinanet, class_file)