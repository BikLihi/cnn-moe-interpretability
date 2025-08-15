import torch
import numpy as np
import math

from models.moe_layer.soft_gating_networks import SimpleGate
from models.moe_layer.resnet.shallow_resnet_moe import ShallowResNetMoE, ResNetMoeLayer
from datasets.cifar100_dataset import CIFAR100Dataset
from utils.dataset_utils import train_test_split, get_transformation


# Setting seeds
torch.manual_seed(0)
np.random.seed(0)

# Loading datasets
transformations_training = get_transformation('cifar100', phase='training')
transformations_test = get_transformation('cifar100', phase='test')
cifar_data = CIFAR100Dataset(
    root_dir='/home/lb4653/mixture-of-experts-thesis/data/cifar100/training', transform=transformations_training)
training_data, validation_data = train_test_split(cifar_data, [0.8, 0.2])
validation_data.transform = transformations_test

# Define scheduler functions
def schedule_1(epoch):
    if epoch <= 50:
        return 10
    if epoch < 100:
        return 1
    return 0.1

def schedule_2(epoch):
    return 1

def schedule_3(epoch):
    if epoch <= 100:
        return 10
    if epoch < 180:
        return 1
    return 0.1


# Fix parameters
num_exp = 1
top_k = 1
use_noise = False
num_epochs = 200
batch_size = 480
learning_rate = 0.0001
optimizer_class = torch.optim.Adam
gating_network = SimpleGate
w_aux_loss = 0.0
loss_fkt = None


# Variable parameters
schedule_lambda_values = [schedule_1, schedule_2, schedule_3]
num_blocks_values = [1, 2]


# Variable parameters
for schedule_lambda in schedule_lambda_values:
    for num_blocks in num_blocks_values:

        # Build MoE Layer
        moe_layer = ResNetMoeLayer(
            in_channels=64,
            out_channels=128,
            num_experts=num_exp,
            top_k=top_k,
            num_blocks=num_blocks,
            use_noise=use_noise,
            gating_network=gating_network,
            loss_fkt=loss_fkt,
            w_aux_loss=w_aux_loss
        )

        # Build complete model with integrated MoE layer
        model = ShallowResNetMoE(
            num_blocks=1,
            moe_layer=moe_layer,
            name='ShallowResNetMoE_Baseline_blocks=' + str(num_blocks)
        )

        optimizer = optimizer_class(model.parameters(), learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule_lambda)


        custom_config = {
            'gating_network':None,
            'num_experts':num_exp,
            'num_blocks':num_blocks
        }

        model.fit(
            training_data=training_data,
            validation_data=validation_data,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            optimizer=optimizer,
            early_stopping=False,
            enable_logging=True,
            lr_scheduler=lr_scheduler,
            wandb_project='analyze_moe_gates',
            wandb_name='ShallowResNetMoE_Baseline_blocks=' + str(num_blocks),
            wandb_tags=['compare_gates', 'Baseline', 'ShallowResNetMoE'],
            wandb_custom_config=custom_config
        )

