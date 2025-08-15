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
    if epoch <= 80:
        return epoch
    if epoch < 160:
        return 160 - epoch
    return 1.0 / math.sqrt(epoch - 160 + 1)

def schedule_2(epoch):
    if epoch <= 80:
        return 80
    if epoch < 160:
        return 160 - epoch
    return 1.0 / math.sqrt(epoch - 160 + 1)

def schedule_3(epoch):
    if epoch <= 25:
        return 1
    if epoch <= 50:
        return 10
    if epoch <= 100:
        return 80
    if epoch <= 150:
        return 10
    return 1



# Fix parameters
top_k = 2
use_noise = True
num_blocks = 1
num_epochs = 200
batch_size = 400
learning_rate = 0.0001
optimizer_class = torch.optim.Adam
gating_network = SimpleGate


# Variable parameters
w_aux_loss_values = [0.1, 0.5, 1]
schedule_lambda_values = [schedule_1, schedule_2, schedule_3]
num_exp_values = [4]


for w_aux_loss in w_aux_loss_values:
    for schedule_lambda in schedule_lambda_values:
        for num_exp in num_exp_values:

            # Build MoE Layer
            moe_layer = ResNetMoeLayer(
                in_channels=64,
                out_channels=128,
                num_experts=num_exp,
                top_k=top_k,
                num_blocks=num_blocks,
                use_noise=use_noise,
                gating_network=gating_network,
                w_aux_loss=w_aux_loss,
                loss_fkt='importance',
            )

            # Build complete model with integrated MoE layer
            model = ShallowResNetMoE(
                num_blocks=1,
                moe_layer=moe_layer,
                name='ShallowResNetMoE_SimpleGate_Importance_exp=' + str(num_exp) + '_w=' + str(w_aux_loss)
            )

            optimizer = optimizer_class(model.parameters(), learning_rate)
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule_lambda)


            custom_config = {
                'gating_network':gating_network.__qualname__,
                'num_experts':num_exp,
                'w_aux_loss': w_aux_loss
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
                wandb_name='ShallowResNetMoE_SimpleGate_Importance_exp=' + str(num_exp) + '_w=' + str(w_aux_loss),
                wandb_tags=['compare_gates', 'SimpleGate', 'ShallowResNetMoE', 'Importance'],
                wandb_custom_config=custom_config
            )

