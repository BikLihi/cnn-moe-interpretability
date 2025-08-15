import torch
import numpy as np
import math

from models.moe_layer.hard_gating_networks import AbsoluteImportanceGate
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
    if epoch <= 100:
        return epoch
    return 100 - (epoch - 100) / 2.0




# Fix parameters
top_k = 2
use_noise = True
num_blocks = 1
num_epochs = 200
batch_size = 400
learning_rate = 0.0001
optimizer_class = torch.optim.Adam
gating_network = AbsoluteImportanceGate


# Variable parameters
constr_threshold_values = [0.1, 0.3]
schedule_lambda_values = [schedule_1, schedule_2, schedule_3]
num_exp_values = [4, 8]


for num_exp in num_exp_values:
    for constr_threshold in constr_threshold_values:
        for schedule_lambda in schedule_lambda_values:

            # Build MoE Layer
            moe_layer = ResNetMoeLayer(
                in_channels=64,
                out_channels=128,
                num_experts=num_exp,
                top_k=top_k,
                num_blocks=num_blocks,
                use_noise=use_noise,
                gating_network=gating_network,
                constr_threshold=constr_threshold
            )

            # Build complete model with integrated MoE layer
            model = ShallowResNetMoE(
                num_blocks=1,
                moe_layer=moe_layer,
                name='ShallowResNetMoE_AbsoluteImportanceGate_exp=' + str(num_exp) + '_thresh=' + str(constr_threshold)
            )

            optimizer = optimizer_class(model.parameters(), learning_rate)
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule_lambda)


            custom_config = {
                'gating_network':gating_network.__qualname__,
                'num_experts':num_exp,
                'constr_threshold': constr_threshold
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
                wandb_name='ShallowResNetMoE_AbsoluteImportanceGate_exp=' + str(num_exp) + '_thresh=' + str(constr_threshold),
                wandb_tags=['compare_gates', 'AbsoluteImportanceGate', 'ShallowResNetMoE', 'HardConstraint'],
                wandb_custom_config=custom_config
            )

