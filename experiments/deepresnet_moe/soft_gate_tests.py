import torch
import numpy as np
import math

from models.moe_layer.resnet.bottleneck_moe_layer import BottleneckMoeLayer
from models.moe_layer.soft_gating_networks import SimpleGate
from models.moe_layer.resnet101_moe.resnet101_moe import DeepResNetMoE
from datasets.cifar100_dataset import CIFAR100Dataset
from utils.dataset_utils import train_test_split, get_transformation
from utils.cifar100_utils import CIFAR100_LABELS


# Setting seeds
torch.manual_seed(42)
np.random.seed(42)

# Loading datasets
transformations_training = get_transformation('cifar100', phase='training')
transformations_test = get_transformation('cifar100', phase='test')
cifar_data = CIFAR100Dataset(
    root_dir='/home/lb4653/mixture-of-experts-thesis/data/cifar100/training', transform=transformations_training)
training_data, validation_data = train_test_split(cifar_data, [0.8, 0.2], seed=42)
validation_data.transform = transformations_test

# Define scheduler functions
def schedule1(epoch):
    if epoch <= 100:
        return 10
    if epoch < 180:
        return 1
    return 0.1


def schedule2(epoch):
    if epoch <=70:
        return 1 + 9.0 / 70.0 * epoch
    if epoch <= 100:
        return 10
    if epoch < 130:
        return 1
    if epoch < 140:
        return 0.1
    return 0.1

    
# Fix parameters
num_epochs = 150
batch_size = 128
learning_rate = 0.0001
optimizer_class = torch.optim.Adam


# Variable parameters
loss_function_Values = ['importance', 'kl_divergence']
schedule_function_values = [schedule1, schedule2]
lift_constraint_values = [None, 50, 75] 

for loss_function in loss_function_Values:
    for schedule_function in schedule_function_values:
        for lift_constraint in lift_constraint_values:
            # Create model
            gate = SimpleGate(
                in_channels=64, 
                num_experts=4,
                top_k=2,
                use_noise=True,
                name='SimpleGate_' + loss_function,
                loss_fkt=loss_function,
                w_aux_loss=0.1
                )

            moe_layer = BottleneckMoeLayer(64, 256, num_experts=4, top_k=2, use_noise=True, gating_network=gate )
            model = DeepResNetMoE(num_experts=4, top_k=2, moe_layer=moe_layer, lift_constraint=lift_constraint,name='SimpleGate_' + loss_function + '_lift=' + str(lift_constraint))

            optimizer = optimizer_class(model.parameters(), learning_rate)
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule_function)

            # Train model with constraint
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
                wandb_project='analyze_deep_moe',
                wandb_name='ResNet101_MoE_SimpleGate_' + loss_function + "_w=0.1_liftConstrained=" + str(lift_constraint),
                wandb_tags=['MoE', 'ResNet101', 'SimpleGate', 'SoftConstraint', loss_function],
                wandb_checkpoints=20
            )