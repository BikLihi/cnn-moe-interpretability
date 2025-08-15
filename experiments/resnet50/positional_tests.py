import torch
import numpy as np
import math

from models.moe_layer.resnet.moe_block_layer import MoeBlockLayer
from models.moe_layer.resnet.bottleneck_expert import ShallowResNet50Expert
from models.moe_layer.resnet50.resnet50_moe import ResNet50MoE
from models.moe_layer.soft_gating_networks import SimpleGate
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

def schedule1(epoch):
    if epoch <= 100:
        return 0.001
    if epoch < 130:
        return 0.0001
    return 0.00001
    
# Fix parameters
num_epochs = 150
batch_size = 128
learning_rate = 1
optimizer_class = torch.optim.Adam
schedule_function = schedule1
top_k=2
w_aux_loss = 0.5
num_experts=4

# Variable parameters
loss_function_values = ['importance', 'kl_divergence']
position_values = [0] 
in_channels = [3]

for loss_function in loss_function_values:
    for i in range(len(position_values)):
        # Create model
        gate = SimpleGate(
            in_channels=in_channels[i], 
            num_experts=num_experts,
            top_k=top_k,
            use_noise=True,
            name='SimpleGate_' + loss_function,
            loss_fkt=loss_function,
            w_aux_loss=w_aux_loss
            )
                
        moe_layer = BottleneckMoeLayer(
            num_experts=num_experts, 
            layer_position=position_values[i], 
            top_k=top_k,
            gating_network=gate,
            resnet_expert=ShallowResNet50Expert)

        model = ResNet50MoE(
            moe_layers=[moe_layer],
            name='ResNet50_Input_' + loss_function + '_w=' + str(w_aux_loss) + '_moePosition=' + str(position_values[i]),
            lift_constraint=None
        )

        # Optimizer and lr scheduler
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
            enable_logging=False,
            lr_scheduler=lr_scheduler,
            wandb_project='resnet50_moe',
            wandb_name='ResNet50_MoE_Input' + loss_function + '_w=' + str(w_aux_loss) + '_moePosition=' + str(position_values[i]),
            wandb_tags=['MoE', 'ResNet50', 'SimpleGate', 'SoftConstraint', loss_function],
            wandb_checkpoints=25
        )
