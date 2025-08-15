import torch
import numpy as np
import math

from models.moe_layer.resnet.bottleneck_moe_layer import BottleneckMoeLayer
from models.moe_layer.resnet.bottleneck_expert import ResNet101Expert
from models.moe_layer.soft_gating_networks import SimpleGateVarianceLoss
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

def schedule1(epoch):
    # if epoch <=70:
    #     return 1 + 9.0 / 70.0 * epoch
    if epoch <= 100:
        return 10
    if epoch < 60:
        return 1
    if epoch < 80:
        return 0.1
    return 0.1

    
# Fix parameters
num_epochs = 100
batch_size = 100
learning_rate = 0.0001
optimizer_class = torch.optim.Adam
schedule_function = schedule1
top_k=2
w_aux_loss = 0.5

# Variable parameters
loss_function_values = ['importance', 'kl_divergence']
position_values = [1]
in_channels = [64, 256, 512, 1024] 
w_variance_loss = 0.1

for loss_function in loss_function_values:
    for i in range(len(position_values)):

        # Create model
        gate = SimpleGateVarianceLoss(
            in_channels=in_channels[i], 
            w_variance_loss=w_variance_loss,
            num_experts=4,
            top_k=top_k,
            use_noise=True,
            name='SimpleGate_' + loss_function,
            loss_fkt=loss_function,
            w_aux_loss=w_aux_loss
            )
          
        expert_list = []
        for _ in range(4):
            expert_list.append(
                ResNet101Expert(layer_position=position_values[i])
            )
        experts = torch.nn.ModuleList(expert_list)
        
        moe_layer = BottleneckMoeLayer(
            num_experts=len(expert_list), 
            layer_position=position_values[i], 
            top_k=top_k,
            gating_network=gate)

        model = DeepResNetMoE(
            moe_layers=[moe_layer],
            name='DeepResNetMoE_moePosition=' + str(position_values[i]), 
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
            enable_logging=True,
            lr_scheduler=lr_scheduler,
            wandb_project='analyze_deep_moe',
            wandb_name='ResNet101_MoE_SimpleGate_' + loss_function + '_w=' + str(w_aux_loss) + '_w_var_loss=' + str(w_variance_loss) + '_moePosition=' + str(position_values[i]),
            wandb_tags=['MoE', 'ResNet101', 'SimpleGate', 'SoftConstraint', loss_function, 'VarianceLoss'],
            wandb_checkpoints=20
        )
