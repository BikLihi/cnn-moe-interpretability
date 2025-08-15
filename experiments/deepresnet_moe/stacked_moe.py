import torch
import numpy as np
import math

from models.moe_layer.resnet.bottleneck_moe_layer import BottleneckMoeLayer
from models.moe_layer.resnet.bottleneck_expert import ResNet101Expert
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

def schedule1(epoch):
    if epoch <= 100:
        return 10
    if epoch < 150:
        return 1
    if epoch < 175:
        return 0.5
    return 0.1

    
# Fix parameters
num_epochs = 200
batch_size = 64
learning_rate = 0.0001
optimizer_class = torch.optim.Adam
schedule_function = schedule1
w_aux_loss = 0.5

# Variable parameters
loss_function_values = ['importance', 'kl_divergence']
position_values = [[1, 4]] #[[1, 4], [1, 2, 3, 4], [1, 2], [3, 4], [1, 4], [2, 3]]
in_channels = [64, 256, 512, 1024]
w_aux_loss = [0.5, 0.1, 0.1, 0.5]

for positions in position_values:
    for loss_function in loss_function_values:
        # Create model
        moe_layers = []
        for p in positions:
            gate = SimpleGate(
                in_channels=in_channels[p-1], 
                num_experts=4,
                top_k=2,
                use_noise=True,
                name='SimpleGate_' + loss_function + '_moePosition=' + str(p),
                loss_fkt=loss_function,
                w_aux_loss=w_aux_loss[p-1]
                )
                        
            moe_layer = BottleneckMoeLayer(
                num_experts=4, 
                layer_position=p,
                top_k=2,
                gating_network=gate)
            
            moe_layers.append(moe_layer)

        model = DeepResNetMoE(
            moe_layers=moe_layers,
            name='DeepResNetMoE_moePositions=' + str(positions), 
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
            wandb_name='ResNet101_MoE_SimpleGate_' + loss_function + '_w= ' + str(w_aux_loss) + '_moePosition=' + str(positions),
            wandb_tags=['MoE', 'ResNet101', 'SimpleGate', 'SoftConstraint', loss_function],
            wandb_checkpoints=20
        )
