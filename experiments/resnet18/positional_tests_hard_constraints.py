import torch
import numpy as np
import math

from models.moe_layer.resnet.moe_block_layer import MoeBlockLayer, ResidualMoeBlockLayer
from models.moe_layer.resnet18.resnet18_experts import NarrowResNet18Expert
from models.moe_layer.resnet18.resnet18_moe import ResNet18MoE
from models.moe_layer.soft_gating_networks import SimpleGate
from models.moe_layer.hard_gating_networks import RelativeImportanceGate, AbsoluteImportanceGate, RelativeImportanceGatConv
from datasets.cifar100_dataset import CIFAR100Dataset
from utils.dataset_utils import train_test_split, get_transformation
from utils.cifar100_utils import CIFAR100_LABELS


# Setting seeds
torch.manual_seed(42)
np.random.seed(42)

# Loading datasets
transformations_training = get_transformation('cifar100', phase='training')
transformations_test = get_transformation('cifar100', phase='test')
training_data = CIFAR100Dataset(
    root_dir='/home/lb4653/mixture-of-experts-thesis/data/cifar100/training', transform=transformations_training)
test_data = CIFAR100Dataset(
    root_dir='/home/lb4653/mixture-of-experts-thesis/data/cifar100/testing', transform=transformations_test)

# def schedule1(epoch):
#     if epoch <= 80:
#         return 0.001
#     if epoch < 120:
#         return 0.0001
#     return 0.00001

def schedule1(epoch):
    if epoch < 20:
       return 4.5e-5 * epoch + 4.5e-5
    if epoch <= 80:
        return 0.001
    if epoch < 120:
        return 0.0001
    return 0.00001

# def schedule1(epoch):
#     if epoch <= 100:
#         return 0.001
#     if epoch < 150:
#         return 0.0001
#     return 0.00001
    
# Fix parameters
num_epochs = 150
batch_size = 128
learning_rate = 1
optimizer_class = torch.optim.Adam
schedule_function = schedule1
constr_threshold = 0.5
in_channels = [64, 64, 128, 256]
gating_network = RelativeImportanceGatConv
# Variable parameters
num_expert_values = [4]
top_k = 2
position_values = [4]

for num_experts in num_expert_values:
    for position in position_values:
        for i in range(3):
                # Create model
                gate = gating_network(
                    in_channels=in_channels[position-1], 
                    num_experts=num_experts,
                    top_k=top_k,
                    use_noise=True,
                    name=gating_network.__name__,
                    constr_threshold=constr_threshold
                    )
                        
                moe_layer = ResidualMoeBlockLayer(
                    num_experts=num_experts, 
                    layer_position=position, 
                    top_k=top_k,
                    gating_network=gate,
                    resnet_expert=NarrowResNet18Expert)

                model = ResNet18MoE(
                    moe_layers=[moe_layer],
                    name='Residual_' + str(num_experts) + '_topK=' + str(top_k) + '_' + gating_network.__name__ + '_constr=' + str(constr_threshold) + '_moePosition=' + str(position) + '_' + str(i),
                    lift_constraint=None
                )

                # Optimizer and lr scheduler
                optimizer = optimizer_class(model.parameters(), learning_rate)
                lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule_function)

                # Train model with constraint
                model.fit(
                    training_data=training_data,
                    validation_data=test_data,
                    test_data=test_data,
                    num_epochs=num_epochs,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    optimizer=optimizer,
                    early_stopping=False,
                    enable_logging=True,
                    return_best_model=False,
                    lr_scheduler=lr_scheduler,
                    wandb_project='resnet_18_gating_complexity',
                    wandb_name='Residual_' + str(num_experts) + '_topK=' + str(top_k) + '_' + gating_network.__name__ + '_constr=' + str(constr_threshold) + '_moePosition=' + str(position) + '_' + str(i),
                    wandb_tags=['MoE', 'ResNet18', gating_network.__name__, 'HardConstraint', 'Residual'],
                )
