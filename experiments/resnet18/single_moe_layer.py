import torch
import numpy as np
import math

from models.moe_layer.resnet.moe_conv_layer import MoEConvLayer
from models.moe_layer.resnet18.resnet18_single_layer import ResNet18SingleLayer
from models.moe_layer.soft_gating_networks import SimpleGate
from models.moe_layer.hard_gating_networks import RelativeImportanceGate, AbsoluteImportanceGate
from models.resnet.resnet18 import ResNet18
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
    if epoch <= 100:
        return 0.001
    if epoch < 150:
        return 0.0001
    return 0.00001


# Fix parameters
num_epochs = 180
batch_size = 128
learning_rate = 1
optimizer_class = torch.optim.Adam
schedule_function = schedule1
w_aux_loss = 0.0
constr_threshold = 0.3
gating_network = AbsoluteImportanceGate
# Variable parameters
loss_function = None
num_expert_values = [10]
top_k = 2
position_values = [0]

for num_experts in num_expert_values:
    for position in position_values:
            for i in range(3):
                name='SingleLayer_' + str(num_experts) + '_topK=' + str(top_k) + '_' + gating_network.__name__ + '_constr=' + str(constr_threshold) + '_moePosition=' + str(position) + '_' + str(i)
                model = ResNet18SingleLayer(position, num_experts, top_k, gating_network, loss_function, w_aux_loss, constr_threshold, name)

                # Optimizer and lr scheduler
                optimizer = optimizer_class(model.parameters(), learning_rate)
                lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                    optimizer, schedule_function)

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
                    wandb_project='final_hardConstr_resnet18',
                    wandb_name=name,
                    wandb_tags=['MoE', 'ResNet18', gating_network.__name__,
                                'HardConstraint', 'Single Layer'],
                )
