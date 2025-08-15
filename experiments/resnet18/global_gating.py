import torch
import numpy as np
import math

from models.resnet.resnet18 import ResNet18
from models.moe_layer.resnet18.resnet18_global import ResNet18GlobalMoE, GlobalMoEGate, ComplexGlobalMoEGate, InputMoEGate
from models.moe_layer.resnet18.resnet18_experts import NarrowResNet18Expert
from datasets.cifar100_dataset import CIFAR100Dataset
from utils.dataset_utils import train_test_split, get_transformation
from utils.cifar100_utils import CIFAR100_LABELS
from ptflops import get_model_complexity_info
from torchsummary import summary


# Setting seeds
torch.manual_seed(42)
np.random.seed(42)

# Loading datasets
transformations_training = get_transformation('cifar100', phase='training')
transformations_test = get_transformation('cifar100', phase='test')
cifar_data = CIFAR100Dataset(
    root_dir='/home/lb4653/mixture-of-experts-thesis/data/cifar100/training', transform=transformations_training)
training_data, validation_data = train_test_split(
    cifar_data, [0.8, 0.2], seed=42)
validation_data.transform = transformations_test


def schedule1(epoch):
    if epoch <= 80:
        return 0.001
    if epoch < 120:
        return 0.0001
    return 0.00001


# Fix parameters
num_epochs = 150
batch_size = 128
learning_rate = 1
optimizer_class = torch.optim.Adam
schedule_function = schedule1
num_experts = [4]
w_var_losses = [0]
w_aux_losses = [0.5]
top_k = [2]
expert_classes = [NarrowResNet18Expert]
w_aux_loss = [0.5]
loss_fkts = ['kl_divergence']
# Variable parameters
layer_positions_values = [[1], [2], [3], [4], [1, 4]]


for layer_positions in layer_positions_values:
    model = ResNet18GlobalMoE(layer_positions=layer_positions, expert_classes=expert_classes, num_experts=num_experts, top_k=top_k,
                              loss_fkts=loss_fkts, w_aux_losses=w_aux_losses, w_var_losses=w_var_losses,
                              gating_network=InputMoEGate, name='InputGateMoE_' + str(loss_fkts[0]) + '_' + 'pos_' + str(layer_positions) + '_NarrowResNet18Expert'
                              )

    # Optimizer and lr scheduler
    optimizer = optimizer_class(model.parameters(), learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, schedule_function)

    model.fit(
        training_data=training_data,
        validation_data=validation_data,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        early_stopping=False,
        enable_logging=True,
        wandb_project='resnet18_moe',
        wandb_name='InputGateMoE_' +
        str(loss_fkts[0]) + '_' + 'pos_' + str(layer_positions) +
        '_' + str(expert_classes[0].__name__),
        wandb_tags=['MoE', 'ResNet18', 'InputGate'],
        wandb_checkpoints=None
    )
