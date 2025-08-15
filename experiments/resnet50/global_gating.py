import torch
import numpy as np
import math

from models.resnet.resnet50 import ResNet50
from models.moe_layer.resnet50.resnet50_global import ResNet50GlobalMoE, GlobalMoEGate
from models.moe_layer.resnet.bottleneck_expert import NarrowResNet50Expert, ShallowResNet50Expert
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
num_experts = [4, 4]
layer_positions = [1, 2]
top_k = [2, 2]
w_var_losses = [0.001, 0.001]
w_aux_losses = [0.5, 0.5]

# Variable parameters
loss_function_values = [['importance', 'importance'], ['kl_divergence', 'kl_divergence']]
expert_classes_values = [[NarrowResNet50Expert, NarrowResNet50Expert]]

for loss_fkts in loss_function_values:
    for expert_classes in expert_classes_values:

        model = ResNet50GlobalMoE(layer_positions=layer_positions, expert_classes=expert_classes, num_experts=num_experts, top_k=top_k,
                                  loss_fkts=loss_fkts, w_aux_losses=w_aux_losses, w_var_losses=w_var_losses,
                                  gating_network=GlobalMoEGate, name='GlobalGating_' + str(loss_fkts) + '_' + str(expert_classes[0].__name__))

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
            enable_logging=False,
            wandb_project='resnet50_moe',
            wandb_name='GlobalMoE_' +
            str(loss_fkts) + '_' +  str(expert_classes[0].__name__),
            wandb_tags=['MoE', 'ResNet50', 'GlobalGating'],
            wandb_checkpoints=None
        )
