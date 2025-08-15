import torch
import numpy as np
import math

from models.moe_layer.resnet18.resnet18_multi_moe import ResNet18MultiMoE
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

# Create model
model = ResNet18MultiMoE(
    moe_layer_positions=[1, 2, 3, 4],
    name='ResNet18MultiMoE_FirstLayer_' + str([1, 2, 3, 4])
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
    return_best_model=False,
    lr_scheduler=lr_scheduler,
    wandb_project='resnet18_moe',
    wandb_name='ResNet18_MultiMoE_FirstLayers',
    wandb_tags=['MultiMoE', 'ResNet18'],
    wandb_checkpoints=50
)
