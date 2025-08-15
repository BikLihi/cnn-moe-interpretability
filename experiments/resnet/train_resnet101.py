import torch
import numpy as np
import math

from models.resnet.resnet101 import Resnet101
from datasets.cifar100_dataset import CIFAR100Dataset
from utils.dataset_utils import train_test_split, get_transformation
from utils.cifar100_utils import CIFAR100_LABELS


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
    if epoch <= 50:
        return 10
    if epoch < 100:
        return 1
    return 0.1

def schedule_2(epoch):
    return 1


def schedule_3(epoch):
    if epoch <= 100:
        return 10
    if epoch < 180:
        return 1
    return 0.1


def schedule_4(epoch):
    if epoch <= 100:
        return epoch
    return 100 - (epoch - 100) / 2.0



# Fix parameters
num_epochs = 200
batch_size = 128
learning_rate = 0.0001
optimizer_class = torch.optim.Adam


# Variable parameters
schedule_lambda_values = [schedule_1, schedule_2, schedule_3, schedule_4]


# Variable parameters
for schedule_lambda in schedule_lambda_values:
    model = Resnet101(classes=CIFAR100_LABELS)

    optimizer = optimizer_class(model.parameters(), learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule_lambda)

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
        wandb_name='ResNet101_Baseline',
        wandb_tags=['Baseline', 'ResNet101'],
    )

