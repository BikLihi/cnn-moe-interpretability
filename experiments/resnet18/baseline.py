import torch
import numpy as np
import math

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


def schedule1(epoch):
    if epoch <= 60:
        return 0.001
    if epoch < 100:
        return 0.0001
    return 0.00001
    

# Fix parameters
num_epochs = 150
batch_size = 128
learning_rate = 1
optimizer_class = torch.optim.Adam

for j, schedule_function in enumerate([schedule1]):
    for i in range(0, 3):
        model = ResNet18(CIFAR100_LABELS, name='Baseline_' + str(i) + '_schedule_' + str(j))

        # Optimizer and lr scheduler
        optimizer = optimizer_class(model.parameters(), learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule_function)

        # Train model with constraint
        model.fit(
            training_data=training_data,
            test_data=test_data,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            optimizer=optimizer,
            early_stopping=False,
            enable_logging=True,
            return_best_model=False,
            lr_scheduler=lr_scheduler,
            wandb_project='final_resnet_18',
            wandb_name='Baseline_' + str(i) + '_schedule_' + str(j),
            wandb_tags=['ResNet18', 'Baseline'],
            wandb_checkpoints=50
        )
