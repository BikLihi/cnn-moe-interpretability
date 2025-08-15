"""A baseline model (baseline_default) is trained only on mnist data using the full dataset for training (60,000 samples) and testing (10,000 samples) 
to make it comparable in accordance with MoE models using only the default mnist data (without swiss mnist). The model has the same architecture as the experts (MnistNet). 
"""
import torch
import torchvision
import numpy as np

from models.mnist.mnist_net import MnistNet
from datasets.mnist_dataset import MNISTDataset
from utils.dataset_utils import train_test_split, build_subset, get_transformation


# Set seeds
torch.manual_seed(42)
np.random.seed(42)

# Define transformations
transformations_mnist_default = get_transformation('mnist')
transformations_mnist_swiss = get_transformation('swiss_mnist')

# Load datasets
mnist_default_train = MNISTDataset('data/default_mnist/training', transform=transformations_mnist_default)
mnist_default_test = MNISTDataset('data/default_mnist/testing', transform=transformations_mnist_default)

mnist_swiss_train = MNISTDataset('data/swiss_mnist/training', transform=transformations_mnist_swiss)
mnist_swiss_test = MNISTDataset('data/swiss_mnist/testing', transform=transformations_mnist_swiss)

# Train baselines
for i in range(3):
    default_model = MnistNet(classes=[i for i in range(10)], name='baseline_mnist_default ' + str(i))
    optimizer = torch.optim.Adam(default_model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    default_model.fit(
        training_data=mnist_default_train,
        validation_data=None,
        test_data={'default_test': mnist_default_test, 'swiss_test': mnist_swiss_test},
        num_epochs=25,
        batch_size=256,
        criterion=torch.nn.CrossEntropyLoss(),
        optimizer=optimizer,
        learning_rate=0.001,
        early_stopping=False,
        enable_logging=True,
        wandb_project='mnist_tests',
        wandb_checkpoints=None,
        wandb_name='MnistNet Baseline Default ' + str(i),
        wandb_tags=['Baseline', 'MnistNet', 'Default Mnist'],
    )

    swiss_model = MnistNet(classes=[i for i in range(10)], name='baseline_mnist_swiss ' + str(i))
    optimizer = torch.optim.Adam(swiss_model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    swiss_model.fit(
        training_data=mnist_swiss_train,
        validation_data=None,
        test_data={'default_test': mnist_default_test, 'swiss_test': mnist_swiss_test},
        num_epochs=25,
        batch_size=256,
        criterion=torch.nn.CrossEntropyLoss(),
        optimizer=optimizer,
        learning_rate=0.001,
        early_stopping=False,
        enable_logging=True,
        wandb_project='mnist_tests',
        wandb_checkpoints=None,
        wandb_name='MnistNet Baseline Swiss ' + str(i),
        wandb_tags=['Baseline', 'MnistNet', 'Swiss Mnist'],
    )