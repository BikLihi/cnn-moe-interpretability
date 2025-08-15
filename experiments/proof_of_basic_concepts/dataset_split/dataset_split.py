import torch
import torchvision
import numpy as np

from models.mnist.mnist_net import MnistNet
from datasets.mnist_dataset import MNISTDataset
from utils.dataset_utils import train_test_split, get_transformation
from models.mnist.gating_network import FMGate, MnistNetGate

# Set seeds
torch.manual_seed(42)
np.random.seed(42)

# Define transformations
transformations_mnist_default = get_transformation('mnist')
transformations_mnist_swiss = get_transformation('swiss_mnist')

# Load datasets
mnist_default_train = MNISTDataset(
    'data/default_mnist/training', transform=transformations_mnist_default)
mnist_default_test = MNISTDataset(
    'data/default_mnist/testing', transform=transformations_mnist_default)

mnist_swiss_train = MNISTDataset(
    'data/swiss_mnist/training', transform=transformations_mnist_swiss)
mnist_swiss_test = MNISTDataset(
    'data/swiss_mnist/testing', transform=transformations_mnist_swiss)

# Subsample default MNIST
mnist_default_train, _ = train_test_split(mnist_default_train, proportions=[
                                          len(mnist_swiss_train) / len(mnist_default_train), 0], seed=42)
mnist_default_test, _ = train_test_split(mnist_default_test, proportions=[
                                         len(mnist_swiss_test) / len(mnist_default_test), 0], seed=42)

# Combine datasets
combined_train = torch.utils.data.ConcatDataset([mnist_swiss_train, mnist_default_train])
combined_test = torch.utils.data.ConcatDataset([mnist_swiss_test, mnist_default_test])


for i in range(3):
    swiss_expert = MnistNet(classes=[i for i in range(10)], name='swiss_expert ' + str(i))
    default_expert = MnistNet(classes=[i for i in range(10)], name='mnist_expert' + str(i))
    fmgate = FMGate(classes=[i for i in range(10)], experts=[swiss_expert, default_expert], name='FMNet ' + str(i) )
    mnistnetgate = MnistNetGate(classes=[i for i in range(10)], experts=[swiss_expert, default_expert], name='MnistNetGate ' + str(i) )

    swiss_expert.fit(
        training_data=mnist_swiss_train,
        test_data={'combined_test': combined_test, 'default_test': mnist_default_test, 'swiss_test': mnist_swiss_test},
        num_epochs=25,
        batch_size=256,
        learning_rate=0.001,
        enable_logging=True,
        wandb_project='mnist_tests',
        wandb_name='MnistNet Swiss Expert ' + str(i),
        wandb_tags=['MnistNet', 'MixedMNIST', 'Dataset Split'],
    )

    default_expert.fit(
        training_data=mnist_default_train,
        test_data={'combined_test': combined_test, 'default_test': mnist_default_test, 'swiss_test': mnist_swiss_test},
        num_epochs=25,
        batch_size=256,
        learning_rate=0.001,
        enable_logging=True,
        wandb_project='mnist_tests',
        wandb_name='MnistNet Default Expert ' + str(i),
        wandb_tags=['MnistNet', 'MixedMNIST', 'Dataset Split'],
    )

    
    fmgate.fit(
        training_data=combined_train,
        test_data={'combined_test': combined_test, 'default_test': mnist_default_test, 'swiss_test': mnist_swiss_test},
        num_epochs=25,
        batch_size=256,
        learning_rate=0.0001,
        enable_logging=True,
        wandb_project='mnist_tests',
        wandb_name='MnistNet Dataset Split FMGate ' + str(i),
        wandb_tags=['MnistNet', 'MixedMNIST', 'Dataset Split', 'FMGate']
    )

    mnistnetgate.fit(
        training_data=combined_train,
        test_data={'combined_test': combined_test, 'default_test': mnist_default_test, 'swiss_test': mnist_swiss_test},
        num_epochs=25,
        batch_size=256,
        learning_rate=0.0001,
        enable_logging=True,
        wandb_project='mnist_tests',
        wandb_name='MnistNet Dataset Split MnistNetGate ' + str(i),
        wandb_tags=['MnistNet', 'MixedMNIST', 'Dataset Split', 'MnmistNetGate']
    )

