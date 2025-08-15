import torch
import torchvision
import numpy as np

from models.mnist.mnist_net import MnistNet
from models.mnist.gating_network import FMGate, MnistNetGate
from models.mnist.generic_gating_network import GenericFMGate, GenericMnistNetGet
from datasets.mnist_dataset import MNISTDataset
from utils.plot_confusion_matrix import plot_confusion_matrix
from utils.dataset_utils import train_test_split, build_subset, get_transformation
from torchsummary import summary

# Set seeds
torch.manual_seed(42)
np.random.seed(42)

# Define transformations
transformations_mnist_default = get_transformation('mnist')

# Load datasets
mnist_default_train_full = MNISTDataset('data/default_mnist/training', transform=transformations_mnist_default)
mnist_default_train_bottom = MNISTDataset('data/default_mnist/training', transform=transformations_mnist_default, labels=[i for i in range(5)])
mnist_default_train_top = MNISTDataset('data/default_mnist/training', transform=transformations_mnist_default, labels=[i for i in range(5, 10)])

mnist_default_test_full = MNISTDataset('data/default_mnist/testing', transform=transformations_mnist_default)
mnist_default_test_bottom = MNISTDataset('data/default_mnist/testing', transform=transformations_mnist_default, labels=[i for i in range(5)])
mnist_default_test_top = MNISTDataset('data/default_mnist/testing', transform=transformations_mnist_default, labels=[i for i in range(5, 10)])


for i in range(3):
    bottom_expert = MnistNet(classes=[i for i in range(5)], name='BinSplit_Bottom_Expert_' + str(i))
    top_expert = MnistNet(classes=[i for i in range(5, 10)], name='BinSplit_Top_Expert_' + str(i))
    
    fmgate = FMGate(classes=[i for i in range(10)], experts=[top_expert, bottom_expert], name='BinSplit_FMGate_' + str(i))
    generic_fmgate = GenericFMGate(classes=[i for i in range(10)], experts=[top_expert, bottom_expert], name='BinSplit_GenericFMGate_' + str(i))
    
    mnistnetgate = MnistNetGate(classes=[i for i in range(10)], experts=[top_expert, bottom_expert], name='BinSplit_MnistNetGate_' + str(i))
    generic_mnistnetgate = GenericMnistNetGet(classes=[i for i in range(10)], experts=[top_expert, bottom_expert], name='BinSplit_GenericMnistGate_' + str(i))

    top_expert.fit(
        training_data=mnist_default_train_top,
        test_data=mnist_default_test_top,
        num_epochs=25,
        batch_size=256,
        learning_rate=0.0001,
        enable_logging=True,
        wandb_project='mnist_tests',
        wandb_name='BinSplit Bottom Expert ' + str(i),
        wandb_tags=['MnistNet', 'BinSplit', 'Label Split'],
    )

    bottom_expert.fit(
        training_data=mnist_default_train_bottom,
        test_data=mnist_default_test_bottom,
        num_epochs=25,
        batch_size=256,
        learning_rate=0.0001,
        enable_logging=True,
        wandb_project='mnist_tests',
        wandb_name='BinSplit Top Expert ' + str(i),
        wandb_tags=['MnistNet', 'BinSplit', 'Label Split'],
    )

    mnistnetgate.fit(
        training_data=mnist_default_train_full,
        test_data={
            'mnist_default_test_full': mnist_default_test_full, 
            'mnist_default_test_bottom':mnist_default_test_bottom, 
            'mnist_default_test_top':mnist_default_test_top},
        num_epochs=25,
        batch_size=256,
        learning_rate=0.0001,
        enable_logging=True,
        wandb_project='mnist_tests',
        wandb_name='BinSplit MnistNetGate ' + str(i),
        wandb_tags=['MnistNet', 'BinSplit', 'Label Split'],
    )

    fmgate.fit(
        training_data=mnist_default_train_full,
        test_data={
            'mnist_default_test_full': mnist_default_test_full, 
            'mnist_default_test_bottom':mnist_default_test_bottom, 
            'mnist_default_test_top':mnist_default_test_top},
        num_epochs=25,
        batch_size=256,
        learning_rate=0.0001,
        enable_logging=True,
        wandb_project='mnist_tests',
        wandb_name='BinSplit FMGate ' + str(i),
        wandb_tags=['MnistNet', 'BinSplit', 'Label Split'],
    )

    generic_mnistnetgate.fit(
        training_data=mnist_default_train_full,
        test_data={
            'mnist_default_test_full': mnist_default_test_full, 
            'mnist_default_test_bottom':mnist_default_test_bottom, 
            'mnist_default_test_top':mnist_default_test_top},
        num_epochs=50,
        batch_size=256,
        learning_rate=0.0001,
        enable_logging=True,
        wandb_project='mnist_tests',
        wandb_name='BinSplit GenericMnistNetGate ' + str(i),
        wandb_tags=['MnistNet', 'BinSplit', 'Label Split', 'Generic Gate'],
    )

    generic_fmgate.fit(
        training_data=mnist_default_train_full,
        test_data={
            'mnist_default_test_full': mnist_default_test_full, 
            'mnist_default_test_bottom':mnist_default_test_bottom, 
            'mnist_default_test_top':mnist_default_test_top},
        num_epochs=50,
        batch_size=256,
        learning_rate=0.0001,
        enable_logging=True,
        wandb_project='mnist_tests',
        wandb_name='BinSplit GenericFMGate ' + str(i),
        wandb_tags=['MnistNet', 'BinSplit', 'Label Split', 'Generic Gate'],
    )
