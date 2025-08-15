import torch
import torchvision
import numpy as np

from models.mnist.mnist_net import MnistNet
from models.mnist.gating_network import FMGate, MnistNetGate
from models.mnist.generic_gating_network import GenericFMGate, GenericMnistNetGet
from datasets.mnist_dataset import MNISTDataset
from utils.dataset_utils import train_test_split, build_subset, get_transformation

from itertools import combinations

# Set seeds
torch.manual_seed(42)
np.random.seed(42)

# Define transformations
transformations_mnist_default = get_transformation('mnist')

# Load datasets
mnist_default_train_full = MNISTDataset('data/default_mnist/training', transform=transformations_mnist_default)
mnist_default_test_full = MNISTDataset('data/default_mnist/testing', transform=transformations_mnist_default)

training_sets = []
test_sets = []

batch_size=64

comb = combinations([i for i in range(10)], 2)
for i, labels in enumerate(comb):
    mnist_train_sub = MNISTDataset('data/default_mnist/training', transform=transformations_mnist_default, labels=labels)
    mnist_test_sub = MNISTDataset('data/default_mnist/testing', transform=transformations_mnist_default, labels=labels)
    training_sets.append(mnist_train_sub)
    test_sets.append(mnist_test_sub)


for run in range(3):
    experts = []
    for j, classes in enumerate(combinations([i for i in range(10)], 2)):
        expert = MnistNet(classes=classes, name='Label_expert_' + str(classes))
        expert.fit(
            training_data=training_sets[j],
            num_epochs=10,
            batch_size=batch_size,
            learning_rate=0.0001,
            enable_logging=False,
        )
        experts.append(expert)
    experts = torch.nn.ModuleList(experts)
    
    fmgate = FMGate(classes=[i for i in range(10)], experts=experts, name='CombinationSplit_FMGate_' + str(run))
    generic_fmgate = GenericFMGate(classes=[i for i in range(10)], experts=experts, name='CombinationSplit_GenericFMGate_' + str(run))
    
    mnistnetgate = MnistNetGate(classes=[i for i in range(10)], experts=experts, name='CombinationSplit_MnistNetGate_' + str(run))
    generic_mnistnetgate = GenericMnistNetGet(classes=[i for i in range(10)], experts=experts, name='CombinationSplit_GenericMnistGate_' + str(run))

    
    mnistnetgate.fit(
        training_data=mnist_default_train_full,
        test_data={'mnist_default_test_top':mnist_default_test_full},
        num_epochs=50,
        batch_size=batch_size,
        learning_rate=0.0001,
        enable_logging=False,
        wandb_project='mnist_tests',
        wandb_name='Combination Split MnistNetGate ' + str(run),
        wandb_tags=['MnistNet', 'CombinationSplit', 'Label Split'],
    )

    fmgate.fit(
        training_data=mnist_default_train_full,
        test_data={'mnist_default_test_full': mnist_default_test_full},
        num_epochs=50,
        batch_size=batch_size,
        learning_rate=0.0001,
        enable_logging=False,
        wandb_project='mnist_tests',
        wandb_name='Combination Split FMGate ' + str(run),
        wandb_tags=['MnistNet', 'CombinationSplit', 'Label Split'],
    )

    generic_mnistnetgate.fit(
        training_data=mnist_default_train_full,
        test_data={'mnist_default_test_full': mnist_default_test_full},
        num_epochs=75,
        batch_size=batch_size,
        learning_rate=0.0001,
        enable_logging=False,
        wandb_project='mnist_tests',
        wandb_name='Combination Split GenericMnistNetGate ' + str(run),
        wandb_tags=['MnistNet', 'CombinationSplit', 'Label Split', 'Generic Gate'],
    )

    generic_fmgate.fit(
        training_data=mnist_default_train_full,
        test_data={'mnist_default_test_full': mnist_default_test_full},
        num_epochs=75,
        batch_size=batch_size,
        learning_rate=0.0001,
        enable_logging=False,
        wandb_project='mnist_tests',
        wandb_name='Combination Split GenericFMGate ' + str(run),
        wandb_tags=['MnistNet', 'CombinationSplit', 'Label Split', 'Generic Gate'],
    )

