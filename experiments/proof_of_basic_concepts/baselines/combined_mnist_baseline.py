"""To compare the results, a baseline model is trained on the combined data swiss and default mnist (baseline_combined). 
Default MNIST data are downsampled to match the number of samples in the swiss MNIST dataset. 
Training data (28,112 samples) and test data (4,674 samples) are equally sized and merged into one set for training, validation and testing. 
The baseline model has the same architecture as the experts used in MoE architectures (MnistNet).
"""
import torch
import torchvision
import numpy as np

from models.mnist.mnist_net import MnistNet
from datasets.mnist_dataset import MNISTDataset
from utils.dataset_utils import train_test_split, get_transformation

# Set seeds
torch.manual_seed(42)
np.random.seed(42)

# Define transformations
transformations_mnist_default = get_transformation('mnist')
transformations_mnist_swiss = get_transformation('swiss_mnist')

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

# Train baselines
for i in range(3):
    model = MnistNet(classes=[i for i in range(10)],
                     name='baseline_mnist_combined ' + str(i))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(
        0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    model.fit(
        training_data=combined_train,
        validation_data=None,
        test_data={'combined_test': combined_test, 'default_test': mnist_default_test, 'swiss_test': mnist_swiss_test},
        num_epochs=25,
        batch_size=256,
        criterion=torch.nn.CrossEntropyLoss(),
        optimizer=optimizer,
        learning_rate=0.001,
        early_stopping=False,
        enable_logging=True,
        wandb_project='mnist_tests',
        wandb_checkpoints=None,
        wandb_name='MnistNet Baseline Combined ' + str(i),
        wandb_tags=['Baseline', 'MnistNet', 'MixedMNIST']
    )

    print('Evaluation on combined test')
    model.evaluate(combined_test)
    model.evaluate(mnist_default_test)
    model.evaluate(mnist_swiss_test)
