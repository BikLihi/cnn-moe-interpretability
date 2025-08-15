import torch
import torchvision

from models.resnet.restnet18 import Resnet18
from datasets.cifar100_dataset import CIFAR100Dataset
from utils.cifar100_utils import CIFAR100_LABELS
from torchsummary import summary
from utils.dataset_utils import train_test_split, get_transformation
from torch.utils.tensorboard import SummaryWriter
from metrics.top_k_accuracy import TopKAccuracy


def train_resnet18_models():
    # Set seed
    torch.manual_seed(42)

    # Define transformations
    transformations_training = get_transformation('cifar100', phase='training')
    transformations_test = get_transformation('cifar100', phase='test')

    # Load Dataset
    cifar_data = CIFAR100Dataset(
        root_dir='./data/cifar100/training', transform=transformations_training)
    training_data, validation_data = train_test_split(cifar_data, [0.8, 0.2])
    validation_data.transform = transformations_test
    test_data = CIFAR100Dataset(
        root_dir='./data/cifar100/testing', transform=transformations_test)

    # Grid Search Parameters
    learning_rates = [0.001, 0.01, 0.05, 0.1]
    optimizer = [torch.optim.SGD, torch.optim.Adam]
    batch_sizes = [32, 64, 128, 256]

    # Define metrics
    metrics = []
    metrics.append(TopKAccuracy(top_k=5))

    for lr in learning_rates:
        for opt in optimizer:
            for batch_size in batch_sizes:
                model = Resnet18(classes=CIFAR100_LABELS, name='ReNet18_Cifar100_lr={}_bs={}_opt={}'.format(
                    lr, batch_size, opt.__name__))
                model.fit(
                    training_data=training_data,
                    validation_data=validation_data,
                    num_epochs=1,
                    batch_size=batch_size,
                    learning_rate=lr,
                    optimizer=opt,
                    metrics=metrics,
                    early_stopping=True,
                    enable_logging=False,
                    logging_dir='runs/resnet_baseline/model_lr={}_bs={}_opt={}'.format(
                        lr, batch_size, opt.__name__)
                )


train_resnet18_models()