import torch

from datasets.cifar100_dataset import CIFAR100Dataset
from utils.dataset_utils import train_test_split, get_transformation
from models.moe_layer.resnet.shallow_resnet_moe import ShallowResNetMoE
from models.moe_layer.gating_networks import ConvGate, SimpleGate


def main():
    # Load training data and transformations
    transformations_training = get_transformation('cifar100', phase='training')
    transformations_test = get_transformation('cifar100', phase='test')
    cifar_data = CIFAR100Dataset(
        root_dir='/home/lb4653/mixture-of-experts-thesis/data/cifar100/training', transform=transformations_training)
    training_data, validation_data = train_test_split(cifar_data, [0.8, 0.2])
    validation_data.transform = transformations_test

    # Define fix training parameters
    num_exp = 4
    top_k = 2
    learning_rate = 0.0001
    batch_size = 512
    num_epochs = 100
    use_noise = True


    # Define varying training parameter
    w_importance_values = [0.001, 0.01, 0.1, 1, 10]
    gates = [ConvGate, SimpleGate]


    # # Training cycle
    for i in range(3):
        for w_importance in w_importance_values:
            for gate in gates:
                # Create model and optimizer
                model = ShallowResNetMoE(
                    num_experts=num_exp, top_k=top_k, w_importance=w_importance, use_noise=use_noise, num_blocks=1, gating_network=gate)
                optimizer = torch.optim.Adam(model.parameters(), learning_rate)
                # Set directory for TensorBoard logs
                logging_dir = '/home/lb4653/mixture-of-experts-thesis/experiments/resnet_moe_layer/logs/loss_weighting_tests/{}_weighting={}_test={}'.format(gate, w_importance, i)
                # Train model
                model.fit(
                    training_data=training_data,
                    validation_data=validation_data,
                    num_epochs=num_epochs,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    optimizer=optimizer,
                    early_stopping=False,
                    enable_logging=True,
                    logging_dir=logging_dir
                )
