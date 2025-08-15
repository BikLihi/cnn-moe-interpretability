import torch

from datasets.cifar100_dataset import CIFAR100Dataset
from utils.dataset_utils import train_test_split, get_transformation
from models.moe_layer.resnet.shallow_resnet_moe import ShallowResNetMoE
from models.moe_layer.gating_networks import ConvGateComplexNoise, SimpleGate


def main():
    # Load training data and transformations
    transformations_training = get_transformation('cifar100', phase='training')
    transformations_test = get_transformation('cifar100', phase='test')
    cifar_data = CIFAR100Dataset(
        root_dir='/home/lb4653/mixture-of-experts-thesis/data/cifar100/training', transform=transformations_training)
    training_data, validation_data = train_test_split(cifar_data, [0.8, 0.2])
    validation_data.transform = transformations_test

    # Define fix training parameters
    num_exp = 8
    top_k = 2
    w_importance = 0.1
    learning_rate = 0.0001
    batch_size = 256
    num_epochs = 10

    # Define varying training parameter
    use_noise_values = [False, True]

    # # Training cycle
    for i in range(3):
        for use_noise in use_noise_values:
            # Create model and optimizer
            model = ShallowResNetMoE(
                num_experts=num_exp, top_k=top_k, w_importance=w_importance, use_noise=use_noise, num_blocks=1, gating_network=ConvGate)
            optimizer = torch.optim.Adam(model.parameters(), learning_rate)
            # Set directory for TensorBoard logs
            if use_noise:
                logging_dir = '/home/lb4653/mixture-of-experts-thesis/experiments/resnet_moe_layer/logs/compare_noisy_gate/simple_noise_test_' + \
                    str(i)
            else:
                logging_dir = '/home/lb4653/mixture-of-experts-thesis/experiments/resnet_moe_layer/logs/compare_noisy_gate/no_noise_test_' + \
                    str(i)
            # Train model
            model.fit(
                training_data=training_data,
                validation_data=validation_data,
                num_epochs=num_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                lr_Scheduler=None,
                optimizer=optimizer,
                early_stopping=False,
                enable_logging=True,
                logging_dir=logging_dir,
            )

    for i in range(3):
        # Create model and optimizer
        model = ShallowResNetMoE(
            num_experts=num_exp, top_k=top_k, w_importance=w_importance, use_noise=use_noise, num_blocks=1, gating_network=ConvGateComplexNoise)
        optimizer = torch.optim.Adam(model.parameters(), learning_rate)
        # Set directory for TensorBoard logs
        logging_dir = '/home/lb4653/mixture-of-experts-thesis/experiments/resnet_moe_layer/logs/compare_noisy_gate/complex_noise_test_' + \
            str(i)
        # Train model
        model.fit(
            training_data=training_data,
            validation_data=validation_data,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            lr_Scheduler=None,
            optimizer=optimizer,
            early_stopping=False,
            enable_logging=True,
            logging_dir=logging_dir,
        )

    # Test simple gate
    for i in range(1, 3):
        # Create model and optimizer
        for use_noise in use_noise_values:
            model = ShallowResNetMoE(
                num_experts=num_exp, top_k=top_k, w_importance=w_importance, use_noise=use_noise, num_blocks=1, gating_network=SimpleGate)
            optimizer = torch.optim.Adam(model.parameters(), learning_rate)
            # Set directory for TensorBoard logs
            logging_dir = '/home/lb4653/mixture-of-experts-thesis/experiments/resnet_moe_layer/logs/compare_noisy_gate/simple_gate_noise=' + \
                str(use_noise) + '_test_' + str(i)
            # Train model
            model.fit(
                training_data=training_data,
                validation_data=validation_data,
                num_epochs=num_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                lr_Scheduler=None,
                optimizer=optimizer,
                early_stopping=False,
                enable_logging=True,
                logging_dir=logging_dir,
            )


if __name__ == '__main__':
    main()
