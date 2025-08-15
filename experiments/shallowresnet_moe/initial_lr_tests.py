import torch

from datasets.cifar100_dataset import CIFAR100Dataset
from utils.dataset_utils import train_test_split, get_transformation
from models.moe_layer.gating_networks import ConvGate, SimpleGate, HardConstraintGate
from models.moe_layer.resnet.shallow_resnet_moe import ShallowResNetMoE


def main():
    # Load training data and respective transformation
    transformations_training = get_transformation('cifar100', phase='training')
    transformations_test = get_transformation('cifar100', phase='test')

    cifar_data = CIFAR100Dataset(
        root_dir='/home/lb4653/mixture-of-experts-thesis/data/cifar100/training', transform=transformations_training)
    training_data, validation_data, = train_test_split(cifar_data, [0.8, 0.2])
    validation_data.transform = transformations_test

    # Define fix training parameter
    w_importance = 0.1
    batch_size = 512
    num_epochs = 30
    use_noise = True
    num_blocks = 1

    # Define varying training parameter
    num_expert_values = [4]
    top_k_values = [2]
    gate_values = [ConvGate, SimpleGate, HardConstraintGate]
    learning_rate_values = [0.01, 0.005, 0.001]



    for num_experts in num_expert_values:
        for top_k in top_k_values:
            for gate in gate_values:
                for learning_rate in learning_rate_values:
                    for i in range(3):

                        model = ShallowResNetMoE(
                            num_experts=num_experts,
                            top_k=top_k,
                            w_importance=w_importance,
                            use_noise=use_noise,
                            num_blocks=num_blocks,
                            gating_network=gate,
                            name='{}_test={}_numExp={}_topk={}_lr={}_noise={}'.format(
                                gate.__qualname__, i, num_experts, top_k, learning_rate, use_noise)
                        )

                        optimizer = torch.optim.Adam(model.parameters(), learning_rate)

                        logging_dir = '/home/lb4653/mixture-of-experts-thesis/experiments/resnet_moe_layer/logs/initial_lr_tests/{}_test={}_numExp={}_topk={}_lr={}_noise={}'.format(
                            gate.__qualname__, i, num_experts, top_k, learning_rate, use_noise)

                        model.fit(
                            training_data=training_data,
                            validation_data=validation_data,
                            num_epochs=num_epochs,
                            batch_size=batch_size,
                            learning_rate=learning_rate,
                            optimizer=optimizer,
                            early_stopping=False,
                            enable_logging=True,
                            logging_dir=logging_dir,
                        )
                        
