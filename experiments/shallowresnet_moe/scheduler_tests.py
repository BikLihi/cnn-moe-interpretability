import torch

from datasets.cifar100_dataset import CIFAR100Dataset
from utils.dataset_utils import train_test_split, get_transformation
from models.moe_layer.gating_networks import ConvGate, SimpleGate
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
    learning_rate = 0.0001
    batch_size = 512
    num_epochs = 100
    use_noise = True
    num_blocks = 1

    # Define varying training parameter
    num_expert_values = [4]
    top_k_values = [2]
    gate_values = [ConvGate, SimpleGate]
    lambda_values = [lambda epoch: epoch, lambda epoch: epoch // 2 + 1, lambda epoch: epoch // 5 + 1, lambda epoch: epoch * 5]
    for num_experts in num_expert_values:
        for top_k in top_k_values:
            for gate in gate_values:
                for scheduler_num, lambda_lr in enumerate(lambda_values):
                    for i in range(1):

                        model = ShallowResNetMoE(
                            num_experts=num_experts,
                            top_k=top_k,
                            w_importance=w_importance,
                            use_noise=use_noise,
                            num_blocks=num_blocks,
                            gating_network=gate
                        )

                        optimizer = torch.optim.Adam(model.parameters(), learning_rate)
                        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda_lr)

                        logging_dir = '/home/lb4653/mixture-of-experts-thesis/experiments/resnet_moe_layer/logs/scheduler_tests/{}_test={}_numExp={}_topk={}_sched={}_noise={}'.format(
                            gate.__qualname__, i, num_experts, top_k, scheduler_num, use_noise)
                        save_state_path = '/home/lb4653/mixture-of-experts-thesis/experiments/resnet_moe_layer/trained_models/scheduler_tests/{}_test={}_numExp={}_topk={}_sched={}_noise={}'.format(
                            gate.__qualname__, i, num_experts, top_k, scheduler_num, use_noise)

                        try:
                            model.fit(
                                training_data=training_data,
                                validation_data=validation_data,
                                num_epochs=num_epochs,
                                batch_size=batch_size,
                                learning_rate=learning_rate,
                                lr_Scheduler=lr_scheduler,
                                optimizer=optimizer,
                                early_stopping=False,
                                enable_logging=True,
                                logging_dir=logging_dir,
                                save_state_path=save_state_path
                            )
                        
                        except:
                            pass
