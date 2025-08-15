import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import math
import wandb
import matplotlib.pyplot as plt


from models.moe_layer.resnet.moe_block_layer import MoeBlockLayer, ResidualMoeBlockLayer
from models.moe_layer.resnet18.resnet18_experts import NarrowResNet18Expert
from models.moe_layer.resnet18.resnet18_moe import ResNet18MoE
from models.moe_layer.soft_gating_networks import SimpleGate
from models.moe_layer.hard_gating_networks import RelativeImportanceGate, AbsoluteImportanceGate
from models.moe_layer.static_gating_networks import EqualWeightGatingNetwork, SingleWeightingGatingNetwork
from datasets.cifar100_dataset import CIFAR100Dataset
from utils.dataset_utils import train_test_split, get_transformation
from utils.cifar100_utils import CIFAR100_LABELS

from losses.importance_loss import importance_loss


def build_model(moe_variant, constraint, num_experts, pos, run_path, filename):
    in_channels = [64, 64, 128, 256]

    if moe_variant == 'Residual':
        moe_layer = ResidualMoeBlockLayer
    elif moe_variant == 'Default':
        moe_layer = MoeBlockLayer
    else:
        raise ValueError('Invalid model')

    if constraint == 'importance':
        gate = SimpleGate(
            in_channels=in_channels[pos-1],
            num_experts=num_experts,
            top_k=2,
            use_noise=True,
            name='SimpleGate',
            loss_fkt='importance',
            w_aux_loss=0.5
        )
    elif constraint == 'kl':
        gate = SimpleGate(
            in_channels=in_channels[pos-1],
            num_experts=num_experts,
            top_k=2,
            use_noise=True,
            name='SimpleGate',
            loss_fkt='kl_divergence',
            w_aux_loss=0.5
        )

    elif constraint == 'relative':
        gate = RelativeImportanceGate(
            in_channels=in_channels[pos-1],
            num_experts=num_experts,
            top_k=2,
            use_noise=True,
            name='relative Importance',
            constr_threshold=0.5
        )

    elif constraint == 'mean':
        gate = AbsoluteImportanceGate(
            in_channels=in_channels[pos-1],
            num_experts=num_experts,
            top_k=2,
            use_noise=True,
            name='mean Importance',
            constr_threshold=0.3
        )

    else:
        raise ValueError('Invalid Constraint')

    moe_block = moe_layer(
        num_experts=num_experts,
        layer_position=pos,
        top_k=2,
        gating_network=gate,
        resnet_expert=NarrowResNet18Expert)

    name = str(moe_variant) + '_' + str(constraint) + '_' + str(num_experts)

    model = ResNet18MoE(
        moe_layers=[moe_block],
        name=name
    )

    # Load model weights
    file_model = wandb.restore(
        filename, run_path=run_path, root='./model_weights')

    # # Load parameters
    model.load_state_dict(torch.load(file_model.name)['model_state_dict'])

    # # Move model to cuda
    model.to(model.device)

    print('Load model: ' + filename)

    return model, moe_block


def varying_k_accuracy(model, moe_block):
    transformations_test = get_transformation('cifar100', phase='test')
    total_results = dict()
    for k in range(1, moe_block.num_experts + 1):
        moe_block.gate.top_k = k
        eval_results = dict()
        for label in CIFAR100_LABELS:
            test_data = CIFAR100Dataset(root_dir='/home/lb4653/mixture-of-experts-thesis/data/cifar100/testing', transform=transformations_test, labels=[label])
            eval_results[label] = model.evaluate(test_data)['acc']
        test_data = CIFAR100Dataset(root_dir='/home/lb4653/mixture-of-experts-thesis/data/cifar100/testing', transform=transformations_test)
        eval_results['total'] = model.evaluate(test_data)['acc']
        total_results[k] = eval_results
    return total_results



def main():
    # Setting seeds,
    torch.manual_seed(42)
    np.random.seed(42)

    # Load wandb data
    models_df = pd.read_csv('analysis/resnet18/model_paths.csv', sep=';')

    for constraint in ['kl', 'mean', 'importance', 'relative']:
        for position in [1, 2, 3, 4]:
            for num_exp in [4, 10]:
                try:
                    total_results = []
                    models_df_filtered = models_df[models_df['Constraint'] == constraint]
                    models_df_filtered = models_df_filtered[models_df_filtered['Position'] == position]
                    models_df_filtered = models_df_filtered[models_df_filtered['NumExperts'] == num_exp]
                    models_df_filtered = models_df_filtered[models_df_filtered['Model'] == 'Residual']

                    for i in range(len(models_df_filtered)):
                        model, moe_block = build_model(models_df_filtered.iloc[i]['Model'], models_df_filtered.iloc[i]['Constraint'], models_df_filtered.iloc[i]
                                                ['NumExperts'], models_df_filtered.iloc[i]['Position'], models_df_filtered.iloc[i]['RunPath'], models_df_filtered.iloc[i]['Filename'])

                        eval_results = varying_k_accuracy(model, moe_block)
                        total_results.append(eval_results)
                        df = pd.DataFrame(total_results)
                        del model
                        del moe_block

                    result_df = pd.DataFrame(total_results[0])
                    result_df = result_df + pd.DataFrame(total_results[1])
                    result_df = result_df + pd.DataFrame(total_results[2])
                    result_df = result_df / 3.0
                    result_df.to_csv('analysis/resnet18/results/varying_k_accuracy_{}_{}_{}.csv'.format(constraint, position, num_exp), index=True)
                except:
                    print('Error for {} {} {}'.format(constraint, position, num_exp))
if __name__ == "__main__":
    main()