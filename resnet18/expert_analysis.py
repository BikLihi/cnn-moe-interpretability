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
from models.resnet.resnet18 import ResNet18
from models.moe_layer.soft_gating_networks import SimpleGate
from models.moe_layer.hard_gating_networks import RelativeImportanceGate, AbsoluteImportanceGate
from models.moe_layer.static_gating_networks import EqualWeightGatingNetwork, SingleWeightingGatingNetwork
from datasets.cifar100_dataset import CIFAR100Dataset
from utils.dataset_utils import train_test_split, get_transformation
from utils.cifar100_utils import CIFAR100_LABELS

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

    elif constraint == 'baseline':
        pass       
    else:
        raise ValueError('Invalid Constraint')

    if constraint != 'baseline':  
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
    
    else:
        model = ResNet18(CIFAR100_LABELS, name='Baseline')
        moe_block = None


    # Load model weights
    file_model = wandb.restore(
        filename, run_path=run_path, root='./model_weights')

    # # Load parameters
    model.load_state_dict(torch.load(file_model.name)['model_state_dict'])

    # # Move model to cuda
    model.to(model.device)

    print('Load model: ' + filename)

    return model, moe_block


def per_class_accuracy(model):
    transformations_test = get_transformation('cifar100', phase='test')
    df_result = pd.DataFrame()
    for label in CIFAR100_LABELS:
        print('evaluation on {}'.format(label))
        # Loading dataset
        model.eval()
        model.training = False
        cifar_test_small = CIFAR100Dataset(root_dir='/home/lb4653/mixture-of-experts-thesis/data/cifar100/testing', transform=transformations_test, labels=[label])
        eval_acc= model.evaluate(cifar_test_small)['acc']
        df_result = df_result.append({'label': label, 'acc': eval_acc}, ignore_index=True)
    
    cifar_test = CIFAR100Dataset(root_dir='/home/lb4653/mixture-of-experts-thesis/data/cifar100/testing', transform=transformations_test)
    overall_acc = model.evaluate(cifar_test)['acc']
    df_result = df_result.append({'label': 'overall', 'acc': overall_acc}, ignore_index=True)
    return df_result



def main():  
    constraint = 'mean'
    position = 1
    num_exp = 4
    # run_path = 'lukas-struppek/final_resnet_18/qropxr6y'
    # filename = 'Baseline_2_schedule_0_final.tar'

    # Load wandb data
    models_df = pd.read_csv('analysis/resnet18/model_paths.csv', sep=';')
    result_df = pd.DataFrame()

    models_df = models_df[models_df['Constraint'] == constraint]
    models_df = models_df[models_df['Position'] == position]
    models_df = models_df[models_df['NumExperts'] == num_exp]
    models_df = models_df[models_df['Model'] == 'Residual']

    for i in range(len(models_df)):
        model, moe_layer = build_model('Residual', constraint, num_exp, position, models_df.iloc[i].RunPath, models_df.iloc[i].Filename)
        df = per_class_accuracy(model)
        result_df = result_df.append(df)

    # run_paths = ['lukas-struppek/final_resnet_18/3qp2c0vg', 'lukas-struppek/final_resnet_18/2i3oi5fi', 'lukas-struppek/final_resnet_18/qropxr6y']
    # filenames = ['Baseline_0_schedule_0_final.tar', 'Baseline_1_schedule_0_final.tar', 'Baseline_2_schedule_0_final.tar']
    # result_df = pd.DataFrame()
    # for i in range(3):
    #     model, moe_layer = build_model('Residual', 'baseline', 1, 1, run_paths[i], filenames[i])
    #     df = per_class_accuracy(model)
    #     result_df = result_df.append(df)

    result_df.groupby('label').mean().to_csv('analysis/resnet18/results/per_class_accuaracy_{}_{}_{}.csv'.format(constraint, position, num_exp))


if __name__ == "__main__":
    main()