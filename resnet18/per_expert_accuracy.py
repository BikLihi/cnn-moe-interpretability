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
from models.moe_layer.base_components.base_gating_network import BaseGatingNetwork
from collections import Counter

from losses.importance_loss import importance_loss


class WeightHook:
    def __init__(self, module, num_experts):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.weights = []
        self.weights_sparsely = []
        self.selected_experts = []
        self.selected_experts_combinations = []
        self.num_experts = num_experts

    def hook_fn(self, module, input, output):
        for pred in output:
            self.weights.append(nn.functional.softmax(pred, dim=0).cpu().tolist())
            top_k_logits, top_k_indices = pred.topk(2)
            top_k_weights = nn.functional.softmax(top_k_logits, dim=0)
            weights = np.zeros(self.num_experts)
            weights[top_k_indices[0]] = top_k_weights[0].cpu()
            weights[top_k_indices[1]] = top_k_weights[1].cpu()
            self.weights_sparsely.append(weights.tolist())
            experts = [top_k_indices[0].cpu().item(),
                       top_k_indices[1].cpu().item()]
            self.selected_experts = self.selected_experts + experts
            self.selected_experts_combinations.append(set(experts))

    def close(self):
        self.hook.remove()

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


def compute_number_of_pair_activations(model, moe_block):
    handle = WeightHook(moe_block.gate.fc, moe_block.num_experts)
    model.eval()
    transformations_test = get_transformation('cifar100', phase='test')
    test_data = CIFAR100Dataset(root_dir='/home/lb4653/mixture-of-experts-thesis/data/cifar100/testing', transform=transformations_test)
    eval_results = model.evaluate(test_data)['acc']
    handle.close()
    selected_experts_combinations = handle.selected_experts_combinations
    c = Counter(frozenset(s) for s in selected_experts_combinations)
    print('Accuracy: ', eval_results)
    print(c)


def single_expert_accuracy(model, moe_block, expert_index):
    transformations_test = get_transformation('cifar100', phase='test')
    gate_old = moe_block.gate
    moe_block.gate = SingleWeightingGatingNetwork(moe_block.gate.in_channels, expert_index, moe_block.num_experts)
    eval_results = dict()
    for label in CIFAR100_LABELS:
        test_data = CIFAR100Dataset(root_dir='/home/lb4653/mixture-of-experts-thesis/data/cifar100/testing', transform=transformations_test, labels=[label])
        eval_results[label] = model.evaluate(test_data)['acc']
    test_data = CIFAR100Dataset(root_dir='/home/lb4653/mixture-of-experts-thesis/data/cifar100/testing', transform=transformations_test)
    eval_results['total'] = model.evaluate(test_data)['acc']
    moe_block.gate = gate_old
    return eval_results


def per_expert_accuracy(model, moe_block):
    total_results = dict()
    transformations_test = get_transformation('cifar100', phase='test')
    # Evaluate model with all experts
    eval_results_full = dict()
    eval_results_full_weights = [dict() for i in range(moe_block.num_experts)]
    eval_results_full_weights_sparsely = [dict() for i in range(moe_block.num_experts)]
    eval_results_full_activations = [dict() for i in range(moe_block.num_experts)]

    for label in CIFAR100_LABELS:
        handle = WeightHook(moe_block.gate.fc, moe_block.num_experts)
        model.eval()
        test_data = CIFAR100Dataset(root_dir='/home/lb4653/mixture-of-experts-thesis/data/cifar100/testing', transform=transformations_test, labels=[label])
        eval_results_full[label] = model.evaluate(test_data)['acc']
        handle.close()

        handle.close()
        weights = handle.weights
        weights_sparsely = handle.weights_sparsely
        selected_experts = handle.selected_experts

        importance = np.sum(weights, 0)
        importance_sparsely = np.sum(weights_sparsely, 0)
        activations = Counter(selected_experts)

        for i in range(len(eval_results_full_weights)):
            eval_results_full_weights[i][label] = importance[i]
            eval_results_full_weights_sparsely[i][label] = importance_sparsely[i]
            eval_results_full_activations[i][label] = activations[i]

    test_data = CIFAR100Dataset(root_dir='/home/lb4653/mixture-of-experts-thesis/data/cifar100/testing', transform=transformations_test)
    eval_results_full['total'] = model.evaluate(test_data)['acc']
    total_results['Full Model'] = eval_results_full

    for i in range(len(eval_results_full_weights)):
        total_results['weights ' + str(i)] = eval_results_full_weights[i]
        total_results['weights sparsely ' + str(i)] = eval_results_full_weights_sparsely[i]
        total_results['activations ' + str(i)] = eval_results_full_activations[i]

    # Evaluate model with all experts and mean gating
    eval_results_mean = dict()
    gate_old = moe_block.gate
    moe_block.gate = EqualWeightGatingNetwork(moe_block.gate.in_channels, moe_block.gate.num_experts)
    for label in CIFAR100_LABELS:
        test_data = CIFAR100Dataset(root_dir='/home/lb4653/mixture-of-experts-thesis/data/cifar100/testing', transform=transformations_test, labels=[label])
        eval_results_mean[label] = model.evaluate(test_data)['acc']
    test_data = CIFAR100Dataset(root_dir='/home/lb4653/mixture-of-experts-thesis/data/cifar100/testing', transform=transformations_test)
    eval_results_mean['total'] = model.evaluate(test_data)['acc']
    total_results['Mean'] = eval_results_mean
    moe_block.gate = gate_old
        
    handle = WeightHook(moe_block.gate.fc, moe_block.num_experts)
    model.eval()
    model.to('cuda:0')
    with torch.no_grad():
        acc = model.evaluate(test_data)['acc']

    handle.close()
    weights = handle.weights
    weights_sparsely = handle.weights_sparsely
    selected_experts = handle.selected_experts

    importance = np.sum(weights, 0)
    importance_sparsely = np.sum(weights_sparsely, 0)
    activations = Counter(selected_experts)


    indices = np.argsort(np.array(importance))[::-1]

    for i, index in enumerate(indices):
        expert_results = single_expert_accuracy(model, moe_block, i)
        expert_results['Total importance'] = importance[i]
        expert_results['Sparse importance'] = importance_sparsely[i]
        expert_results['# Activations'] = activations[i]
        total_results['Accuracy : ' + str(i)] = expert_results

    return total_results


def main():
    # Setting seeds,
    torch.manual_seed(42)
    np.random.seed(42)

    # Load wandb data
    model_type = 'Residual'
    constraint = 'importance'
    num_experts = 4
    position = 1
    run_path = 'lukas-struppek/final_resnet_18/3k0n0th2'
    filename = 'Residual_4_topK=2_loss=kl_divergence_w_aux=0.5_moePosition=1_1_final.tar'

    model, moe_block = build_model(model_type, constraint, num_experts, position, run_path, filename)
    compute_number_of_pair_activations(model, moe_block)
    # eval_results = per_expert_accuracy(model, moe_block)
    # df = pd.DataFrame(eval_results)
    # df.to_csv('analysis/resnet18/results/per_expert_accuracy_{}_{}_{}.csv'.format(constraint, position, num_experts), index=True)

if __name__ == "__main__":
    main()