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

class WeightHook:
    def __init__(self, module, num_experts):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.weights = []
        self.selected_experts = []
        self.num_experts = num_experts

    def hook_fn(self, module, input, output):
        for pred in output:
            top_k_logits, top_k_indices = pred.topk(2)
            top_k_weights = nn.functional.softmax(top_k_logits, dim=0)
            weights = np.zeros(self.num_experts)
            weights[top_k_indices[0]] = top_k_weights[0].cpu()
            weights[top_k_indices[1]] = top_k_weights[1].cpu()
            self.weights.append(weights)
            experts = [top_k_indices[0].cpu().item(),
                       top_k_indices[1].cpu().item()]
            self.selected_experts.append(experts)

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


def analyze_weighting(model, moe_layer, testset, batch_size=64):
    dataloader = torch.utils.data.DataLoader(testset, batch_size=batch_size)

    handle = WeightHook(moe_layer.gate.fc, moe_layer.num_experts)

    model.eval()
    with torch.no_grad():
        acc = model.evaluate(testset)['acc']
        # for images, labels in dataloader:
        #     images = images.to(model.device)
        #     model(images)

    handle.close()

    mean_importance = np.round(np.mean(handle.weights, axis=0), 4)
    living_experts = np.sum(mean_importance > 0.01)

    # Using std from total population
    cv_imp =  np.std(mean_importance, axis=0) / (np.mean(mean_importance, axis=0) + 1e-10)

    selected_experts = handle.selected_experts
    num_activations = np.zeros(moe_layer.num_experts)
    for element in selected_experts:
        num_activations[element[0]] += 1
        num_activations[element[1]] += 1
    mean_activations = num_activations / len(testset)

    cv_activations = np.std(mean_activations, axis=0) / (np.mean(mean_activations, axis=0) + 1e-10)
    return mean_importance, cv_imp, living_experts, mean_activations, cv_activations, acc


def main():
    # Setting seeds,
    torch.manual_seed(42)
    np.random.seed(42)

    # Loading datasets,
    transformations_test = get_transformation('cifar100', phase='test')
    basic_transformation = get_transformation('no_transform')
    cifar_test = CIFAR100Dataset(
        root_dir='/home/lb4653/mixture-of-experts-thesis/data/cifar100/testing', transform=transformations_test)
    cifar_test_no_prep = CIFAR100Dataset(
        root_dir='/home/lb4653/mixture-of-experts-thesis/data/cifar100/testing', transform=basic_transformation)

    # Load wandb data
    models_df = pd.read_csv('analysis/resnet18/model_paths.csv', sep=';')
    result_df = pd.DataFrame()


    models_df = models_df[models_df['Constraint'] == 'mean']
    for i in range(len(models_df)):
        model, moe_layer = build_model(models_df.iloc[i]['Model'], models_df.iloc[i]['Constraint'], models_df.iloc[i]
                                ['NumExperts'], models_df.iloc[i]['Position'], models_df.iloc[i]['RunPath'], models_df.iloc[i]['Filename'])

        avg_weights, cv_imp, living_experts, num_activations, cv_activations, acc = analyze_weighting(model, moe_layer, cifar_test)

        del model
        del moe_layer

        result_dict = models_df.iloc[i].to_dict()
        result_dict['avg_weights'] = avg_weights
        result_dict['cv_imp'] = cv_imp
        result_dict['living_experts'] = living_experts
        result_dict['num_activations'] = num_activations
        result_dict['cv_activations'] = cv_activations
        result_dict['acc'] = acc

        result_df = result_df.append(result_dict, ignore_index=True)
    
    result_df.to_csv('analysis/resnet18/utilization_relative_results.csv', index=False)

if __name__ == "__main__":
    main()