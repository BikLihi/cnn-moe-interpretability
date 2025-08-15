import numpy as np
from torchvision import transforms
import torch
import math
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
import torch.nn as nn

class GatingHook:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.inputs = []
        self.outputs = []
        self.weights = []
        self.selected_experts = []
        
    def hook_fn(self, module, input, output):
        for pred in output:
            self.outputs.append(pred.detach().cpu().numpy())
            top_k_logits, top_k_indices = pred.topk(2)
            top_k_weights = nn.functional.softmax(top_k_logits)
            weights = torch.zeros(10, requires_grad=False)
            weights[top_k_indices[0]] = top_k_weights[0]
            weights[top_k_indices[1]] = top_k_weights[1]
            self.weights.append(weights.detach().cpu().numpy())
            experts = [top_k_indices[0].cpu().item(), top_k_indices[1].cpu().item()]
            self.selected_experts.append(experts)
        
    def close(self):
        self.hook.remove()


def visualize_gating_decision(model, layer, dataset, save_path):

    hook = GatingHook(layer)
    labels = []
    superclass_labels = []
    for image, label in dataloader:
        for l in label:
            labels.append(CIFAR100_DECODING[l.cpu().item()])
            superclass_labels.append(get_superclass(CIFAR100_DECODING[l.cpu().item()]))
        model(image.to('cuda:0'))
    
    tsne = TSNE(n_components=2, perplexity=5, learning_rate=10, verbose=1, n_iter=1000).fit_transform(hook.outputs)

    tx, ty = tsne[:,0], tsne[:,1]
    tx = (tx-np.min(tx)) / (np.max(tx) - np.min(tx))
    ty = (ty-np.min(ty)) / (np.max(ty) - np.min(ty))

    df_subset = dict()
    df_subset['Dimension 1'] = tx
    df_subset['Dimension 2'] = ty

    plt.figure(figsize=(16,10))
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue=superclass_labels,
        palette=sns.color_palette("muted", 20),
        data=df_subset,
        legend="full",
        alpha=0.3
    )

    width = 4000
    height = 3000
    max_dim = 100
    full_image = Image.new('RGB', (width, height))

    for idx, x in enumerate(cifar_test_no_transform):
        tile = transforms.ToPILImage()(x[0].squeeze_(0))

        rs = max(1, tile.width / max_dim, tile.height / max_dim)
        tile = tile.resize((int(tile.width / rs),
                            int(tile.height / rs)),
                        Image.ANTIALIAS)
        full_image.paste(tile, (int((width-max_dim) * tx[idx]),
                                int((height-max_dim) * ty[idx])))
    full_image.save(save_path)