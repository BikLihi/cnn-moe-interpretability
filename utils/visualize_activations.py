import torch
import numpy as np
import matplotlib.pyplot as plt

class CNNActivationVisualization():
    def __init__(self, model, selected_layer):
        self.model = model
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.selected_layer = selected_layer
        self.activations = []
        self.hook = None

    def hook_layer(self):
        def hook_function(module, input, output):
            if type(output) is dict:
                self.activations.append(output['output'].cuda())
            else:
                self.activations.append(output.cuda())
        self.hook = self.selected_layer.register_forward_hook(hook_function)

    def compute_activations(self, image, number_of_filters, feature_level, save_path=None, top_k=5, file_name=None):
        self.hook_layer()
        self.model.to('cuda:0')
        if type(image) == np.ndarray:
            image = torch.from_numpy(image)
        image = image.to('cuda:0')
        self.model(image)
        self.hook.remove()

        avg_activation = [self.activations[feature_level][0, i].mean().item() for i in range(number_of_filters)]
        ###########
        self.activations = []
        ###########
        plt.figure(figsize=(7,5))
        act = plt.plot(avg_activation, linewidth=2.)
        ax = act[0].axes
        ax.set_xlim(0, number_of_filters)
        ax.set_xlabel("feature map")
        ax.set_ylabel("mean activation")
        ax.set_xticks([0, np.floor(number_of_filters/2), number_of_filters])
        if save_path:
            if file_name:
                plt.savefig(save_path + file_name + 'png')
            else:
                plt.savefig(save_path + 'avg_activation_'+str(self.selected_layer)+'.png')

        top_k_activations = (-np.array(avg_activation)).argsort()[:top_k]
        return top_k_activations
