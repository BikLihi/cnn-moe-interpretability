import torch
import numpy as np

from models.moe_layer.gating_networks import SimpleGate, HardConstraintGate, RelativeImportanceGate
from models.moe_layer.resnet.shallow_resnet_moe import ShallowResNetMoE
from datasets.cifar100_dataset import CIFAR100Dataset
from utils.dataset_utils import train_test_split, get_transformation


# Setting seeds
torch.manual_seed(0)
np.random.seed(0)

# Loading datasets
transformations_training = get_transformation('cifar100', phase='training')
transformations_test = get_transformation('cifar100', phase='test')
cifar_data = CIFAR100Dataset(
    root_dir='/home/lb4653/mixture-of-experts-thesis/data/cifar100/training', transform=transformations_training)
training_data, validation_data = train_test_split(cifar_data, [0.8, 0.2])
validation_data.transform = transformations_test

# fix parameters
num_exp = 4
top_k = 2
use_noise = True
num_blocks = 1
num_epochs = 200
batch_size = 512
learning_rate = 0.001
optimizer_class = torch.optim.Adam

# Variable parameters
gating_networks = [SimpleGate, HardConstraintGate, RelativeImportanceGate]
constr_threshold_values = [0.1, 0.3]
milestones_values = [[100, 125, 150, 175]]
w_importance = 0.1


for gating_network in gating_networks:
    for constr_threshold in constr_threshold_values:
        for milestones in milestones_values:

            if gating_network.__qualname__ == 'SimpleGate' and constr_threshold == 0.3:
                continue

            model = ShallowResNetMoE(num_experts=num_exp, top_k=top_k, use_noise=use_noise,
                                     num_blocks=1, gating_network=gating_network, w_importance=w_importance,
                                     name='ShallowResNetMoE_' + gating_network.__qualname__)


            optimizer = optimizer_class(model.parameters(), learning_rate)
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones, gamma=0.5)

            if gating_network.__qualname__ == 'SimpleGate':
                custom_config = {'gate': gating_network.__qualname__,
                             'w_importance': w_importance, 'scheduler_milestones': milestones}
                model.moe.gate.w_importance = w_importance

            else:
                custom_config = {'gate': gating_network.__qualname__,
                             'constr_threshold': constr_threshold, 'scheduler_milestones': milestones}
                model.moe.gate.constr_threshold = constr_threshold


            model.fit(
                training_data=training_data,
                validation_data=validation_data,
                num_epochs=num_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                optimizer=optimizer,
                early_stopping=True,
                enable_logging=True,
                lr_scheduler=lr_scheduler,
                wandb_name='ShallowResNetMoE' + gating_network.__qualname__,
                wandb_tags=['compare_gates', gating_network.__qualname__, 'ShallowResNetMoE'],
                wandb_custom_config=custom_config
            )
