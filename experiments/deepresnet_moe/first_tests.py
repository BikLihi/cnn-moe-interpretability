import torch
import numpy as np
import math

from models.moe_layer.resnet.bottleneck_moe_layer import BottleneckMoeLayer
from models.moe_layer.soft_gating_networks import SimpleGate
from models.moe_layer.hard_gating_networks import AbsoluteImportanceGate, RelativeImportanceGate
from models.moe_layer.resnet101_moe.resnet101_moe import DeepResNetMoE
from datasets.cifar100_dataset import CIFAR100Dataset
from utils.dataset_utils import train_test_split, get_transformation
from utils.cifar100_utils import CIFAR100_LABELS


# Setting seeds
torch.manual_seed(42)
np.random.seed(42)

# Loading datasets
transformations_training = get_transformation('cifar100', phase='training')
transformations_test = get_transformation('cifar100', phase='test')
cifar_data = CIFAR100Dataset(
    root_dir='/home/lb4653/mixture-of-experts-thesis/data/cifar100/training', transform=transformations_training)
training_data, validation_data = train_test_split(cifar_data, [0.8, 0.2])
validation_data.transform = transformations_test

# Define scheduler functions
def schedule(epoch):
    if epoch <= 100:
        return 10
    if epoch < 180:
        return 1
    return 0.1
    

# Fix parameters
num_epochs = 200
batch_size = 128
learning_rate = 0.0001
optimizer_class = torch.optim.Adam



# Create model
gate = AbsoluteImportanceGate(64, 4, constr_threshold=0.1)
moe_layer = BottleneckMoeLayer(64, 256, num_experts=4, top_k=2, use_noise=True, gating_network=gate)
model = DeepResNetMoE(num_experts=4, top_k=2, gating_network=gate)

optimizer = optimizer_class(model.parameters(), learning_rate)
lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)

model.fit(
     training_data=training_data,
     validation_data=validation_data,
     num_epochs=num_epochs,
     batch_size=batch_size,
     learning_rate=learning_rate,
     optimizer=optimizer,
     early_stopping=False,
     enable_logging=True,
     lr_scheduler=lr_scheduler,
     wandb_project='analyze_deep_moe',
     wandb_name='ResNet101_MoE_Absolute_w=0.1',
     wandb_tags=['MoE', 'ResNet101', 'AbsoluteImportanceGate', 'HardConstraint'],
 )


# Create model
gate = RelativeImportanceGate(64, 4, constr_threshold=0.1)
moe_layer = BottleneckMoeLayer(64, 256, num_experts=4, top_k=2, use_noise=True, gating_network=gate)
model = DeepResNetMoE(num_experts=4, top_k=2, gating_network=gate)

optimizer = optimizer_class(model.parameters(), learning_rate)
lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)

model.fit(
     training_data=training_data,
     validation_data=validation_data,
     num_epochs=num_epochs,
     batch_size=batch_size,
     learning_rate=learning_rate,
     optimizer=optimizer,
     early_stopping=False,
     enable_logging=True,
     lr_scheduler=lr_scheduler,
     wandb_project='analyze_deep_moe',
     wandb_name='ResNet101_MoE_Relative_w=0.1',
     wandb_tags=['MoE', 'ResNet101', 'RelativeImportanceGate', 'HardConstraint'],
 )