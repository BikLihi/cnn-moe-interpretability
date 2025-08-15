import torch
import torch.nn as nn

import numpy as np

from utils.cifar100_dataset import CIFAR100Dataset, create_subset
from scipy.stats import entropy


class MoE(BaseModel):
    def __init__(self, experts, Gate, data_folder, classes=CIFAR100Dataset.CIFAR100_LABELS, transform=None, name=None, unknown_class=False, feature_extractor=None):
        super().__init__()
        self.classes = classes
        self.experts = sorted(experts, key=lambda expert: expert.classes_encoded)
        self.name = name
        self.gate = Gate(classes=classes, experts=self.experts, name=str(name) + '_gate', feature_extractor=feature_extractor)
        self.data_folder = data_folder
        self.unknown_class = unknown_class
        self.feature_extractor = feature_extractor

        self.dataset = CIFAR100Dataset(data_folder=self.data_folder, classes=self.classes, transform=transform)
        self.trainingset, self.validationset = self.dataset.train_test_split([0.8, 0.2])



    def forward(self, x):
        self.eval()
        with torch.set_grad_enabled(False):
            combined_output, _ = self.gate(x)
        return {'output': combined_output}


    def train_experts(self, save_state_path=None, num_epochs=50):
        for expert in self.experts:
            training_subset = create_subset(self.trainingset, expert.classes)
            validation_subset = create_subset(self.validationset, expert.classes)
            expert.set_parameter_requires_grad(True)
            if save_state_path:
                expert.fit(training_subset, validation_subset, num_epochs=num_epochs, save_state_path=save_state_path + expert.name + '.pth')
            else:
                expert.fit(training_subset, validation_subset, num_epochs=num_epochs)
            expert.set_parameter_requires_grad(False)


    def train_gate(self, save_state_path=None, num_epochs=30, learning_rate=0.01):
        self.gate.experts = self.experts
        if save_state_path:
            self.gate.fit(
                self.trainingset, 
                self.validationset, 
                save_state_path=save_state_path + self.gate.name + '.pth', 
                num_epochs=num_epochs, 
                learning_rate=learning_rate
            )
        else:
            self.gate.fit(
                self.trainingset, 
                self.validationset, 
                num_epochs=num_epochs, 
                learning_rate=learning_rate
            )

    
    def evaluate(            
            self,
            test_data=None,
            batch_size=128,
            criterion=torch.nn.CrossEntropyLoss(),
            device='cuda:0'):

        if test_data is None:
            test_data = self.validationset
        self.to(device)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

        self.eval()
        running_test_loss = 0.0
        running_test_corrects = 0
        mean_entropy = 0.0

        with torch.no_grad():
            for _, data in enumerate(test_loader):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = self.forward(inputs)
                pred = torch.argmax(outputs, dim=1)
                loss = criterion(outputs, labels)

                outputs_sm = torch.softmax(outputs, dim=1).cpu().numpy()
                mean_entropy += np.sum(entropy(outputs_sm, base=2, axis=1))

                running_test_loss += loss.item() * inputs.size(0)
                pred = torch.argmax(outputs, dim=1)
                running_test_corrects += torch.sum(pred == labels.data)

        test_loss = np.round(running_test_loss / len(test_data), decimals=4)
        test_acc = np.round(float(running_test_corrects.item()) / len(test_data), decimals=4)
        mean_entropy = np.round(mean_entropy / len(test_data), decimals=4)

        result = {'loss': test_loss, 'acc': test_acc, 'entropy': mean_entropy}

        if self.name:
            result['name'] = self.name

        return result
    

    def evaluate_decisions(            
            self,
            test_data=None,
            batch_size=128,
            criterion=torch.nn.CrossEntropyLoss(),
            device='cuda:0'):

        self.to(device)

        if test_data is None:
            test_data = self.validationset

        for label in CIFAR100Dataset.CIFAR100_LABELS:
            subset = create_subset(test_data, [label])
            test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
            with torch.no_grad():
                if torch.cuda.is_available():
                    weights_sum = torch.cuda.FloatTensor(len(self.experts)).fill_(0)
                else:
                    weights_sum = torch.zeros(len(self.experts))

                for _, data in enumerate(test_loader):
                    inputs, labels = data
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    _, weights = self.gate(inputs)
                    weights_sum += torch.sum(weights, dim=0)
                print(weights_sum / len(subset))

