import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import copy
from models.mnist.mnist_net import MnistNet
import pandas as pd
from datasets.mnist_dataset import MNISTDataset

from sklearn.metrics import confusion_matrix

from models.base_model import BaseModel

from scipy.stats import entropy


class GenericFMGate(BaseModel):
    def __init__(self, classes, experts, name=None):
        super().__init__()
        self.classes = classes
        self.num_classes= len(self.classes)
        self.experts = experts
        self.num_experts = len(experts)
        self.name = name
        
        # Encoding labels into labels [0, ..., N-1]
        self.target_encoding = dict()
        for index, c in enumerate(self.classes):
            self.target_encoding[c] = index

        self.target_decoding = dict()
        for index, c in enumerate(self.classes):
            self.target_decoding[index] = c

        self.inflate = nn.ModuleList()
        for expert in self.experts:
            self.inflate.append(nn.Linear(expert.num_classes, self.num_classes, bias=False))

        self.conv1 = nn.Conv2d(in_channels=64 * self.num_experts, out_channels=256, kernel_size=3)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * 5 * 5, 128)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(128, self.num_experts)


    def forward(self, x, output_only=True):
        device = "cuda:0" if torch.cuda.is_available() else "cpu:0"

        for expert in self.experts:
            for param in expert.parameters():
                param.requires_grad = False
            expert.to(device)
            expert.eval()

        pool_output_experts = []
        inflated_expert_logits = []

        for i, expert in enumerate(self.experts):
            model_output = expert(x, output_only=False)
            logit_expert = model_output['output']
            pool_output = model_output['maxpool']

            inflated_expert_logits.append(self.inflate[i](logit_expert))
            pool_output_experts.append(pool_output)

        concat = torch.cat(pool_output_experts, dim=1)

        out = self.conv1(concat)
        out = self.flatten(out)
        out = self.dropout(self.fc1(out))
        out = self.fc2(out)
        weights = F.softmax(out, dim=1)

        combined_output = np.sum(
            [ (weights[:, i].repeat(self.num_classes).reshape((self.num_classes, weights.size()[0])).T * inflated_expert_logits[i])
              for i in range(self.num_experts) ]
        )
        if output_only:
            return combined_output
        else:
            return {'output': combined_output, 'weights': weights}


    # def evaluate(self, test_data, batch_size, criterion, device):
    #     self.to(device)
    #     test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

    #     self.eval()
    #     running_test_loss = 0.0
    #     running_test_corrects = 0
    #     mean_entropy = 0.0

    #     with torch.no_grad():
    #         for i, data in enumerate(test_loader):
    #             inputs, labels = data
    #             inputs = inputs.to(device)
    #             labels = labels.apply_(lambda x: self.target_encoding[x]).to(device)


    #             model_outputs = self.forward(inputs)
    #             outputs = model_outputs['output']
    #             weights = model_outputs['weights']
    #             pred = torch.argmax(outputs, dim=1)
    #             loss = criterion(outputs, labels)

    #             outputs_sm = torch.softmax(outputs, dim=1).cpu().numpy()
    #             mean_entropy += np.sum(entropy(outputs_sm, base=2, axis=1))

    #             running_test_loss += loss.item() * inputs.size(0)
    #             pred = torch.argmax(outputs, dim=1)
    #             running_test_corrects += torch.sum(pred == labels.data)

    #     test_loss = np.round(running_test_loss / len(test_data), decimals=4)
    #     test_acc = np.round(float(running_test_corrects.item()) / len(test_data), decimals=4)
    #     mean_entropy = np.round(mean_entropy / len(test_data), decimals=4)

    #     result = {'loss': test_loss, 'acc': test_acc, 'entropy': mean_entropy}

    #     if self.name:
    #         result['name'] = self.name

    #     return result


    def predict(self, data, device, batch_size=32, decoding_dict=None):
        self.to(device)
        for expert in self.experts:
            expert.to(device)
        self.eval()

        pred_tensors = torch.Tensor().to(device)

        data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False)

        with torch.no_grad():
            for _, (images, labels) in enumerate(data_loader):
                images = images.to(device)

                model_output = self(images)
                gate_logits = model_output['output']
                weights = model_output['weights']
                pred_tensors = torch.cat([pred_tensors, gate_logits], dim=0)

        pred = np.argmax(pred_tensors.cpu().numpy(), axis=1)

        pred = np.array([self.target_decoding[x] for x in pred])

        return pred

    
class GenericMnistNetGet(BaseModel):
    def __init__(self, classes, experts, name=None):
        super().__init__()
        self.classes = classes
        self.num_classes= len(self.classes)
        self.experts = experts
        self.num_experts = len(experts)
        self.name = name
        
        # Encoding labels into labels [0, ..., N-1]
        self.target_encoding = dict()
        for index, c in enumerate(self.classes):
            self.target_encoding[c] = index

        self.target_decoding = dict()
        for index, c in enumerate(self.classes):
            self.target_decoding[index] = c

        self.inflate = nn.ModuleList()
        for expert in self.experts:
            self.inflate.append(nn.Linear(expert.num_classes, self.num_classes, bias=False))

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(7 * 7 * 64, 128)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(128, self.num_experts)


    def forward(self, x, output_only=True):
        device = "cuda:0" if torch.cuda.is_available() else "cpu:0"

        for expert in self.experts:
            for param in expert.parameters():
                param.requires_grad = False
            expert.to(device)
            expert.eval()

        pool_output_experts = []
        inflated_expert_logits = []

        for i, expert in enumerate(self.experts):
            model_output = expert(x, output_only=False)
            logit_expert = model_output['output']
            pool_output = model_output['maxpool']

            inflated_expert_logits.append(self.inflate[i](logit_expert))
            pool_output_experts.append(pool_output)

        concat = torch.cat(pool_output_experts, dim=1)

        out_conv1 = F.relu(self.conv1(x))
        out_maxpool1 = self.maxpool1(out_conv1)
        out_conv2 = F.relu(self.conv2(out_maxpool1))
        out_maxpool2 = self.maxpool2(out_conv2)
        out_flatten = self.flatten(out_maxpool2)
        out_fc1 = self.dropout(F.relu(self.fc1(out_flatten)))
        out = self.fc2(out_fc1)

        weights = F.softmax(out, dim=1)

        combined_output = np.sum(
            [ (weights[:, i].repeat(self.num_classes).reshape((self.num_classes, weights.size()[0])).T * inflated_expert_logits[i])
              for i in range(self.num_experts) ]
        )

        if output_only:
            return combined_output
        else:
            return {'output': combined_output, 'weights': weights}

    # def evaluate(self, test_data, batch_size, criterion, device):
    #     self.to(device)
    #     test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

    #     self.eval()
    #     running_test_loss = 0.0
    #     running_test_corrects = 0
    #     mean_entropy = 0.0

    #     with torch.no_grad():
    #         for i, data in enumerate(test_loader):
    #             inputs, labels = data
    #             inputs = inputs.to(device)
    #             labels = labels.apply_(lambda x: self.target_encoding[x]).to(device)


    #             model_outputs = self.forward(inputs)
    #             outputs = model_outputs['output']
    #             weights = model_outputs['weights']
    #             pred = torch.argmax(outputs, dim=1)
    #             loss = criterion(outputs, labels)

    #             outputs_sm = torch.softmax(outputs, dim=1).cpu().numpy()
    #             mean_entropy += np.sum(entropy(outputs_sm, base=2, axis=1))

    #             running_test_loss += loss.item() * inputs.size(0)
    #             pred = torch.argmax(outputs, dim=1)
    #             running_test_corrects += torch.sum(pred == labels.data)

    #     test_loss = np.round(running_test_loss / len(test_data), decimals=4)
    #     test_acc = np.round(float(running_test_corrects.item()) / len(test_data), decimals=4)
    #     mean_entropy = np.round(mean_entropy / len(test_data), decimals=4)

    #     result = {'loss': test_loss, 'acc': test_acc, 'entropy': mean_entropy}

    #     if self.name:
    #         result['name'] = self.name

    #     return result


    def predict(self, data, device, batch_size=32, decoding_dict=None):
        self.to(device)
        for expert in self.experts:
            expert.to(device)
        self.eval()

        pred_tensors = torch.Tensor().to(device)

        data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False)

        with torch.no_grad():
            for _, (images, labels) in enumerate(data_loader):
                images = images.to(device)

                model_output = self(images)
                gate_logits = model_output['output']
                weights = model_output['weights']
                pred_tensors = torch.cat([pred_tensors, gate_logits], dim=0)

        pred = np.argmax(pred_tensors.cpu().numpy(), axis=1)

        pred = np.array([self.target_decoding[x] for x in pred])

        return pred

