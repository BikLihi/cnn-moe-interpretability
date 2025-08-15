import copy
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import entropy

from models.base_model import BaseModel


class FMGate(BaseModel):
    def __init__(self, classes, experts, name=None):
        super().__init__()
        self.classes = classes
        self.num_classes = len(self.classes)
        self.experts = experts
        self.num_experts = len(experts)
        self.name = name

        # Encoding labels into labels [0, ..., N-1]
        self.target_encoding = dict()
        for index, class_ in enumerate(self.classes):
            self.target_encoding[class_] = index

        self.target_decoding = dict()
        for index, class_ in enumerate(self.classes):
            self.target_decoding[index] = class_

        self.conv1 = nn.Conv2d(in_channels=64 * self.num_experts, out_channels=256, kernel_size=3)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * 5 * 5, 128)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(128, self.num_experts)


    def forward(self, x, output_only=True):
        for expert in self.experts:
            expert.set_parameter_requires_grad(False)

        logits_experts = []
        pool_output_experts = []

        for expert in self.experts:
            expert.to(self.device)
            x = x.to(self.device)
            with torch.no_grad():
                model_output = expert(x, output_only=False)
            logit_expert = model_output['output']
            pool_output_expert = model_output['maxpool']

            # Ignore unknown class prediction
            if expert.unknown_class:
                logit_expert = logit_expert[:, :-1]

            decoded_logits = torch.zeros([logit_expert.size()[0], self.num_classes])
            for i in range(logit_expert.size()[1]):
                decoded_logits[:, expert.target_decoding[i]] = logit_expert[:, i]
            logits_experts.append(decoded_logits.to(self.device))
            pool_output_experts.append(pool_output_expert)

        concat = torch.cat(pool_output_experts, dim=1)

        out = self.conv1(concat)
        out = self.flatten(out)
        out = self.dropout(self.fc1(out))
        out = self.fc2(out)

        weights = F.softmax(out, dim=1)

        combined_output = np.sum(
            [(weights[:, i].repeat(self.num_classes)
                .reshape((self.num_classes, weights.size()[0])).T * logits_experts[i])
             for i in range(self.num_experts)]
        )

        if output_only:
            return combined_output
        else:
            return {'output': combined_output, 'weights': weights}


    def evaluate_experts(self, test_data, batch_size, criterion):
        results = []
        for expert in self.experts:
            results.append(expert.evaluate(test_data, batch_size, criterion))

        return results


    def analyze_decisions(self, test_data, batch_size=256):
        self.to(self.device)
        weights = np.zeros(self.num_experts)
        dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
        with torch.no_grad():
            for (images, labels) in dataloader:
                results = self.forward(images, False)
                weights += results['weights'].sum(dim=0).cpu().numpy()
        return weights / len(test_data)
    

    def get_highest_weighted_examples(self, test_data):
        self.to(self.device)
        columns = ['sample_index']
        for i in range(self.num_experts):
            columns += 'weight expert ' + str(i)
        df = pd.DataFrame(columns=['sample_index'])
        dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
        with torch.no_grad():
            for i, (images, labels) in enumerate(dataloader):
                results = self.forward(images, False)
                result = {'sample_index' : i}
                result['label'] = int(labels.cpu().item())
                result['prediction'] = results['output'].argmax(dim=1).cpu().numpy()
                for i in range(self.num_experts):
                    result['weight expert ' + str(i)] = results['weights'][0][i].cpu().numpy()
                df = df.append(result, ignore_index=True)
        return df



    ### ToDo - redefine ###
    def analyze_decision_process(self, test_data, size, criterion, device):
        self.to(device)
        for expert in self.experts:
            expert.to(device)
        self.eval()

        pred_tensors = torch.Tensor().to(device)

        test_loader = torch.utils.data.DataLoader(test_data, batch_size=size, shuffle=True)

        with torch.no_grad():
            _, (images, labels) = next(enumerate(test_loader))
            images = images.to(device)
            labels = labels.cpu().numpy()

            gate_logits, weights = self(images)

            pred_tensors = torch.cat([pred_tensors, gate_logits], dim=0)
            gate_pred = np.argmax(pred_tensors.cpu().numpy(), axis=1)
            weights = weights.cpu().numpy()

        result_df = pd.DataFrame(
            {'True Label': labels,
            'Gate Pred': gate_pred}
        )

        for i, expert in enumerate(self.experts):
            result_df['expert ' + str(i+1)] = expert.predict(images, device, size)
            result_df['weight ' + str(i+1)] = np.round(weights[:, i], decimals=4)
        return result_df



class MnistNetGate(BaseModel):
    def __init__(self, classes, experts, name=None):
        super().__init__()
        self.classes = classes
        self.num_classes = len(self.classes)
        self.experts = experts
        self.num_experts = len(experts)
        self.name = name

        # Encoding labels into labels [0, ..., N-1]
        self.target_encoding = dict()
        for index, class_ in enumerate(self.classes):
            self.target_encoding[class_] = index

        self.target_decoding = dict()
        for index, class_ in enumerate(self.classes):
            self.target_decoding[index] = class_

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(7 * 7 * 64, 128)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(128, self.num_experts)


    def forward(self, x, output_only=True):
        for expert in self.experts:
            expert.set_parameter_requires_grad(False)

        logits_experts = []

        for expert in self.experts:
            expert.to(self.device)
            x = x.to(self.device)
            with torch.no_grad():
                logit_expert = expert(x, output_only=True)

            # Ignore unknown class prediction
            if expert.unknown_class:
                logit_expert = logit_expert[:, :-1]

            decoded_logits = torch.zeros([logit_expert.size()[0], self.num_classes])
            for i in range(logit_expert.size()[1]):
                decoded_logits[:, expert.target_decoding[i]] = logit_expert[:, i]
            logits_experts.append(decoded_logits.to(self.device))
        
        out_conv1 = F.relu(self.conv1(x))
        out_maxpool1 = self.maxpool1(out_conv1)
        out_conv2 = F.relu(self.conv2(out_maxpool1))
        out_maxpool2 = self.maxpool2(out_conv2)
        out_flatten = self.flatten(out_maxpool2)
        out_fc1 = self.dropout(F.relu(self.fc1(out_flatten)))
        out = self.fc2(out_fc1)

        weights = F.softmax(out, dim=1)

        combined_output = np.sum(
            [(weights[:, i].repeat(self.num_classes)
                .reshape((self.num_classes, weights.size()[0])).T * logits_experts[i])
             for i in range(self.num_experts)]
        )

        if output_only:
            return combined_output
        else:
            return {'output': combined_output, 'weights': weights}



    def evaluate_experts(self, test_data, batch_size, criterion):
        results = []
        for expert in self.experts:
            results.append(expert.evaluate(test_data, batch_size, criterion))

        return results


    def analyze_decisions(self, test_data, batch_size=256):
        self.to(self.device)
        weights = np.zeros(self.num_experts)
        dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
        self.eval()
        with torch.no_grad():
            for (images, labels) in dataloader:
                results = self.forward(images, False)
                weights += results['weights'].sum(dim=0).cpu().numpy()
        return weights / len(test_data)

    def get_highest_weighted_examples(self, test_data):
        self.to(self.device)
        columns = ['sample_index']
        for i in range(self.num_experts):
            columns += 'weight expert ' + str(i)
        df = pd.DataFrame(columns=['sample_index'])
        dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
        self.eval()
        with torch.no_grad():
            for i, (images, labels) in enumerate(dataloader):
                results = self.forward(images, False)
                result = {'sample_index' : i}
                result['label'] = int(labels.cpu().item())
                result['prediction'] = results['output'].argmax(dim=1).cpu().numpy()
                for i in range(self.num_experts):
                    result['weight expert ' + str(i)] = results['weights'][0][i].cpu().numpy()
                df = df.append(result, ignore_index=True)
        return df
