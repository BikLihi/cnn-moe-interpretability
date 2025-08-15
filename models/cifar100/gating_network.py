import copy
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import entropy

from utils.cifar100_dataset import CIFAR100Dataset

from models.basenet import BaseModel

from torchsummary import summary


class GatingNetwork(BaseModel):
    def __init__(self, experts, classes=None, name=None, unknown_class=False, feature_extractor=None, shared_feature_extractor=True):
        super().__init__()
        self.classes = classes
        if self.classes is None:
            self.classes = CIFAR100Dataset.CIFAR100_LABELS
        self.experts = experts
        self.num_experts = len(self.experts)
        self.name = name
        self.unknown_class = unknown_class
        self.feature_extractor = feature_extractor
        self.shared_feature_extractor = shared_feature_extractor
        self.num_classes = len(self.classes) + unknown_class

        self.conv1 = nn.Conv2d(128, 256, 3, stride=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(256, self.num_experts)


    def forward(self, x, device='cuda:0'):
        logits_experts = []
        shared_features = self.feature_extractor.extract_features(x)


        for expert in self.experts:
            expert.set_parameter_requires_grad(False)
            expert.eval()
            expert = expert.to(device)
            if self.shared_feature_extractor:
                logit_expert = expert(shared_features, extracted_features_input=True)
            else:
                logit_expert = expert(x)

            if expert.unknown_class:
                logit_expert = logit_expert[:, :-1]
            
            if torch.cuda.is_available():
                decoded_logits = torch.cuda.FloatTensor(logit_expert.size()[0], self.num_classes).fill_(0)
            else:
                decoded_logits = torch.zeros(logit_expert.size()[0], self.num_classes)

            for i in range(logit_expert.size()[1]):
                decoded_logits[:, expert.target_decoding[i]] = logit_expert[:, i]
            logits_experts.append(decoded_logits)

        out = self.conv1(shared_features)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.avgpool(out)
        out = self.flatten(out)
        out = self.fc(out)

        weights = F.softmax(out, dim=1)
        weights = weights.to(device)

        combined_output = np.sum(
            [(weights[:, i].repeat(self.num_classes)
                .reshape((self.num_classes, weights.size()[0])).T * logits_experts[i])
             for i in range(self.num_experts)]
        )

        return {'output': combined_output, 'weights': weights}


    # def fit(
    #         self,
    #         training_data,
    #         validation_data,
    #         num_epochs=50,
    #         batch_size=128,
    #         device="cuda:0",
    #         criterion=torch.nn.CrossEntropyLoss(),
    #         optimizer=torch.optim.Adam,
    #         scheduler=None,
    #         learning_rate=0.01,
    #         save_state_path=None,
    #         return_best_model=True):

    #     optimizer = optimizer(self.parameters(), learning_rate)
    #     start_time = time.time()
    #     self.to(device)
    #     for expert in self.experts:
    #         expert.to(device)

    #     dataloaders = dict()
    #     dataloaders['training'] = torch.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=True)
    #     dataloaders['validation'] = torch.utils.data.DataLoader(validation_data, batch_size=batch_size, shuffle=False)
    #     dataset_sizes = dict()
    #     dataset_sizes['training'] = len(training_data)
    #     dataset_sizes['validation'] = len(validation_data)

    #     best_accuracy = 0.0
    #     best_model = None

    #     if self.name:
    #         print('Training of', self.name)
    #     print('Training on device:', device)
    #     print('Training on {:,} samples'.format(dataset_sizes['training']))
    #     print('Validation on {:,} samples'.format(dataset_sizes['validation']))
    #     print('Number of parameters: {:,}'.format(self.count_parameters()))
    #     print()

    #     for epoch in range(num_epochs):
    #         print('Epoch {}/{}'.format(epoch + 1, num_epochs))
    #         print('-' * 10)

    #         for phase in ['training', 'validation']:
    #             if phase == 'training':
    #                 self.train()
    #             else:
    #                 self.eval()

    #             running_loss = 0.0
    #             running_corrects_top1 = 0
    #             running_corrects_top5 = 0

    #             for inputs, labels in dataloaders[phase]:
    #                 inputs = inputs.to(device)
    #                 labels = labels.to(device)

    #                 optimizer.zero_grad()

    #                 with torch.set_grad_enabled(phase == 'training'):
    #                     outputs = self(inputs)[0]
    #                     loss = criterion(outputs, labels)
    #                     pred_top5 = torch.topk(outputs, 5, dim=1).indices
    #                     pred_top1 = pred_top5[:, 0]

    #                     if phase == 'training':
    #                         loss.backward()
    #                         optimizer.step()

    #                 running_loss += loss.item() * inputs.size(0)
    #                 running_corrects_top1 += torch.sum(pred_top1 == labels)

    #                 for i in range(labels.size()[0]):
    #                     if labels[i] in pred_top5[i]:
    #                         running_corrects_top5 += 1

    #             if phase == 'training' and scheduler:
    #                 scheduler.step()

    #             epoch_loss = running_loss / dataset_sizes[phase]
    #             epoch_top1 = running_corrects_top1.double() / dataset_sizes[phase]
    #             epoch_top5 = float(running_corrects_top5) / dataset_sizes[phase]

    #             print('{} Loss: {:.4f}  Top1 Accuracy: {:.4f}  Top5 Accuracy: {:.4f}'.format(
    #                 phase, epoch_loss, epoch_top1, epoch_top5))

    #             if (phase == 'validation' and epoch_top1 > best_accuracy) and return_best_model:
    #                 best_model = copy.deepcopy(self.state_dict())
    #                 best_accuracy = epoch_top1

    #         print()

    #     time_elapsed = time.time() - start_time
    #     print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    #     print('Best Top1 Accuracy on validation set: {0:.4f}'.format(best_accuracy))

    #     if best_model:
    #         self.load_state_dict(best_model)

    #     if save_state_path is not None:
    #         torch.save(self.state_dict(), save_state_path)
    #         print('Saved model state at ', save_state_path)

    #     print('------------------------------------ Finished Training ------------------------------------ \n \n')


    def evaluate(
            self, 
            test_data,    
            batch_size=128,
            device="cuda:0",
            criterion=torch.nn.CrossEntropyLoss(),
    ):
        self.to(device)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

        self.eval()
        running_test_loss = 0.0
        running_test_corrects = 0
        mean_entropy = 0.0

        with torch.no_grad():
            for _, data in enumerate(test_loader):
                inputs, labels = data
                inputs = inputs.to(device)

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


    def evaluate_experts(self, test_data, batch_size, criterion, device):
        results = []
        for expert in self.experts:
            results.append(expert.evaluate(test_data, batch_size, criterion, device))

        return results