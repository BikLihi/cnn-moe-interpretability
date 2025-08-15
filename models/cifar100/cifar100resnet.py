import copy
import time

from scipy.stats import entropy

import numpy as np
import torch
import torch.nn as nn

from models.base_model import BaseModel
from utils.cifar100_utils import CIFAR100_LABELS, CIFAR100_ENCODING, CIFAR100_DECODING


class ResNetBlock(BaseModel):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        self.stride = stride

        if (stride != 1) or (in_channels != out_channels):
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, bias=False, padding=1),
                nn.BatchNorm2d(out_channels)
            )


    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        x = self.shortcut(x)
        out += x
        out = self.relu(out)

        return out


class CifarResNet(BaseModel):
    def __init__(self, classes=None, name=None, unknown_class=False, in_channels=3):
        super().__init__()

        self.classes = classes
        if self.classes is None:
            self.classes = CIFAR100_LABELS

        self.name = name
        self.unknown_class = unknown_class
        self.in_channels = in_channels
        self.num_classes = len(self.classes) + self.unknown_class
        self.target_encoding = CIFAR100_ENCODING
        self.target_decoding = CIFAR100_DECODING

        self.conv1 = nn.Conv2d(self.in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.02)

        self.block2 = nn.Sequential(ResNetBlock(32, 32, stride=1), ResNetBlock(32, 32, stride=1))
        self.block3 = nn.Sequential(ResNetBlock(32, 64, stride=2), ResNetBlock(64, 64, stride=1), ResNetBlock(64, 64, stride=1), ResNetBlock(64, 64, stride=1))
        self.block4 = nn.Sequential(ResNetBlock(64, 128, stride=2), ResNetBlock(128, 128, stride=1), ResNetBlock(128, 128, stride=1), ResNetBlock(128, 128, stride=1))
        self.block5 = nn.Sequential(ResNetBlock(128, 256, stride=2), ResNetBlock(256, 256, stride=1))

        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc = nn.Linear(256, self.num_classes)


    def forward(self, x):
        out = self.conv1(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        # out = self.avgpool(out)
        out = self.maxpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return {'output': out}

    
    def extract_features(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        return out


    def fit(
            self,
            training_data,
            validation_data,
            num_epochs=50,
            batch_size=128,
            device="cuda:0",
            criterion=torch.nn.CrossEntropyLoss(),
            optimizer=torch.optim.Adam,
            scheduler=None,
            learning_rate=0.01,
            save_state_path=None,
            return_best_model=True):

        optimizer = optimizer(self.parameters(), learning_rate)
        start_time = time.time()
        self.to(device)
        dataloaders = dict()
        dataloaders['training'] = torch.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=True)
        dataloaders['validation'] = torch.utils.data.DataLoader(validation_data, batch_size=batch_size, shuffle=False)
        dataset_sizes = dict()
        dataset_sizes['training'] = len(training_data)
        dataset_sizes['validation'] = len(validation_data)

        best_accuracy = 0.0
        best_model = None

        if self.name:
            print('Training of', self.name)
        print('Training on device:', device)
        print('Training on {:,} samples'.format(dataset_sizes['training']))
        print('Validation on {:,} samples'.format(dataset_sizes['validation']))
        print('Trainable parameters:', self.count_parameters())
        print()

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch + 1, num_epochs))
            print('-' * 10)

            for phase in ['training', 'validation']:
                if phase == 'training':
                    self.train()
                else:
                    self.eval()

                running_loss = 0.0
                running_corrects_top1 = 0
                running_corrects_top5 = 0

                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'training'):
                        outputs = self(inputs)['output']
                        loss = criterion(outputs, labels)
                        pred_top5 = torch.topk(outputs, 5, dim=1).indices
                        pred_top1 = pred_top5[:, 0]

                        if phase == 'training':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects_top1 += torch.sum(pred_top1 == labels)

                    for i in range(labels.size()[0]):
                        if labels[i] in pred_top5[i]:
                            running_corrects_top5 += 1

                if phase == 'training' and scheduler:
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_top1 = running_corrects_top1.double() / dataset_sizes[phase]
                epoch_top5 = float(running_corrects_top5) / dataset_sizes[phase]

                print('{} Loss: {:.4f}  Top1 Accuracy: {:.4f}  Top5 Accuracy: {:.4f}'.format(
                    phase, epoch_loss, epoch_top1, epoch_top5))

                if (phase == 'validation' and epoch_top1 > best_accuracy) and return_best_model:
                    best_model = copy.deepcopy(self.state_dict())
                    best_accuracy = epoch_top1

            print()

        time_elapsed = time.time() - start_time
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best Top1 Accuracy on validation set: {0:.4f}'.format(best_accuracy))

        if best_model:
            self.load_state_dict(best_model)

        if save_state_path is not None:
            torch.save(self.state_dict(), save_state_path)
            print('Saved model state at ', save_state_path)

        print('------------------------------------ Finished Training ------------------------------------ \n \n')


    def evaluate(
            self,
            test_data,
            batch_size=128,
            criterion=torch.nn.CrossEntropyLoss(),
            device='cuda:0'):

        self.to(device)
        self.eval()
        dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

        running_loss = 0.0
        running_corrects_top1 = 0
        running_corrects_top5 = 0
        mean_entropy = 0.0

        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = self(inputs)['output']
                pred_top5 = torch.topk(outputs, 5, dim=1).indices
                pred_top1 = pred_top5[:, 0]
                loss = criterion(outputs, labels)
                outputs_sm = torch.softmax(outputs, dim=1).cpu().numpy()
                mean_entropy += np.sum(entropy(outputs_sm, base=2, axis=1))

                running_loss += loss.item() * inputs.size(0)
                running_corrects_top1 += torch.sum(pred_top1 == labels)

                for i in range(labels.size()[0]):
                    if labels[i] in pred_top5[i]:
                        running_corrects_top5 += 1

        test_loss = np.round(running_loss / len(test_data), decimals=4)
        test_top1 = float(running_corrects_top1.item()) / len(test_data)
        test_top5 = float(running_corrects_top5) / len(test_data)
        mean_entropy = np.round(mean_entropy / len(test_data), decimals=4)

        result = {'loss': test_loss, 'top1': test_top1, 'top5': test_top5, 'entropy': mean_entropy}
        if self.name:
            result['name'] = self.name
        return result

