import copy
import time

from collections import defaultdict
from scipy.stats import entropy

from models.base_model import BaseModel

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import wandb
import copy
import time
import torch
import torch.nn as nn
import os
from abc import abstractmethod
from scipy.stats import entropy
from metrics.accuracy import Accuracy
import numpy as np
from ptflops import get_model_complexity_info

import wandb

class MnistNet(BaseModel):
    """Convolutional Neural Network classifier for MNIST-like datasets with a variable number
        of classes  and the possibility to add an additional class for unknown labels.
        Every label during training which is not contained in the classes-argument will be
        considered unknown if option is set True.
    """

    def __init__(self, classes, name=None, unknown_class=False):
        """Constructs all necessary attributes for the classifier.

        Arguments:
            classes {list} -- List of all class labels, no mentioning of unknown_class

        Keyword Arguments:
            name {string} -- Name of the classifier (default: {None})
            unknown_class {bool} -- Adds an additional class for unknown labels (default: {None})
        """
        super().__init__()
        self.classes = classes
        self.name = name
        self.unknown_class = unknown_class
        self.num_classes = len(classes) + self.unknown_class

        # Encoding labels into labels [0, ..., N-1]. Using a defaultdict in case of unknown_class
        if unknown_class:
            self.target_encoding = defaultdict(lambda: self.num_classes - 1)
            self.target_decoding = defaultdict(lambda: 'unknown')
        else:
            self.target_encoding = dict()
            self.target_decoding = dict()

        for index, cls in enumerate(self.classes):
            self.target_encoding[cls] = index
        for index, cls in enumerate(self.classes):
            self.target_decoding[index] = cls

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(7 * 7 * 64, 128)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(128, self.num_classes)


    def forward(self, x, output_only=True):
        out_conv1 = F.relu(self.conv1(x))
        out_maxpool1 = self.maxpool1(out_conv1)
        out_conv2 = F.relu(self.conv2(out_maxpool1))
        out_maxpool2 = self.maxpool2(out_conv2)
        out_flatten = self.flatten(out_maxpool2)
        out_fc1 = self.dropout(F.relu(self.fc1(out_flatten)))
        out_fc2 = self.fc2(out_fc1)
        if output_only:
            return out_fc2
        return {'output' : out_fc2, 'maxpool':out_maxpool2}

    #             for inputs, labels in dataloaders[phase]:
    #                 inputs = inputs.to(self.device)
    #                 #labels.apply_(lambda x: self.target_encoding[x])
    #                 labels = labels.to(self.device)
    #                 num_batches += 1

    #                 with torch.set_grad_enabled(phase == 'training'):
    #                     optimizer.zero_grad()

    #                     # Forward pass
    #                     model_output = self(inputs, output_only=False)

    #                     # Calculate losses
    #                     loss = criterion(model_output['output'], labels)
    #                     if 'aux_loss' in model_output:
    #                         loss += model_output['aux_loss']
    #                         running_aux_loss += model_output['aux_loss']
                        

    #                     # Update metrics and loss
    #                     running_loss += loss.item()
    #                     acc.update(model_output, labels)
    #                     if 'examples_per_expert' in model_output:
    #                         examples_per_expert = model_output['examples_per_expert']
    #                         running_examples_cof += examples_per_expert.float().std() / examples_per_expert.float().mean()
    #                         if running_examples_per_expert is not None:
    #                             running_examples_per_expert += examples_per_expert
    #                         else:
    #                             running_examples_per_expert = examples_per_expert

    #                     if 'expert_importance' in model_output:
    #                         expert_importance = model_output['expert_importance']
    #                         running_importance_cof += expert_importance.float().std() / expert_importance.float().mean()
    #                         if running_expert_importance is not None:
    #                             running_expert_importance += expert_importance
    #                         else:
    #                             running_expert_importance = expert_importance

    #                     if metrics:
    #                         for metric in metrics:
    #                             metric.update(model_output, labels)

    #                     # Update weights and lr
    #                     if phase == 'training':
    #                         loss.backward()
    #                         optimizer.step()

    #             # Calulate metrics and loss
    #             epoch_loss = running_loss / num_batches
    #             epoch_aux_loss = running_aux_loss / num_batches
    #             epoch_acc = acc.compute_metric()
    #             epoch_examples_cof = running_examples_cof / num_batches
    #             epoch_importance_cof = running_importance_cof / num_batches

    #             # Log metrics and losses with W&B
    #             if enable_logging:
    #                 wandb.log({
    #                     'Accuracy ' + phase: epoch_acc,
    #                     'Loss ' + phase: epoch_loss,
    #                     'Auxiliary loss ' + phase: epoch_aux_loss,
    #                     'Examples per batch and expert CV ' + phase: epoch_examples_cof.item(),
    #                     'Expert Importance per batch CV ' + phase: epoch_importance_cof.item()
    #                 }, step=epoch)

    #                 if phase == 'training':
    #                     wandb.log(
    #                         {'Learning rate ' + phase: optimizer.param_groups[0]['lr']}, step=epoch)

    #                 if metrics:
    #                     for metric in metrics:
    #                         metric_result = metric.compute_metric()
    #                         wandb.log(
    #                             {metric.name + ' ' + phase: metric_result}, step=epoch)

    #             # Calculate and log avg examples per expert and importance
    #             if enable_logging:
    #                 if running_examples_per_expert is not None:
    #                     for i, number_examples in enumerate(running_examples_per_expert.tolist()):
    #                         avg_examples = number_examples / \
    #                             dataset_sizes[phase]
    #                         wandb.log({'Avg. Share of examples ' + phase +
    #                                 '/expert_' + str(i): avg_examples}, step=epoch)
    #                 if running_expert_importance is not None:
    #                     for i, importance in enumerate(running_expert_importance):
    #                         avg_importance = importance / dataset_sizes[phase]
    #                         wandb.log({'Avg. importance ' + phase +
    #                                 '/expert_' + str(i): avg_importance}, step=epoch)

    #             if (phase == 'validation' and epoch_acc > best_acc) and return_best_model:
    #                 best_model = copy.deepcopy(self.state_dict())
    #                 best_acc = epoch_acc
    #                 n_epochs_best_model = epoch
    #                 epochs_no_improve = 0

    #             if phase == 'validation' and epoch_acc < best_acc:
    #                 epochs_no_improve += 1

    #             if running_aux_loss != 0.0:
    #                 print('{} Loss: {:.4f}  Aux Loss: {:.4f}  Accuracy: {:.4f}'.format(
    #                     phase, epoch_loss, epoch_aux_loss, epoch_acc))
    #             else:
    #                 print('{} Loss: {:.4f}  Accuracy: {:.4f}'.format(
    #                     phase, epoch_loss, epoch_acc))
    #         print()

    #         # Update learning rate
    #         if lr_scheduler:
    #             lr_scheduler.step()

    #         # Save model checkpoint with W&B
    #         if enable_logging and wandb_checkpoints is not None:
    #             if wandb_checkpoints != 0 and epoch % wandb_checkpoints == 0 and epoch > 0:
    #                 filename = self.name + '_epoch_' + str(epoch) + '.tar'
    #                 torch.save({
    #                     'epoch': epoch,
    #                     'model_state_dict': self.state_dict(),
    #                     'optimizer_state_dict': optimizer.state_dict(),
    #                 }, filename)
    #                 wandb.save(filename)
    #                 garbage_files.append(filename)

    #         # Use early stopping
    #         if epochs_no_improve > n_epochs_stop and early_stopping:
    #             print('Early stopping at epoch ', str(epoch))
    #             break

    #     time_elapsed = time.time() - start_time
    #     time_elapsed_string = time.strftime(
    #         "%H:%M:%S", time.gmtime(time_elapsed))
    #     print('Training complete in ', time_elapsed_string)
    #     print('Best accuracy on validation set: {0:.4f}'.format(best_acc))
    #     print('------------------------------------ Finished Training ------------------------------------\n')

    #     if enable_logging:
    #         wandb.config.update({
    #             'best_acc': best_acc,
    #             'num_epochs_best_model': n_epochs_best_model,
    #             'total_training_time': time_elapsed_string
    #         })

    #     # Load model with best accuracy on validation set
    #     if best_model:
    #         self.load_state_dict(best_model)

    #     # Save final / best model at W&B
    #     if enable_logging:
    #         filename = self.name + '_final.tar'
    #         torch.save({
    #             'epoch': epoch,
    #             'model_state_dict': self.state_dict(),
    #             'optimizer_state_dict': optimizer.state_dict(),
    #         }, filename)
    #         wandb.save(filename)
    #         garbage_files.append(filename)

    #     # Finish logging for the current run
    #     wandb.join()

    #     # Save best model to .pth-file
    #     if save_state_path is not None:
    #         torch.save(self.state_dict(), save_state_path)
    #         print('Saved model state at ', save_state_path)

    #     if enable_logging:
    #         for file in garbage_files:
    #             os.remove(file)


    # def evaluate(
    #         self,
    #         test_data,
    #         batch_size=128,
    #         criterion=torch.nn.CrossEntropyLoss(),
    #         device="cuda:0"):
    #     """Validate the classifier with test data. Computes loss, accuracy and entropy.

    #     Arguments:
    #         test_data {map-style dataset of iterable-style dataset} -- [description]
    #         batch_size {int} -- batch size for training and validation
    #         device {string} -- device on which tensors will be processed
    #         criterion {torch.nn.modules.loss or python function} -- Loss function

    #     Returns:
    #         [dict] -- evaluation results
    #     """
    #     self.to(device)
    #     dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

    #     self.eval()
    #     running_loss = 0.0
    #     running_corrects = 0
    #     mean_entropy = 0.0

    #     with torch.no_grad():
    #         for inputs, labels in dataloader:
    #             inputs = inputs.to(device)
    #             labels.apply_(lambda x: self.target_encoding[x])
    #             labels = labels.to(device)

    #             outputs = self(inputs)
    #             pred = torch.argmax(outputs, dim=1)
    #             loss = criterion(outputs, labels)

    #             outputs_sm = torch.softmax(outputs, dim=1).cpu().numpy()
    #             mean_entropy += np.sum(entropy(outputs_sm, base=2, axis=1))

    #             running_loss += loss.item() * inputs.size(0)
    #             running_corrects += torch.sum(pred == labels.data)
    #     test_loss = np.round(running_loss / len(test_data), decimals=4)
    #     test_acc = float(running_corrects.item()) / len(test_data)
    #     mean_entropy = np.round(mean_entropy / len(test_data), decimals=4)

    #     result = {'loss': test_loss, 'acc': test_acc, 'entropy': mean_entropy}
    #     if self.name:
    #         result['name'] = self.name

    #     return result

    # def predict(self, data, device, batch_size=32):
    #     """Predicts the labels of a dataset.

    #     Arguments:
    #         data {map-style dataset of iterable-style dataset} -- data to apply classifier on
    #         device {string} -- device on which tensors will be processed

    #     Keyword Arguments:
    #         criterion {torch.nn.modules.loss or python function} -- Loss function

    #     Returns:
    #         [n.array] -- predicted labels
    #     """
    #     self.to(device)
    #     self.eval()
    #     pred_tensors = torch.Tensor().to(device)
    #     data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False)

    #     with torch.no_grad():
    #         for _, (images, _) in enumerate(data_loader):
    #             images = images.to(device)
    #             logits = self(images)['output']
    #             pred_tensors = torch.cat([pred_tensors, logits], dim=0)

    #     pred = np.argmax(pred_tensors.cpu().numpy(), axis=1)

    #     pred = np.array([self.target_decoding[x] for x in pred])

    #     return pred


    def fit(
            self,
            training_data,
            validation_data=None,
            test_data=None,
            num_epochs=50,
            batch_size=256,
            criterion=torch.nn.CrossEntropyLoss(),
            metrics=None,
            optimizer=None,
            learning_rate=0.01,
            early_stopping=False,
            return_best_model=True,
            lr_scheduler=None,
            save_state_path=None,
            enable_logging=False,
            wandb_project=None,
            wandb_checkpoints=None,
            wandb_name=None,
            wandb_notes=None,
            wandb_dir=None,
            wandb_tags=None,
            wandb_custom_config=None,
    ):

        # Assert that a lr scheduler always comes in pair with an optimizer
        assert not (lr_scheduler is not None and optimizer is None)

        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters(), learning_rate)

        # Move model to training device
        self.to(self.device)

        # Initialize new W&B run and declare hyperparameters
        if enable_logging:
            wandb.init(project=wandb_project, name=wandb_name,
                       notes=wandb_notes, dir=wandb_dir, tags=wandb_tags, reinit=True,
                       config={
                           'Num epochs': num_epochs,
                           'Batch size': batch_size,
                           'Initial lr': learning_rate,
                           'Optimizer': optimizer,
                           'Num training data': len(training_data),
                           'Total parameters': self.count_parameters(only_trainable=False),
                           'Computational complexity': get_model_complexity_info(self, tuple(training_data[0][0].shape), as_strings=True,
                                                                                 print_per_layer_stat=False, verbose=False)[0]
                       })
            if validation_data:
                wandb.config.update({'Num validation data': len(validation_data)})
            wandb.watch(self)
            wandb.config.update(self.custom_config)
            garbage_files = []

            if wandb_custom_config:
                wandb.config.update(wandb_custom_config, allow_val_change=True)

        start_time = time.time()

        # Create DataLoader for training and validation data
        dataloaders = dict()
        dataset_sizes = dict()
        dataloaders['training'] = torch.utils.data.DataLoader(
            training_data, batch_size=batch_size, shuffle=True)
        dataset_sizes['training'] = len(training_data)
        if validation_data:
            dataloaders['validation'] = torch.utils.data.DataLoader(validation_data, batch_size=batch_size, shuffle=False)
            dataset_sizes['validation'] = len(validation_data)

        # Initialize metrics and early stopping
        acc = Accuracy()
        best_acc = 0.0
        best_model = None
        n_epochs_best_model = 0
        epochs_no_improve = 0
        n_epochs_stop = 10

        # Print basic training information
        print('------------------------------------ Beginning Training ------------------------------------')
        if self.name:
            print('Training of ', self.name)
        print('Training on device: ', self.device)
        print('Training on {:,} samples'.format(len(training_data)))
        if validation_data:
            print('Validation on {:,} samples'.format(len(validation_data)))
        print('Trainable parameters {:,}:'.format(
            self.count_parameters(only_trainable=True)))
        print('Total parameters {:,}:'.format(
            self.count_parameters(only_trainable=False)))

        # Training/Validation cycle
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch + 1, num_epochs))
            print('-' * 10)

            for phase in ['training', 'validation']:
                if phase == 'training':
                    self.train()
                else:
                    self.eval()
                    if validation_data is None:
                        continue

                # Reset metrics
                running_loss = 0.0
                running_aux_loss = 0.0
                acc.reset()
                running_examples_per_expert = None
                running_expert_importance = None
                running_examples_cof = 0
                running_importance_cof = 0
                num_batches = torch.tensor(0)

                if metrics:
                    for metric in metrics:
                        metric.reset()

                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(self.device)
                    labels.apply_(lambda x: self.target_encoding[x])
                    labels = labels.to(self.device)
                    num_batches += 1

                    with torch.set_grad_enabled(phase == 'training'):
                        optimizer.zero_grad()

                        # Forward pass
                        model_output = self(inputs, output_only=False)

                        # Calculate losses
                        loss = criterion(model_output['output'], labels)

                        # Update metrics and loss
                        running_loss += loss.item()
                        acc.update(model_output, labels)

                        if metrics:
                            for metric in metrics:
                                metric.update(model_output, labels)

                        # Update weights and lr
                        if phase == 'training':
                            loss.backward()
                            optimizer.step()

                # Calulate metrics and loss
                epoch_loss = running_loss / num_batches
                epoch_aux_loss = running_aux_loss / num_batches
                epoch_acc = acc.compute_metric()
                epoch_examples_cof = running_examples_cof / num_batches
                epoch_importance_cof = running_importance_cof / num_batches

                # Log metrics and losses with W&B
                if enable_logging:
                    wandb.log({
                        'Accuracy ' + phase: epoch_acc,
                        'Loss ' + phase: epoch_loss,
                        'Auxiliary loss ' + phase: epoch_aux_loss,
                        'Examples per batch and expert CV ' + phase: epoch_examples_cof.item(),
                        'Expert Importance per batch CV ' + phase: epoch_importance_cof.item()
                    }, step=epoch)

                    if phase == 'training':
                        wandb.log(
                            {'Learning rate ' + phase: optimizer.param_groups[0]['lr']}, step=epoch)

                    if metrics:
                        for metric in metrics:
                            metric_result = metric.compute_metric()
                            wandb.log(
                                {metric.name + ' ' + phase: metric_result}, step=epoch)

                # Calculate and log avg examples per expert and importance
                if enable_logging:
                    if running_examples_per_expert is not None:
                        for i, number_examples in enumerate(running_examples_per_expert.tolist()):
                            avg_examples = number_examples / \
                                dataset_sizes[phase]
                            wandb.log({'Avg. Share of examples ' + phase +
                                       '/expert_' + str(i): avg_examples}, step=epoch)
                    if running_expert_importance is not None:
                        for i, importance in enumerate(running_expert_importance):
                            avg_importance = importance / dataset_sizes[phase]
                            wandb.log({'Avg. importance ' + phase +
                                       '/expert_' + str(i): avg_importance}, step=epoch)

                if (phase == 'validation' and epoch_acc > best_acc) and return_best_model:
                    best_model = copy.deepcopy(self.state_dict())
                    best_acc = epoch_acc
                    n_epochs_best_model = epoch
                    epochs_no_improve = 0

                if phase == 'validation' and epoch_acc < best_acc:
                    epochs_no_improve += 1

                if running_aux_loss != 0.0:
                    print('{} Loss: {:.4f}  Aux Loss: {:.4f}  Accuracy: {:.4f}'.format(
                        phase, epoch_loss, epoch_aux_loss, epoch_acc))
                else:
                    print('{} Loss: {:.4f}  Accuracy: {:.4f}'.format(
                        phase, epoch_loss, epoch_acc))
            print()

            # Update learning rate
            if lr_scheduler:
                lr_scheduler.step()

            # Save model checkpoint with W&B
            if enable_logging and wandb_checkpoints is not None:
                if wandb_checkpoints != 0 and epoch % wandb_checkpoints == 0 and epoch > 0:
                    filename = self.name + '_epoch_' + str(epoch) + '.tar'
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, filename)
                    wandb.save(filename)
                    garbage_files.append(filename)

            # Use early stopping
            if epochs_no_improve > n_epochs_stop and early_stopping:
                print('Early stopping at epoch ', str(epoch))
                break

        time_elapsed = time.time() - start_time
        time_elapsed_string = time.strftime(
            "%H:%M:%S", time.gmtime(time_elapsed))
        print('Training complete in ', time_elapsed_string)
        print('Best accuracy on validation set: {0:.4f}'.format(best_acc))
        print('------------------------------------ Finished Training ------------------------------------\n')

        if enable_logging:
            wandb.config.update({
                'best_acc': best_acc,
                'num_epochs_best_model': n_epochs_best_model,
                'total_training_time': time_elapsed_string
            })

        # Load model with best accuracy on validation set
        if best_model:
            self.load_state_dict(best_model)

        # Save final / best model at W&B
        if enable_logging:
            filename = self.name + '_final.tar'
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, filename)
            wandb.save(filename)
            garbage_files.append(filename)


        # Save best model to .pth-file
        if save_state_path is not None:
            torch.save(self.state_dict(), save_state_path)
            print('Saved model state at ', save_state_path)
        
        # Test trained model
        if test_data and enable_logging:
            if type(test_data) is dict:
                for key in test_data:
                    eval_result = self.evaluate(test_data[key], batch_size, criterion, metrics, key)
                    wandb.config.update({
                        key + '_acc': eval_result['acc']
                    })
            else:
                eval_result = self.evaluate(test_data, batch_size, criterion, metrics)
                wandb.config.update({
                    'test_acc': eval_result['acc']
                })

        # Finish logging for the current run
        if enable_logging:
            wandb.join()
            time.sleep(10)
            for file in garbage_files:
                os.remove(file)

