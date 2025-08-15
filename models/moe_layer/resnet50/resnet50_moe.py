import torch.nn as nn
import torch
from models.base_model import BaseModel
from models.moe_layer.resnet.moe_block_layer import MoeBlockLayer
from models.resnet.resnet50 import ResNet50
from utils.cifar100_utils import CIFAR100_LABELS
from datasets.cifar100_dataset import CIFAR100Dataset
from utils.dataset_utils import get_transformation

import copy
import time
import os
from metrics.accuracy import Accuracy
import numpy as np
import pandas as pd
from ptflops import get_model_complexity_info
import wandb
import sys

class ResNet50MoE(BaseModel):
    def __init__(self, moe_layers, name='ResNet50MoE', lift_constraint=None):
        super().__init__(name=name)

        self.lift_constraint = lift_constraint
        # Input
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()

        resnet = ResNet50(classes=CIFAR100_LABELS)
        self.layers = nn.ModuleList([None, None, None, None])

        for moe_layer in moe_layers:
            self.layers[moe_layer.layer_position - 1] = moe_layer
        
        # Insert MoE Layer
        for i in range(4):
            if self.layers[i] is None:
                self.layers[i] = resnet.layers[i]

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=2048, out_features=100, bias=True)

    def forward(self, x, output_only=True):
        aux_loss = torch.tensor(0.0, device=self.device)
        examples_per_expert = dict()
        expert_importance= dict()
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if output_only:
            for layer in self.layers:
                out = layer(out)
            out = self.avgpool(out)
            out = self.flatten(out)
            out = self.fc(out)
            return out

        for i, layer in enumerate(self.layers):
            if type(layer) is MoeBlockLayer:
                moe_output = layer(out, output_only=False)
                aux_loss += moe_output['aux_loss']
                examples_per_expert[i] = moe_output['examples_per_expert']
                expert_importance[i] = moe_output['expert_importance']
                out = moe_output['output']
            else:
                out = layer(out)

        out = self.avgpool(out)
        out = self.flatten(out)
        out = self.fc(out)

        if self.lift_constraint is not None:
            if self._running_epoch >= self.lift_constraint:
                aux_loss = 0.0

        return {'output': out, 'aux_loss': aux_loss, 'examples_per_expert': examples_per_expert, 'expert_importance': expert_importance}


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
                           'Num training data': len(training_data[0][0].shape),
                           'Total parameters': self.count_parameters(only_trainable=False),
                           'Computational complexity': get_model_complexity_info(self, tuple(training_data[0][0].shape), as_strings=True,
                                                                                 print_per_layer_stat=False, verbose=False)[0]
                       })
            #wandb.watch(self)
            if validation_data:
                wandb.config.update({'Num validation data': len(validation_data)})
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
            self._running_epoch = epoch
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
                running_examples_per_expert = dict()
                running_expert_importance = dict()
                running_examples_cof = dict()
                running_importance_cof = dict()
                num_batches = torch.tensor(0)

                if metrics:
                    for metric in metrics:
                        metric.reset()

                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    num_batches += 1

                    with torch.set_grad_enabled(phase == 'training'):
                        optimizer.zero_grad()

                        # Forward pass
                        model_output = self(inputs, output_only=False)

                        # Calculate losses
                        loss = criterion(model_output['output'], labels)
                        if 'aux_loss' in model_output:
                            loss += model_output['aux_loss']
                            running_aux_loss += model_output['aux_loss']
                        

                        # Update metrics and loss
                        running_loss += loss.item()
                        acc.update(model_output, labels)
                        examples_per_expert = model_output['examples_per_expert']
                        for key in examples_per_expert:
                            if key in running_examples_per_expert:
                                running_examples_per_expert[key] += examples_per_expert[key]
                            else:
                                running_examples_per_expert[key] = examples_per_expert[key]

                        expert_importance = model_output['expert_importance']
                        for key in expert_importance:
                            if key in running_expert_importance:
                                running_expert_importance[key] += expert_importance[key]
                            else:
                                running_expert_importance[key] = expert_importance[key]

                        if metrics:
                            for metric in metrics:
                                metric.update(model_output, labels)

                        # Update weights and lr
                        if phase == 'training':
                            loss.backward()
                            optimizer.step()

                # Calulate metrics and loss
                epoch_loss = running_loss / float(num_batches)
                epoch_aux_loss = running_aux_loss / float(num_batches)
                epoch_acc = acc.compute_metric()

                # Log metrics and losses with W&B
                if enable_logging:
                    wandb.log({
                        'Accuracy ' + phase: epoch_acc,
                        'Loss ' + phase: epoch_loss,
                        'Auxiliary loss ' + phase: epoch_aux_loss,
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
                    for key in running_examples_per_expert:
                        for i, number_examples in enumerate(running_examples_per_expert[key].tolist()):
                            avg_examples = number_examples / \
                                dataset_sizes[phase]
                            wandb.log({'Avg. Share of examples ' + phase + ' layer_' + str(key + 1) + 
                                       '/expert_' + str(i): avg_examples}, step=epoch)
                        for i, importance in enumerate(running_expert_importance[key]):
                            avg_importance = importance / dataset_sizes[phase]
                            wandb.log({'Avg. importance ' + phase + ' layer_' + str(key + 1) + 
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
            # time.sleep(10)
            # for file in garbage_files:
            #     os.remove(file)


    def evaluate(
        self,
        test_data,
        batch_size=256,
        criterion=torch.nn.CrossEntropyLoss(),
        metrics=None,
        dataset_name=None
    ):

        # Move model to training device
        self.to(self.device)

        start_time = time.time()

        # Create DataLoader for training and validation data
        dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
        dataset_size = len(test_data)

        # Initialize metrics and early stopping
        acc = Accuracy()

        # Print basic training information
        print('------------------------------------ Beginning Evaluation ------------------------------------')
        if self.name:
            print('Evaluation of ', self.name)
        print('Evaluation on {:,} samples'.format(len(test_data)))
        if dataset_name:
            print('Evaluation on {}'.format(dataset_name))


        self.eval()

        # Reset metrics
        running_loss = 0.0
        running_aux_loss = 0.0
        acc.reset()
        num_batches = torch.tensor(0.0)

        if metrics:
            for metric in metrics:
                metric.reset()

        for inputs, labels in dataloader:
            inputs = inputs.to(self.device)
            if self.target_encoding:
                labels.apply_(lambda x: self.target_encoding[x])
            labels = labels.to(self.device)
            num_batches += 1

            with torch.no_grad():
                # Forward pass
                model_output = self(inputs, output_only=False)

                # Calculate losses
                loss = criterion(model_output['output'], labels)
                if 'aux_loss' in model_output:
                    loss += model_output['aux_loss']
                    running_aux_loss += model_output['aux_loss']
                    

                # Update metrics and loss
                running_loss += loss.item()
                acc.update(model_output, labels)
                if metrics:
                    for metric in metrics:
                        metric.update(model_output, labels)


        # Calulate metrics and loss
        loss = running_loss / num_batches
        aux_loss = running_aux_loss / num_batches
        acc = acc.compute_metric()
                
        time_elapsed = time.time() - start_time
        time_elapsed_string = time.strftime(
            "%H:%M:%S", time.gmtime(time_elapsed))
        print('Evaluation complete in ', time_elapsed_string)
        print('Evaluation Accuracy: {0:.4f}'.format(acc))
        print('------------------------------------ Finished Evaluation ------------------------------------\n')

        result_dict = dict()
        result_dict['loss'] = loss
        result_dict['aux_loss'] = aux_loss
        result_dict['acc'] = acc

        return result_dict

    
    def predict(self, data, batch_size=256, target_decoding=None):
        self.to(self.device)
        self.eval()

        pred_tensors = torch.Tensor().to(self.device)
        data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False)

        with torch.no_grad():
            for images, _ in data_loader:
                images = images.to(self.device)
                output = self(images)
                pred_tensors = torch.cat([pred_tensors, output], dim=0)

        pred = torch.argmax(pred_tensors, dim=1).cpu().numpy()

        if target_decoding:
            pred = np.array([self.target_decoding[x] for x in pred])

        return pred

    
    def get_gating_weights(
        self,
        test_data,
        moe_layer,
        batch_size=256,
        top_k=sys.maxsize
    ):

        class SaveWeights:
            def __init__(self, layer):
                self.weights = torch.empty([0, 4], device=layer.device)
                self.hook = layer.register_forward_hook(self.hook_function)
            def hook_function(self, module, input, output):
                self.weights = torch.cat((self.weights, output['weights']))
            def close(self):
                self.hook.remove()

        
        sv = SaveWeights(moe_layer)
        # Move model to training device
        self.to(self.device)

        # Create DataLoader for training and validation data
        dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
        dataset_size = len(test_data)

        self.eval()

        for inputs, labels in dataloader:
            inputs = inputs.to(self.device)
            if self.target_encoding:
                labels.apply_(lambda x: self.target_encoding[x])
            labels = labels.to(self.device)

            with torch.no_grad():
                # Forward pass
                model_output = self(inputs, output_only=False)
        weights_split = torch.split(sv.weights, [1, 1, 1, 1], dim=1)
        indices_list = []
        weight_list = []
        for expert in range(len(weights_split)):
            weights = weights_split[expert].squeeze().cpu().numpy()
            top_indices = (-np.array(weights)).argsort()[:top_k]
            indices_list.append(top_indices)
            weight_list.append(weights)
        sv.close()
        return indices_list, weight_list


    def mean_weights_per_class(self, moe_layer, batch_size=256):
        transformations_test = get_transformation('cifar100', phase='test')
        result = []
        for label in CIFAR100_LABELS:
            test_data = CIFAR100Dataset(root_dir='/home/lb4653/mixture-of-experts-thesis/data/cifar100/testing', transform=transformations_test, labels=[label])
            indices_list, weight_list = self.get_gating_weights(test_data, moe_layer)
            mean_weights = np.mean(weight_list, axis=1)
            result.append([label, *mean_weights])
        df = pd.DataFrame(result, columns=['label', *['expert ' + str(i) for i in range(moe_layer.num_experts)]])
        return df