import torch.nn as nn
import torch
from models.base_model import BaseModel
from models.moe_layer.resnet.bottleneck_expert import ResNet50Expert
from models.resnet.resnet18 import ResNet18
from utils.cifar100_utils import CIFAR100_LABELS
from losses.importance_loss import importance_loss, importance
from losses.kullback_leibler_divergence import kl_divergence
from losses.variance_loss import variance_loss

import copy
import time
import os
from metrics.accuracy import Accuracy
import numpy as np
from ptflops import get_model_complexity_info
import wandb

class GlobalMoELayer(BaseModel):
    def __init__(self, experts, top_k):
        super().__init__()
        self.experts = experts
        self.num_experts = len(experts)
        self.top_k = top_k
    
    def forward(self, x, weights=None, output_only=True):
        for expert in self.experts:
            expert.to(self.device)

        if self.num_experts == 1:
            out = self.experts[0](x)
            if output_only:
                return out
            return {'output': out,
                    'aux_loss': 0.0,
                    'examples_per_expert': torch.Tensor([x.shape[0]]),
                    'expert_importance': torch.tensor([1.0])}

        if weights is None:
            weights = torch.tensor([1.0 / len(self.experts) for i in range(len(self.experts))])

        examples_per_expert = (weights > 0).sum(dim=0)
        expert_importance = importance(weights)

        mask = weights > 0
        results = []
        for i in range(self.num_experts):
            # select mask according to computed gates (conditional computing)
            mask_expert = mask[:, i]
            # apply mask to inputs
            expert_input = x[mask_expert]
            # compute outputs for selected examples
            expert_output = self.experts[i](expert_input).to(self.device)
            # calculate output shape
            output_shape = list(expert_output.shape)
            output_shape[0] = x.size()[0]
            # store expert results in matrix
            expert_result = torch.zeros(output_shape, device=self.device)
            expert_result[mask_expert] = expert_output
            # weight expert's results
            expert_weight = weights[:, i].reshape(
                expert_result.shape[0], 1, 1, 1).to(self.device)
            expert_result = expert_weight * expert_result
            results.append(expert_result)
        # Combining results
        out = torch.stack(results, dim=0).sum(dim=0)

        if output_only:
            return out
        else:
            return {'output': out,
                    'examples_per_expert': examples_per_expert,
                    'expert_importance': expert_importance}


class GlobalMoEGate(BaseModel):
    def __init__(self, expert_groups, use_noise=True, **kwargs):
        super().__init__()
        self.expert_groups = expert_groups
        self.num_experts = len([len(g) for g in expert_groups])
        self.use_noise = use_noise

        self.conv = nn.Conv2d(3, 256, kernel_size=3, stride=1)
        self.avgpool_1x1 = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.output_layers = nn.ModuleList([nn.Linear(256, len(g)) for g in self.expert_groups])
        self.softplus = nn.Softplus()

    def forward(self, x, output_only=True):
        out = self.conv(x)
        out = self.avgpool_1x1(out)
        out = self.flatten(out)
        outputs = []
        for output_layer in self.output_layers:
            outputs.append(output_layer(out))
        
        if output_only:
            return outputs
        else:
            return {'output': outputs}

    def compute_gating(self, x):
        gate_logits = self(x)
        weights = []
        var_losses = []

        for i in range(len(self.expert_groups)):
            if len(self.expert_groups[i]) == 1:
                weights.append(gate_logits[i], x.shape[0])
                
            # Apply noise during training time if parameter is set
            # if self.use_noise and self.training:
            #     pool_out = self.avgpool_1x1(x)
            #     flatten_out = self.flatten(pool_out)
            #     std = self.softplus(self.w_noise(flatten_out) + 1e-2)
            #     noise = torch.randn_like(gate_logits[i], device=self.device, requires_grad=True) * std
            #     gate_logits[i] += noise
            
            var_losses.append(variance_loss(nn.functional.softmax(gate_logits[i], dim=1)))
            top_k_logits, top_k_indices = gate_logits[i].topk(self.expert_groups[i].top_k, dim=1)
            top_k_weights = nn.functional.softmax(top_k_logits, dim=1)

            weight_zeros = torch.zeros_like(gate_logits[i], device=self.device, requires_grad=True)
            weights.append(weight_zeros.scatter(1, top_k_indices, top_k_weights))
        return weights, var_losses


class InputMoEGate(BaseModel):
    def __init__(self, expert_groups, use_noise=True, **kwargs):
        super().__init__()
        self.expert_groups = expert_groups
        self.num_experts = len([len(g) for g in expert_groups])
        self.use_noise = use_noise

        self.conv1 = nn.Conv2d(3, 256, kernel_size=3, stride=1)
        self.avgpool_1x1 = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.output_layers = nn.ModuleList([nn.Linear(256, len(g)) for g in self.expert_groups])
        self.softplus = nn.Softplus()

    def forward(self, x, output_only=True):
        out = self.conv1(x)
        out = self.avgpool_1x1(out)
        out = self.flatten(out)
        outputs = []
        for output_layer in self.output_layers:
            outputs.append(output_layer(out))
        
        if output_only:
            return outputs
        else:
            return {'output': outputs}

    def compute_gating(self, x):
        gate_logits = self(x)
        weights = []
        var_losses = []

        for i in range(len(self.expert_groups)):
            if len(self.expert_groups[i]) == 1:
                weights.append(gate_logits[i], x.shape[0])
                
            #Apply noise during training time if parameter is set
            # if self.use_noise and self.training:
            #     pool_out = self.avgpool_1x1(x)
            #     flatten_out = self.flatten(pool_out)
            #     std = self.softplus(self.w_noise(flatten_out) + 1e-2)
            #     noise = torch.randn_like(gate_logits[i], device=self.device, requires_grad=True) * std
            #     gate_logits[i] += noise
            
            var_losses.append(variance_loss(nn.functional.softmax(gate_logits[i], dim=1)))
            top_k_logits, top_k_indices = gate_logits[i].topk(self.expert_groups[i].top_k, dim=1)
            top_k_weights = nn.functional.softmax(top_k_logits, dim=1)

            weight_zeros = torch.zeros_like(gate_logits[i], device=self.device, requires_grad=True)
            weights.append(weight_zeros.scatter(1, top_k_indices, top_k_weights))
        return weights, var_losses


class ComplexGlobalMoEGate(BaseModel):
    def __init__(self, expert_groups, use_noise=True, **kwargs):
        super().__init__()
        self.expert_groups = expert_groups
        self.num_experts = len([len(g) for g in expert_groups])
        self.use_noise = use_noise

        self.conv1 = nn.Conv2d(3, 256, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=30, stride=1)
        self.flatten = nn.Flatten()
        self.output_layers = nn.ModuleList([nn.Linear(256, len(g)) for g in self.expert_groups])
        self.softplus = nn.Softplus()

    def forward(self, x, output_only=True):
        out = self.conv1(x)
        out = self.conv2(out)
        #out = self.avgpool_1x1(out)
        out = self.flatten(out)
        outputs = []
        for output_layer in self.output_layers:
            outputs.append(output_layer(out))
        
        if output_only:
            return outputs
        else:
            return {'output': outputs}

    def compute_gating(self, x):
        gate_logits = self(x)
        weights = []
        var_losses = []

        for i in range(len(self.expert_groups)):
            if len(self.expert_groups[i]) == 1:
                weights.append(gate_logits[i], x.shape[0])
                
            # Apply noise during training time if parameter is set
            # if self.use_noise and self.training:
            #     pool_out = self.avgpool_1x1(x)
            #     flatten_out = self.flatten(pool_out)
            #     std = self.softplus(self.w_noise(flatten_out) + 1e-2)
            #     noise = torch.randn_like(gate_logits[i], device=self.device, requires_grad=True) * std
            #     gate_logits[i] += noise
            
            var_losses.append(variance_loss(nn.functional.softmax(gate_logits[i], dim=1)))
            top_k_logits, top_k_indices = gate_logits[i].topk(self.expert_groups[i].top_k, dim=1)
            top_k_weights = nn.functional.softmax(top_k_logits, dim=1)

            weight_zeros = torch.zeros_like(gate_logits[i], device=self.device, requires_grad=True)
            weights.append(weight_zeros.scatter(1, top_k_indices, top_k_weights))
        return weights, var_losses



 
class ResNet18GlobalMoE(BaseModel):
    def __init__(self, layer_positions, expert_classes, num_experts, top_k, gating_network, loss_fkts, w_aux_losses, w_var_losses, name='ResNet50GlobalMoE'):
        super().__init__(name=name)

        # Assure that number of arguments is identical for each parameter
        assert(len(layer_positions) == len(expert_classes))
        assert(len(layer_positions) == len(num_experts))
        self.num_experts = num_experts
        self.top_k = top_k
        self.loss_fkts = loss_fkts
        self.w_aux_losses = w_aux_losses
        self.w_var_losses = w_var_losses

        # Basic input layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()

        # Build default resnet50 layer
        resnet = ResNet18(classes=CIFAR100_LABELS)
        self.layers = nn.ModuleList([None, None, None, None])

        expert_groups = []

        for i, pos in enumerate(layer_positions):
            experts = []    
            for _ in range(num_experts[i]):
                experts.append(expert_classes[i](
                    pos, name='Expert'
                ))
            experts = nn.ModuleList(experts)
            experts.top_k = top_k[i]
            expert_groups.append(experts)
            self.layers[pos - 1] = GlobalMoELayer(experts, top_k[i])
        
        # Insert MoE Layer
        for i in range(4):
            if self.layers[i] is None:
                self.layers[i] = resnet.layers[i]

        # Output layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=512, out_features=100, bias=True)

        # Gating network
        self.gate = gating_network(expert_groups)

    def forward(self, x, output_only=True):
        aux_loss = torch.tensor(0.0, device=self.device)
        examples_per_expert = dict()
        expert_importance= dict()

        # Compute gating weights and aux losses for all layers
        weights, var_losses = self.gate.compute_gating(x)
        var_loss = np.sum(np.array(var_losses) * np.array(self.w_var_losses))
        for i, w in enumerate(weights):
            if self.loss_fkts[i] == 'importance':
                aux_loss += importance_loss(w) * self.w_aux_losses[i]
            elif self.loss_fkts[i] == 'kl_divergence':
                aux_loss += kl_divergence(w, self.num_experts[i]) * self.w_aux_losses[i]
        # Compute first layers
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)


        layer_counter = 0
        if output_only:
            for layer in self.layers:
                if type(layer) is GlobalMoELayer:
                    out = layer(out, weights=weights[layer_counter])
                    layer_counter += 1
                else:
                    out = layer(out)
            out = self.avgpool(out)
            out = self.flatten(out)
            out = self.fc(out)
            return out

        layer_counter = 0
        for layer in self.layers:
            if type(layer) is GlobalMoELayer:
                moe_output = layer(out, weights=weights[layer_counter], output_only=False)
                examples_per_expert[layer_counter] = moe_output['examples_per_expert']
                expert_importance[layer_counter] = moe_output['expert_importance']
                layer_counter += 1
                out = moe_output['output']
            else:
                out = layer(out)

        out = self.avgpool(out)
        out = self.flatten(out)
        out = self.fc(out)
        return {'output': out, 'aux_loss': aux_loss, 'var_loss': var_loss, 'examples_per_expert': examples_per_expert, 'expert_importance': expert_importance}


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
                running_var_loss = 0.0
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
                        loss += model_output['aux_loss'] + model_output['var_loss']
                        running_aux_loss += model_output['aux_loss']
                        running_var_loss += model_output['var_loss']
                        

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
                epoch_var_loss = running_var_loss / float(num_batches)
                epoch_acc = acc.compute_metric()

                # Log metrics and losses with W&B
                if enable_logging:
                    wandb.log({
                        'Accuracy ' + phase: epoch_acc,
                        'Loss ' + phase: epoch_loss,
                        'Auxiliary loss ' + phase: epoch_aux_loss,
                        'Variation loss' + phase: epoch_var_loss
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
            time.sleep(60)
            for file in garbage_files:
                os.remove(file)