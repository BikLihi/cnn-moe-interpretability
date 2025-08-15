# CNN-MoE-Interpretability

This repository contains code for reproducing experiments and analysis from the paper:

**"Sparsely-gated Mixture-of-Expert Layers for CNN Interpretability"**  
by Svetlana Pavlitska et al.

## 🎯 Project Goals

The project focuses on reproducing the following results from the paper:

- **Figure 3**: t-SNE visualizations of gate logits for a ResBlock-MoE with 4 experts and a GAP-FC gate.
- **Figure 4**: t-SNE visualization of sample assignments to experts.
- **Table III**: Class-wise expert weight allocation showing which classes each expert specializes in.

## 🗂 Repository Structure

```bash
.
├── models/
│   └── moe_layer/resnet18/          # ResBlock-MoE architecture
├── experiments/
│   └── cifar100/                    # Training scripts for CIFAR-100
├── analysis/resnet18/
│   ├── tsne_visualization.py       # Reproduces Fig. 3 and Fig. 4
│   ├── expert_utilization.py       # Reproduces Table III
│   └── per_expert_accuracy.py      # Optional: Accuracy per expert
├── datasets/
│   └── cifar100_dataset.py         # CIFAR-100 loading logic
├── requirements.txt                # Dependencies
