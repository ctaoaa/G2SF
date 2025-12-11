<!-- 顶栏徽章 -->
<p align="center">
  <a href="https://arxiv.org/abs/{{2503.10091}}"><img src="https://img.shields.io/badge/arXiv-{{2503.10091}}-b31b1b.svg?style=flat&logo=arxiv" alt="arXiv"></a>
  <a href="https://opensource.org/licenses/{{license}}"><img src="https://img.shields.io/badge/License-{{license}}-green.svg?style=flat" alt="License"></a>
  <a href="https://github.com/{ctaoaa}/{G2SF}"><img src="https://img.shields.io/github/v/release/{ctaoaa}/{G2SF}?include_prereleases&style=flat&logo=github" alt="GitHub release"></a>
  <a href="https://github.com/{ctaoaa}/{G2SF}/issues"><img src="https://img.shields.io/github/issues/{ctaoaa}/{G2SF}?style=flat&logo=github" alt="Issues"></a>
</p>

<p align="center">
  <a href="https://github.com/ctaoaa/G2SF/raw/main/framework.pdf">
    <img src="https://img.shields.io/badge/PDF-Framework-1f7ede?style=flat&logo=adobe-acrobat" alt="Framework PDF"/>
  </a>
</p>


<h1 align="center">G$^2$SF: Geometry-Guided Score Fusion for Multimodal Industrial Anomaly Detection</h1>
<h3 align="center">ICCV 2025</h3>

<p align="center">
  <strong>Chengyu Tao<sup>1</sup>, Xuanming Cao<sup>2</sup>, Juan Du<sup>1,2</sup></strong>
</p>
<p align="center">
  <sup>1</sup>The Hong Kong University of Science and Technology &emsp; <sup>2</sup>The Hong Kong University of Science and Technology (Guangzhou)
</p>

<p align="center">
  📧 Corresponding: <a href="mailto:{{email}}">ctaoaa@connect.ust.hk</a>
</p>


<!-- 主图 -->
<p align="center">
  <img src="framework.png" width="90%">
</p>
---

## 🎯 Overview

This repository contains the official PyTorch implementation of our *G^2SF* accepted at *ICCV 2025*.

**Method**: A systematic G$^{2}$SF framework for industrial multimodal anomaly detection by learning a unified discriminative metric in high-dimensional feature space.

**State-of-the-art performance**:
| Dataset   | I-AUROC  | P-AUROC  | AUPRO@30% | AUPRO@1% |
|-----------|----------|----------|---------- |----------|
|MVTec-3D AD|   97.1   |   99.7   |    97.9   |   46.8   |
|Eyecandies |   90.2   |   98.2   |    89.8   |   35.7   |

*Results from our official paper*

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/ctaoaa/G2SF.git
cd G2SF

# Create conda environment (recommended)
conda create -n G2SF python=3.9
conda activate G2SF
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia

# Install pointnet2-ops 0.3.0 (If any trouble, please go to the repo [Pointnet2_PyTorch](https://github.com/erikwijmans/Pointnet2_PyTorch/tree/master)
pip install pointnet2_ops_lib/. --no-build-isolation

# Install dependencies
pip install -r requirements.txt
```



## 📁 Repository Structure
.
├── configs/               # Configuration files
│   ├── train.yaml        # Training configuration
│   └── eval.yaml         # Evaluation configuration
├── data/                 # Data loading utilities
│   ├── datasets.py       # Dataset classes
│   └── transforms.py     # Data transformations
├── models/               # Model architectures
│   ├── __init__.py
│   ├── backbone.py       # Backbone networks
│   ├── head.py          # Task-specific heads
│   └── losses.py        # Loss functions
├── scripts/              # Training/evaluation scripts
│   ├── train.py
│   ├── evaluate.py
│   └── demo.py
├── utils/                # Utilities
│   ├── logger.py
│   ├── metrics.py
│   └── visualization.py
├── notebooks/            # Jupyter notebooks
│   └── demo.ipynb       # Interactive demo
├── experiments/          # Experiment logs and checkpoints
├── requirements.txt      # Python dependencies
├── setup.py             # Package setup
└── LICENSE
.
Checkpoints/  
├── dino_.pth        # RGB feature extractor
└── pointmae_.pth    # Point cloud feature extractor
Dataset/                 # Data loading utilities
├── __init__.py
├── create_anomaly_source.py  # Collecting indices of samples for pseudo anomaly generation
├── cut_paste.py         # Cut-and-paste anomaly synthesis
├── eyecandies.py        # dataloader for Eyecandies dataset
├── eyecandies_pseudo.py # dataloader for Eyecandies dataset with pseudo anomalies
├── mvtec3d.py           # dataloader for MvTec3D-AD dataset
├── mvtec3d_pseudo.py    # dataloader for MvTec3D-AD dataset with pseudo anomalies
├── mvtec3d_util.py      # utilities
├── perlin.py            # Perlin noise for anomaly masks
└── util.py              # utilities

├── requirements.txt      # Python dependencies

