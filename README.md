<!-- 顶栏徽章 -->
<p align="center">
  <a href="https://arxiv.org/abs/{{2503.10091}}"><img src="https://img.shields.io/badge/arXiv-{{2503.10091}}-b31b1b.svg?style=flat&logo=arxiv" alt="arXiv"></a>
  <a href="https://opensource.org/licenses/{{license}}"><img src="https://img.shields.io/badge/License-{{license}}-green.svg?style=flat" alt="License"></a>
  <a href="https://github.com/{ctaoaa}/{G2SF}/releases"><img src="https://img.shields.io/github/v/release/{ctaoaa}/{G2SF}?include_prereleases&style=flat&logo=github" alt="GitHub release"></a>
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

**Key Contributions:**
- **Method/Model Name**: A systematic G$^{2}$SF framework for industrial multimodal anomaly detection by learning a unified discriminative metric in high-dimensional feature space.
- **State-of-the-art performance**: MVTec-3D AD dataset: 97.1% I-AUROC, 99.7% P-AUROC, 97.9% AUPRO@30%, 46.8% AUPRO@1%;
                                    Eyecandies dataset: 90.2% I-AUROC, 98.2% P-AUROC, 89.8% AUPRO@30%, 35.7% AUPRO@1%;
*Results from the paper (more details in Table X)*

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/ctaoaa/G2SF.git
cd G2SF

# Create conda environment (recommended)
conda create -n G2SF python=3.9
conda activate G2SF


# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
