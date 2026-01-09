# Enhanced DeepLabV3+ for Maritime Scene Segmentation
📌 This repository is archived on Zenodo with DOI: https://doi.org/10.5281/zenodo.18194849

This repository contains the official implementation of the paper:

**Enhanced DeepLabV3+ for Maritime Scene Segmentation: Attention Fusion and Horizon Learning**  
Submitted to *The Visual Computer*.

## Overview
This work proposes an enhanced DeepLabV3+ architecture for maritime semantic segmentation, integrating:
- Attention-guided feature fusion
- Horizon-aware auxiliary learning
- Boundary refinement

The method is evaluated on the LaRS (Lakes, Rivers, and Seas) dataset.

Please download LaRS from the official source:
https://lojzezust.github.io/lars-dataset/

## Requirements
- Python 3.8+
- PyTorch >= 1.10
- CUDA 11.x (recommended)

Install dependencies:
```bash
pip install -r requirements.txt

