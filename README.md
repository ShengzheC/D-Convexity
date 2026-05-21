# D-Convexity

Official implementation of the **CVPR 2026 accepted paper**

> **D-Convexity: A Unified Differentiable Convex Shape Prior via Quasi-Concavity for Data-driven Image Segmentation**

[[arXiv](https://arxiv.org/abs/2605.19210v1)]

## Overview

This repository provides the official implementation of **D-Convexity**, a unified differentiable convex shape prior based on quasi-concavity for data-driven image segmentation.

The method is designed to encourage convexity-aware segmentation while remaining compatible with neural networks that output probability masks.

## Quick Start

For a quick start and intuition behind the convexification algorithm, please refer to:

```text
Convexification_Algorithm.ipynb
```

This notebook also contains the implementation of the **zero-order algorithm**.

## CGPM for Segmentation Framework

The CGPM segmentation framework is implemented in:

```text
CGPM.py
```

And the first, second losses are implemented in:

```text
loss.py
```

## Usage

### 1. Midpoint Convexification Algorithm

Open the notebook:

```bash
jupyter notebook Convexification_Algorithm.ipynb
```

or

```bash
jupyter lab Convexification_Algorithm.ipynb
```

### 2. CGPM Module

Import the CGPM module in your segmentation pipeline:

```python
from CGPM import SegModelWithCGPM
```

The CGPM module can be used with any segmentation model that produces a probability mask output.

Example workflow:

```python
# model: any segmentation network
# image: input image tensor

model = UNet2D().to(device)
model.load_state_dict(ckpt)
model.eval()
SegCGPM = SegModelWithCGPM(model, backprop_to_backbone=False)
cgpm_output = SegCGPM(images)
```

Please refer to `CGPM.py` for details on initialization and usage.
```
