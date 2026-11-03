# SpikeFusion-GI 2.1

**Spiking Neural Network + CNN + Bidirectional Vision Mamba for GI Disease Classification**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> A lightweight (~0.9M parameters), **explainable** deep-learning model that fuses **biologically-inspired retinal encoding**, **spiking neural networks** (sparse edge processing), **ECA-attention CNNs** (colour/texture), and **bidirectional Vision Mamba** (global spatial context) to classify 8 gastrointestinal diseases from endoscopy images.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Key Features](#key-features)
- [What's New in v2.1](#whats-new-in-v21)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Results](#results)
- [Explainable AI (XAI)](#explainable-ai-xai)
- [References](#references)

---

## Overview

**SpikeFusion-GI** is designed for computer-aided diagnosis of GI diseases from endoscopy images. Unlike standard CNNs or heavy transformers, it combines three complementary pathways:

1. **Retinal Encoder → SNN** — Mimics the human visual pathway (retina → LGN → V1) with center-surround (DoG), colour opponency, and orientation-selective Gabor filters, processed by sparse LIF spiking neurons.
2. **ECA-CNN** — Lightweight MobileNet-style network with Efficient Channel Attention for colour and texture.
3. **Bidirectional Vision Mamba** — Selective state-space model that scans the image in both directions for global context, in **linear time** (no quadratic attention cost).

**Target:** ~0.9M parameters, all task-relevant, no wasted decoder params.

---

## Architecture

```
RGB Image [3, 224, 224]
    ├─→ Retinal Encoder ───────────────────────┐
    │    (DoG, Gabor, R-G/B-Y opponency, Sobel)│
    │                              ↓            │
    │    LIF-SNN ──→ Edge Features [B, 32, H, W]│  (sparse, bio-plausible)
    │                                           │
    └─→ ECA-CNN ──→ Colour/Texture [B, 48, 7, 7]│ (MobileNet + channel attention)
                     ↘              ↗
                Lightweight Fusion (concat + 1×1 conv + SE gate)
                           ↓
                    Tokens [B, 49, 128]
                           ↓
            Bidirectional Vision Mamba (3 VSS blocks)
                           ↓
            Global Avg-Pool + LayerNorm + Classifier
                           ↓
               8 GI Disease Labels
```

### Design Philosophy

| Component | What it does | Why it matters for GI endoscopy |
|-----------|-------------|--------------------------------|
| **Retinal Encoder** | DoG (blob detection), Gabor (oriented edges), R-G/B-Y opponency | Polyps are blob-like + red; ulcers are dark pits + yellowish margins; boundaries are multi-oriented |
| **SNN** | Sparse spike processing | Energy-efficient; mimics biological edge processing; only fires on significant changes |
| **ECA-CNN** | Colour + texture with channel attention | Mucosal colour and vascular patterns are key diagnostic cues; ECA boosts salient channels for free |
| **Bi-Mamba** | Forward + backward selective SSM | Captures global spatial relationships (e.g., lesion ↔ anatomical landmarks) in linear time |
| **Fusion** | Cross-modal alignment + SE gate | Ensures edge and colour signals are jointly weighted before sequence modelling |

---

## Key Features

- **Ultra-lightweight:** ~0.9M parameters (vs. 8–31M for U-Net baselines)
- **Biologically-inspired preprocessing:** Retinal encoder with no learnable parameters — pure prior knowledge
- **Energy-efficient SNN:** Sparse spike computation (4 timesteps) for edge processing
- **Linear-time global modelling:** Bidirectional Mamba avoids O(n²) transformer attention
- **ECA attention:** Adds <0.5K parameters but significantly boosts CNN representational power
- **Stochastic depth:** Regularises Mamba blocks during training (drop_path = 0.1)
- **Full XAI pipeline:** Grad-CAM, **SpikeCAM**, retinal channel visualisation, SNN spike rasters, Mamba state heatmaps, prediction confidence bars
- **Production-ready:** AMP, Mixup/CutMix, SWA (Stochastic Weight Averaging), cosine LR, early stopping, checkpointing

---

## What's New in v2.1

| Feature | v2.0 | v2.1 |
|---------|------|------|
| Retinal preprocessing | Basic Sobel only | **DoG + Gabor + color opponency + Sobel** (10 channels) |
| CNN attention | None | **ECA (Efficient Channel Attention)** |
| Mamba direction | Unidirectional | **Bidirectional (VSS Block)** with fusion gate |
| Mamba regularisation | None | **Stochastic depth (drop_path)** |
| Fusion attention | Heavy MultiheadAttention (65K params) | **Lightweight SE gate (~4K params)** |
| SNN XAI | Spike raster only | **SpikeCAM** (class-attentive spike heatmaps) |
| Training augmentations | Mixup only | **Mixup + CutMix + RandAugment-style** |
| Model averaging | None | **SWA (last 10 epochs)** |
| Training epochs | 100 | **120** (with longer patience) |

---

## Installation

```bash
cd spikefusion-gi
pip install -r requirements.txt
```

### Requirements

- Python >= 3.9
- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- See `requirements.txt` for full list

---

## Dataset

**Kvasir v2** — 8,000 GI endoscopy images across 8 classes:

| Class | Description |
|-------|-------------|
| dyed-lifted-polyps | Dyed and lifted polyps |
| dyed-resection-margins | Resection margins after polyp removal |
| esophagitis | Inflammation of the esophagus |
| normal-cecum | Normal cecum anatomy |
| normal-pylorus | Normal pylorus anatomy |
| normal-z-line | Normal Z-line (esophago-gastric junction) |
| polyps | Polyps (non-dyed) |
| ulcerative-colitis | Ulcerative colitis lesions |

### Download

```bash
wget https://datasets.simula.no/kvasir/kvasir.zip
unzip kvasir.zip -d ./data/kvasir_v2
```

Expected directory structure:
```
data/kvasir_v2/
├── dyed-lifted-polyps/
├── dyed-resection-margins/
├── esophagitis/
├── normal-cecum/
├── normal-pylorus/
├── normal-z-line/
├── polyps/
└── ulcerative-colitis/
```

---

## Usage

### Local upload interface

The interface is built with plain HTML, CSS, and JavaScript; Flask connects the page to the Python model. Before unlocking admin controls, copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and set your own `APP_PASSWORD`. Photo and video uploads stay open without the passcode; model and frame-selection settings require admin access. The local secrets file is ignored by Git. Install the requirements and start the website:

```bash
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000` in your browser. An admin can upload a `.pth`, `.pt`, or PyTorch `.zip` checkpoint. A valid checkpoint is saved locally as `checkpoints/active_model.pth` and loaded automatically on later runs; a new valid upload replaces the active checkpoint. The Photo view classifies one uploaded image and shows a Grad-CAM heatmap. The Video view compares low-resolution versions of nearby frames and sends selected frames to the model when the view changes, with a maximum-time-gap fallback. The model classifies frames one at a time; it is not a video model. Video results show timestamps, labels, and confidence. The summary counts frame predictions; it is not a disease probability for the whole video.

### 1. Training

```bash
python train.py \
    --data-root ./data/kvasir_v2 \
    --augment both \
    --num-workers 4
```

**Training features:**
- 70/15/15 train/val/test split
- Data augmentation: random crop, flips, rotation, colour jitter, **Mixup/CutMix**
- Automatic Mixed Precision (AMP)
- Cosine annealing with 5-epoch warmup
- **Stochastic Weight Averaging (SWA)** — averages model weights over last 10 epochs for better generalisation
- Early stopping with patience=20
- Best model checkpointing to `./checkpoints/best_model.pth`

### 2. Evaluation

```bash
python evaluate.py \
    --checkpoint ./checkpoints/best_model.pth \
    --data-root ./data/kvasir_v2
```

### 3. Inference + XAI

```bash
python infer.py \
    --image ./data/kvasir_v2/polyps/image_001.jpg \
    --checkpoint ./checkpoints/best_model.pth \
    --output-dir ./xai_outputs
```

This generates **8 visualisation files**:
- `gradcam.png` — CNN Grad-CAM heatmap + overlay
- `spikecam.png` — **SNN SpikeCAM** (class-attentive spike heatmap) + overlay
- `retinal_channels.png` — All 10 retinal encoder channels (DoG, Gabor, color opponency, Sobel)
- `spike_raster.png` — SNN spike raster plot
- `spike_rate.png` — Average firing rate spatial heatmap
- `tokens_input.png` — Fused token representation heatmap
- `mamba_evolution.png` — Token norm evolution across Mamba blocks
- `fusion_map.png` — Cross-modal fusion feature map
- `prediction_bar.png` — Per-class prediction confidence bar chart

---

## Results

With the v2.1 architecture on Kvasir v2, expected performance:

| Metric | Expected Range |
|--------|---------------|
| Accuracy | 94–97% |
| F1 (macro) | 93–96% |
| Sensitivity (mean) | 92–95% |
| Specificity (mean) | 98–99% |
| AUC (macro) | 98–99% |

*These estimates reflect the improved retinal encoder, ECA attention, bidirectional Mamba, and SWA. Exact results depend on random seed and hardware.*

---

## Explainable AI (XAI)

SpikeFusion-GI is designed as an **XAI-first** system. Every branch produces interpretable outputs:

### 1. Grad-CAM (CNN Branch)
- **What:** Where the CNN "looks" for its decision
- **Clinical value:** Validates the model focuses on lesions, not artefacts
- **File:** `xai/gradcam.py`

### 2. SpikeCAM (SNN Branch) — **NEW in v2.1**
- **What:** Class-discriminative heatmap from spike-rate features via backprop through surrogate gradients
- **Clinical value:** Shows which edge patterns trigger spikes that drive the classification
- **File:** `xai/spikecam.py`

### 3. Retinal Channel Visualisation — **NEW in v2.1**
- **What:** All 10 retinal encoder channels (DoG, Gabor, R-G, B-Y, Sobel)
- **Clinical value:** Doctors can verify which biological features (redness, blob shape, edge orientation) are most active
- **File:** `xai/spike_viz.py`

### 4. SNN Spike Visualisation
- **What:** Spike timing diagrams + firing rate heatmaps
- **Clinical value:** Sparse firing indicates biological plausibility and energy efficiency
- **File:** `xai/spike_viz.py`

### 5. Mamba State Evolution
- **What:** How token representations evolve across bidirectional Mamba blocks
- **Clinical value:** Shows global context propagation across the image
- **File:** `xai/attention_viz.py`

### 6. Prediction Confidence
- **What:** Per-class probability distribution
- **Clinical value:** Low-confidence cases are flagged for human expert review
- **File:** `xai/explain.py`

---

## References

### Core Architecture
1. **Gu & Dao**, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" — *ICML 2024 Spotlight*
2. **Liu et al.**, "Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model" — *arXiv:2401.09417*
3. **Neftci et al.**, "Surrogate Gradient Learning in Spiking Neural Networks" — *IEEE SPM, 2019*
4. **Sandler et al.**, "MobileNetV2: Inverted Residuals and Linear Bottlenecks" — *CVPR 2018*
5. **Wang et al.**, "ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks" — *CVPR 2020*

### Biological Inspiration
6. **Hubel & Wiesel**, "Receptive fields of single neurones in the cat's striate cortex" — *J. Physiology, 1959*
7. **Daugman**, "Uncertainty relation for resolution in space, spatial frequency, and orientation optimized by two-dimensional visual cortical filters" — *JOSA A, 1985*

### XAI / Explainability
8. **Selvaraju et al.**, "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization" — *ICCV 2017*
9. **Sundararajan et al.**, "Axiomatic Attribution for Deep Networks" — *ICML 2017*

### Dataset
10. **Pogorelov et al.**, "Kvasir: A Multi-Class Image Dataset for Computer Aided Gastrointestinal Disease Detection" — *ACM MMSys 2017*

---

## Project Structure

```
spikefusion-gi/
├── frontend/
│   ├── static/                   # CSS styles and browser JavaScript
│   └── templates/                # HTML pages
├── backend/
│   ├── __init__.py
│   └── app.py                    # Flask routes, uploads, and model calls
├── configs/
│   └── config.py                  # Hyperparameters
├── models/
│   ├── __init__.py
│   ├── spikefusion.py             # Main model assembly
│   ├── retinal_encoder.py         # NEW: DoG + Gabor + color opponency
│   ├── snn_branch.py              # LIF-SNN + retinal input
│   ├── cnn_branch.py              # ECA-attention CNN
│   ├── fusion.py                  # Lightweight SE fusion
│   └── mamba_core.py              # Bidirectional VSS Mamba
├── data/
│   ├── __init__.py
│   └── kvasir_dataset.py          # Dataset loader + transforms
├── utils/
│   ├── __init__.py
│   └── metrics.py                 # Metrics, logging, checkpointing
├── xai/
│   ├── __init__.py
│   ├── gradcam.py                 # CNN Grad-CAM
│   ├── spikecam.py                # NEW: SNN SpikeCAM
│   ├── spike_viz.py               # Spike raster + retinal channels
│   ├── attention_viz.py           # Mamba token/state heatmaps
│   └── explain.py                 # Unified XAI pipeline
├── train.py                       # Training (SWA, Mixup, CutMix)
├── evaluate.py                    # Test evaluation
├── infer.py                       # Inference + XAI
├── checkpoints/                  # Active model checkpoint (shared in this repository)
├── references/                   # Project reference material
├── requirements.txt
├── app.py                        # Simple command to start the website
└── README.md
```

---

## Citation

```bibtex
@misc{spikefusion-gi,
  title={SpikeFusion-GI: Spiking-CNN-Mamba Fusion with Retinal Encoding for GI Disease Classification},
  author={Your Name},
  year={2025},
  howpublished={\url{https://github.com/your-repo/spikefusion-gi}}
}
```

---

## License

MIT License
