<div align="center">

# [NeurIPS 2026] Accelerating Video Inverse Problem Solvers with Autoregressive Diffusion Models
  
<p>
  <a href="https://arxiv.org/abs/2605.20624"><img src="https://img.shields.io/badge/Paper-arXiv-b31b1b.svg" alt="Paper"></a>
  <a href="https://avis-project.github.io/"><img src="https://img.shields.io/badge/Project-Page-4b9be8.svg" alt="Project Page"></a>
</p>
  
**AVIS (Autoregressive Video Inverse problem Solver)** is an autoregressive video diffusion framework designed for **accelerating video restoration** toward real-time deployment.

</div>

---

## ✨ Key Features

### 🪄 AVIS: Autoregressive Video Inverse problem Solver

- **Autoregressive Restoration:** Naturally eliminates latency bottlenecks by leveraging autoregressive video diffusion models, enabling continuous processing.
- **Better Initialization:** Reduces sampling steps by initializing the reverse diffusion process with a measurement-consistent estimate, providing a much better starting point.

### ⚡ AVIS *Flash*: Highly Accelerated Variant

To pave the way toward real-time deployment, we additionally introduce **AVIS *Flash***, a high-throughput variant that maximizes efficiency.

**Performance Comparison (Single RTX 4090 GPU)**

| Metric | Leading Non-Autoregressive Solvers | AVIS *Flash* (Ours) | Improvement |
| :--- | :---: | :---: | :---: |
| **Initial Latency** | > 114.0 s | **4.0 s** | **~28x Faster** |
| **Throughput** | < 0.71 FPS | **5.91 FPS** | **~8x Faster** |

---

## 🛠️ Getting Started

### Requirements
- **GPU:** NVIDIA GPU with at least 24GB VRAM (Tested on RTX 4090 and H100)
- **OS:** Linux
- **RAM:** 32GB
> *Note: Other hardware setups may work but have not been formally tested.*

### Installation
Create a conda environment and install the required dependencies:
```
conda create -n avis python=3.10 -y
conda activate avis
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
python setup.py develop
```
> *Note: If you encounter issues with flash-attn, you can download the pre-built wheels directly from the [Dao-AILab/flash-attention releases](https://github.com/Dao-AILab/flash-attention/releases).*

## 🚀 Quick Start

### 1. Download Checkpoints
We use weights from Wan2.1 and Self-Forcing. Download them to your local directory:

```bash
# Download Wan2.1 Checkpoints
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B \
  --local-dir-use-symlinks False \
  --local-dir wan_models/Wan2.1-T2V-1.3B

# Download Self-Forcing DMD Checkpoints
huggingface-cli download gdhe17/Self-Forcing checkpoints/self_forcing_dmd.pt \
  --local-dir .
```

### 2. Sample Restoration
Run the causal restoration script to start video restoration:

```bash
bash causal_restoration.sh
```
> *Note 1: Please place `.mp4` videos in the `./data` directory. The pipeline will automatically degrade these videos for restoration.*

> *Note 2: This repo provides AVIS Flash. To implement AVIS, simply modify the code to apply the measurement update to all tokens.*

### 3. Evaluation
Evaluate the restored outputs using the provided script:

```bash
bash eval.sh
```
> *Note: This script calculates fidelity and perceptual metrics (e.g., PSNR, SSIM, LPIPS, FID, FVD) are calculated here. For VBench evaluation, please refer to their [official implementation](https://github.com/Vchitect/VBench).*


## 💻 Codebase Attribution

This codebase is built upon the open-source implementations of:
- [Self-Forcing](https://github.com/guandeh17/Self-Forcing)
- [Wan2.1](https://github.com/Wan-Video/Wan2.1)
