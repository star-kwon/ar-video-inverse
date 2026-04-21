<p align="center">
<h1 align="center">Accelerating Video Inverse Problem Solvers with Autoregressive Diffusion Models</h1>

<div align="center">
  <h2>🚀 AVIS: Autoregressive Video Inverse problem Solver</h2>
  <p>An autoregressive video diffusion framework for <b>streaming video restoration</b>.</p>
</div>

<h3>✨ Key Features</h3>
<ul>
  <li><strong>Streaming Restoration:</strong> Naturally eliminates latency bottlenecks by leveraging autoregressive video diffusion models.</li>
  <li><strong>Initialization for Better Starting Point:</strong> Reduces sampling steps by initializing the reverse diffusion with a measurement-consistent estimate.</li>
</ul>

<h4>⚡ AVIS <em>Flash</em>: Highly Accelerated Variant</h3>
<!-- <p>A high-throughput version of AVIS designed to pave the way toward real-time deployment.</p> -->
<ul>
  <li><strong>Acceleration with Autoregressive Propagation:</strong> Enforces measurement consistency solely on the first video chunk. Subsequent chunks naturally align via autoregressive propagation from this corrected prefix, bypassing iterative VAE passes.</li>
  <li><strong>Improvements (on a single RTX 4090 GPU, compared to leading non-autoregressive solver):</strong>
    <ul>
      <li>Initial Latency: > 114s ➡️ <strong>4s</strong></li>
      <li>Throughput: < 0.71 FPS ➡️ <strong>5.91 FPS</strong></li>
    </ul>
  </li>
</ul>


## Requirements
We tested this repo on the following setup:
* Nvidia GPU with at least 24 GB memory (RTX 4090 and H100 are tested).
* Linux operating system.
* 64 GB RAM.

Other hardware setup could also work but hasn't been tested.

## Installation
Create a conda environment and install dependencies:
```
conda create -n avis python=3.10 -y
conda activate avis
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
python setup.py develop
```
Note: you may download assets in https://github.com/Dao-AILab/flash-attention/releases

## Quick Start
### Download checkpoints
```
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir-use-symlinks False --local-dir wan_models/Wan2.1-T2V-1.3B
huggingface-cli download gdhe17/Self-Forcing checkpoints/self_forcing_dmd.pt --local-dir .
```

### Inference
```
bash sample.sh
```
Note:
* **Our model works better with long, detailed prompts** since it's trained with such prompts. We will integrate prompt extension into the codebase (similar to [Wan2.1](https://github.com/Wan-Video/Wan2.1/tree/main?tab=readme-ov-file#2-using-prompt-extention)) in the future. For now, it is recommended to use third-party LLMs (such as GPT-4o) to extend your prompt before providing to the model.
* You may want to adjust FPS so it plays smoothly on your device.
* The speed can be improved by enabling `torch.compile`, [TAEHV-VAE](https://github.com/madebyollin/taehv/), or using FP8 Linear layers, although the latter two options may sacrifice quality. It is recommended to use `torch.compile` if possible and enable TAEHV-VAE if further speedup is needed.

### Evaluation
```
bash eval.sh
```

## Codebase
This codebase is built on top of the open-source implementation of [Self-Forcing](https://github.com/guandeh17/Self-Forcing) and the [Wan2.1](https://github.com/Wan-Video/Wan2.1).