<div align="center">

# EAG3R: Event-Augmented 3D Geometry Estimation<br> for Dynamic and Extreme-Lighting Scenes

<h3>NeurIPS 2025 Spotlight</h3>

[![NeurIPS](https://img.shields.io/badge/NeurIPS%202025-Spotlight-red.svg)](https://openreview.net/pdf?id=Lf0W2gmNBg)
[![Website](https://img.shields.io/badge/Project-Website-blue)](https://candy-crusher.github.io/EAG3R_Proj/#) <br/>

<p>
  <em>EAG3R enables robust dense 3D reconstruction in extreme low-light and dynamic scenes where standard RGB methods fail, by effectively fusing asynchronous event streams with image data.</em>
</p>

</div>

## 📖 Introduction

**EAG3R** is a novel framework developed upon [MonST3R](https://github.com/junyi42/monst3r) that leverages the high temporal resolution and high dynamic range of **event cameras** to achieve robust 3D geometry estimation in challenging, real-world conditions.

Traditional RGB-only methods often struggle with severe motion blur in dynamic scenes or reduced visibility in extreme lighting (e.g., night driving). EAG3R addresses these limitations through two key technical contributions:

1.  **SNR-Aware Feature Fusion:** An adaptive mechanism that fuses RGB and event features based on a predicted Signal-to-Noise Ratio (SNR) map derived from a Retinex-inspired enhancement module. It learns to trust RGB in well-lit areas and events in dark or fast-moving regions.
2.  **Event-Based Photometric Consistency Loss:** A novel self-supervised loss function that constrains motion and geometry using event data, providing reliable supervision even when photometric assumptions fail in dark scenes.

Crucially, EAG3R demonstrates remarkable **Zero-Shot Generalization**: trained solely on daytime data, it achieves state-of-the-art performance on nighttime sequences without any fine-tuning.

## ✨ Key Features

- [x] **Event-Augmented Architecture**: Extends the pointmap-based MonST3R framework to process asynchronous event streams alongside RGB images.
- [x] **Retinex-inspired Enhancement & SNR Estimation**: Decomposes low-light images to recover visibility and estimates pixel-wise reliability (SNR).
- [x] **Adaptive SNR-Aware Fusion**: Dynamically modulates the interaction between image and event features using cross-attention based on the SNR map.
- [x] **Self-Supervised Event Loss**: Utilizes the physics of event generation to supervise network training in extreme lighting.
- [x] **Zero-Shot Nighttime Generalization**: Robust performance on unseen nighttime domains when trained only on daytime datasets.

## 🛠️ Getting Started

### 1. Installation

Follow these steps to set up the environment:

```bash
# 1. Clone the repository (ensure recursive for submodules like CroCo/DUSt3R)
git clone --recursive https://github.com/Candy-Crusher/EAG3R.git
cd EAG3R
# If you cloned without --recursive, run: git submodule update --init --recursive

# 2. Create and activate the conda environment
conda create -n eag3r python=3.11 cmake=3.14.0
conda activate eag3r

# 3. Install PyTorch (adjust CUDA version as needed for your system)
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia

# 4. Install dependencies
pip install -r requirements.txt
# Optional packages for training, evaluation, and dataset preparation
pip install -r requirements_optional.txt

# 5. (Optional) Compile CUDA kernels for RoPE (speeds up MonST3R backbone)
cd croco/models/curope/
python setup.py build_ext --inplace
cd ../../../

# 6. (Optional) Install viser for interactive 4D visualization
pip install -e viser
```

### 2. Download Checkpoints & Data
We provide pre-trained weights for EAG3R.

```Bash

# Download pretrained weights (EAG3R and optical flow models)
cd data
bash download_ckpt.sh
cd ..
```

Note: Information on preparing datasets like MVSEC should be added here or linked to a separate DATASET.md.

## 🧪 Evaluation
We provide scripts to evaluate camera pose estimation on event camera datasets, specifically supporting MVSEC for our zero-shot experiments.

### Zero-Shot Evaluation on MVSEC Night
The following command reproduces our core result: evaluating the model (trained only on daytime data) on the challenging outdoor_night1 sequence.

Ensure you have prepared the MVSEC dataset according to instructions.

```Bash

# Example: Evaluating on MVSEC outdoor_night1 sequence (Zero-Shot setting)
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29604 launch.py --mode=eval_pose \
    --pretrained="checkpoints/mvsec_best.pth" \
    --eval_dataset=mvsec \
    --output_dir="./results/mvsec_night1_eval" \
    --seq_list="outdoor_night/outdoor_night1" \
    --seqidx=0 \
    --use_event_control \
    --use_lowlight_enhancer \
    --event_enhance_mode="easy" \
    --use_snr_weighting \
    --snr_weight_strategy snr \
    --snr_weight_type region \
    --event_loss_weight=0.01
```

### Key Arguments Explained:

- --use_lowlight_enhancer & --use_snr_weighting: Activates the Retinex-based enhancement and the SNR-aware fusion mechanism, crucial for low-light performance.

- --use_event_control: Enables the event branch backbone.

- Refer to launch.py or arguments.py for a full description of all flags.

## 👁️ Visualization
You can visualize the generated dense 4D reconstruction interactively using viser.

```Bash
python viser/visualizer_monst3r.py --data ./results/mvsec_night1_eval/outdoor_night_outdoor_night1/
# Tips for better visualization:
# Remove foreground floaters: --init_conf --fg_conf_thre 1.0 (adjust threshold as needed)
```

## 🖊️ Citation
If you find our work useful in your research, please consider citing our NeurIPS 2025 paper:


```bibtex
@inproceedings{wu2025eag3r,
  title={EAG3R: Event-Augmented 3D Geometry Estimation for Dynamic and Extreme-Lighting Scenes},
  author={Wu, Xiaoshan and Yu, Yifei and Lyu, Xiaoyang and Huang, Yi-Hua and Zhang, Baoheng and Wang, Zhongrui and Wang, Bo and Qi, Xiaojuan},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2025}
}
```

## 🙏 Acknowledgements
This project is built upon the excellent open-source works of MonST3R and DUSt3R. We thank the authors for making their code available.