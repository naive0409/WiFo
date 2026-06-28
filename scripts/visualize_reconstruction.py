"""
WiFo2D 预训练可视化：原始 CSI vs 遮掩 vs 重构

用法:
    CUDA_VISIBLE_DEVICES=0 python scripts/visualize_reconstruction.py \
        --model_path ./experiments/csibench/debug_5k/wifo2d_tiny_best.pth \
        --size tiny

默认: --model_path ./experiments/csibench/wifo2d_tiny_best.pth, --size tiny
"""

import sys
import os
import argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model import WiFo2D_model
from csibench_pretrain_dataset import CSIBenchPretrainDataset

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default="./experiments/csibench/wifo2d_tiny_best.pth",
                    help='Path to trained model weights')
parser.add_argument('--size', type=str, default='tiny', choices=['tiny', 'small', 'base'],
                    help='Model size (must match training)')
parser.add_argument('--save_dir', type=str, default="./visualizations",
                    help='Output directory for the image')
parser.add_argument('--mask_ratio', type=float, default=0.80,
                    help='Mask ratio for visualization')
parser.add_argument('--n_samples', type=int, default=8,
                    help='Number of samples to visualize')
args = parser.parse_args()

os.makedirs(args.save_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# 加载模型
model_args = type('Args', (), {
    'size': args.size,
    'patch_size_time': 25,
    'patch_size_freq': 8,
})()
model = WiFo2D_model(args=model_args).to(device)
model.load_state_dict(torch.load(args.model_path, map_location=device))
model.eval()
print(f"Model loaded: {args.model_path}")

# 加载 16 个样本
dataset = CSIBenchPretrainDataset(max_samples=16)
loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False)

with torch.no_grad():
    batch = next(iter(loader)).to(device)  # (16, 1, 500, 232)

    # forward with masking
    loss, _, pred_patches, target_patches, mask = model(batch, mask_ratio=args.mask_ratio)

    # reconstruct full CSI
    B, C, T, F = batch.shape
    p_t, p_f = 25, 8
    n_t, n_f = T // p_t, F // p_f

    # pred_patches: (B, L, patch_dim) — only masked patches predicted
    # target: (B, L, patch_dim) — all patches
    # mask: (B, L) — 1 = masked, 0 = kept
    
    # 创建完整重构: 保留可见patch的原始值, 替换遮掩patch的预测值
    pred_full = pred_patches.clone()  # (B, L, patch_dim)
    target_flat = model.patchify(batch)  # (B, L, patch_dim)
    
    # 对未被遮掩的位置 (mask=0): 保留原始值
    mask_expanded = mask.unsqueeze(-1)  # (B, L, 1)
    reconstructed = target_flat * (1 - mask_expanded) + pred_full * mask_expanded
    
    # unpatchify → (B, 1, T, F)
    recon_csi = model.unpatchify(reconstructed, T, F)
    orig_csi = batch

# 可视化前 n_samples 个样本
n_show = min(args.n_samples, B)
fig, axes = plt.subplots(n_show, 3, figsize=(15, 3 * n_show))

for i in range(n_show):
    orig = orig_csi[i, 0].cpu().numpy()      # (500, 232)
    recon = recon_csi[i, 0].cpu().numpy()     # (500, 232)
    m = mask[i].cpu().numpy().reshape(n_t, n_f)  # (20, 29)
    
    # 插值mask到原始尺寸以便显示
    m_img = np.kron(m, np.ones((p_t, p_f)))  # (500, 232)
    
    # 遮掩后的 CSI: 遮掩处=灰色
    masked_csi = orig.copy()
    masked_csi[m_img > 0.5] = np.nan

    vmin, vmax = -3, 3  # Z-score 后的范围

    axes[i, 0].imshow(orig, aspect='auto', vmin=vmin, vmax=vmax, cmap='viridis')
    axes[i, 0].set_title(f'Sample {i}: Original')
    axes[i, 0].set_xlabel('Frequency (subcarrier)')
    axes[i, 0].set_ylabel('Time')

    axes[i, 1].imshow(masked_csi, aspect='auto', vmin=vmin, vmax=vmax, cmap='viridis')
    axes[i, 1].set_title(f'Masked ({int(args.mask_ratio*100)}% patches hidden)')
    axes[i, 1].set_xlabel('Frequency (subcarrier)')
    axes[i, 1].set_ylabel('Time')

    axes[i, 2].imshow(recon, aspect='auto', vmin=vmin, vmax=vmax, cmap='viridis')
    axes[i, 2].set_title(f'Reconstructed')
    axes[i, 2].set_xlabel('Frequency (subcarrier)')
    axes[i, 2].set_ylabel('Time')

plt.tight_layout()
save_path = os.path.join(args.save_dir, f'reconstruction_{args.size}_{os.path.basename(args.model_path).replace(".pth","")}.png')
plt.savefig(save_path, dpi=150)
print(f"Saved to {save_path}")

# 计算 NMSE
se = (orig_csi - recon_csi) ** 2
mse = se.mean(dim=(1, 2, 3))
power = (orig_csi ** 2).mean(dim=(1, 2, 3))
nmse = (mse / power).mean()
print(f"Average NMSE over {n_show} samples: {nmse:.4f}")
