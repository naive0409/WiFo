"""
回算已训练 checkpoint 的 NMSE，追加到 TensorBoard 记录。

用法:
    python scripts/compute_nmse.py --run_name small_25k --size small --csibench_max 25540
    python scripts/compute_nmse.py --run_name small_63k --size small --csibench_max 63850
    python scripts/compute_nmse.py --run_name debug_small_bs256_full --size small
"""

import os, sys, glob, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from model import WiFo2D_model
from csibench_pretrain_dataset import create_pretrain_loader

parser = argparse.ArgumentParser()
parser.add_argument('--run_name', required=True)
parser.add_argument('--size', default='small')
parser.add_argument('--csibench_max', type=int, default=None)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--mask_ratio', type=float, default=0.80)
args = parser.parse_args()

RUN_DIR = f"./experiments/csibench/{args.run_name}"
TB_DIR = f"./experiments/tensorboard/{args.run_name}"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── 重建 val_loader ──
_, val_loader, _ = create_pretrain_loader(
    batch_size=args.batch_size,
    max_samples=args.csibench_max,
    num_workers=4,
    val_ratio=0.10,
    test_ratio=0.10,
)

# ── 模型 ──
model_args = type('Args', (), {
    'size': args.size,
    'patch_size_time': 25,
    'patch_size_freq': 8,
})()
model = WiFo2D_model(args=model_args).to(device)
model.eval()

# ── 找到所有 checkpoint，按 epoch 数字排序 ──
ckpt_files = glob.glob(os.path.join(RUN_DIR, "*.pth"))

def _epoch_key(p):
    f = os.path.basename(p)
    if 'ep' in f:
        return int(f.split('ep')[1].split('.')[0])
    return 9999  # best/final 排最后

ckpts = sorted(ckpt_files, key=_epoch_key)
print(f"找到 {len(ckpts)} 个 checkpoint in {RUN_DIR}")

# ── TensorBoard 追加写入 ──
os.makedirs(TB_DIR, exist_ok=True)
writer = SummaryWriter(log_dir=TB_DIR, flush_secs=5)

for ckpt_path in ckpts:
    # 从文件名解析 epoch，如 wifo2d_small_ep40.pth → 40
    fname = os.path.basename(ckpt_path)
    if 'best' in fname:
        # best 的 epoch 不确定，跳过 (或手动处理)
        print(f"  [跳过] {fname} (best checkpoint)")
        continue
    elif 'final' in fname:
        # final 也是最后一个 epoch，可以从 ep200 知道
        print(f"  [跳过] {fname} (final checkpoint)")
        continue
    elif 'ep' in fname:
        ep = int(fname.split('ep')[1].split('.')[0])
    else:
        print(f"  [跳过] {fname} (无法解析 epoch)")
        continue

    # 加载权重
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    print(f"  加载 {fname} (epoch {ep})...")

    # ── 计算 val NMSE ──
    nmse_total, n_batches = 0.0, 0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device, non_blocking=True)
            _, _, pred, target, mask = model(batch, mask_ratio=args.mask_ratio)
            diff = (pred - target).norm(dim=-1) ** 2
            tpow = target.norm(dim=-1) ** 2
            m = mask.view(diff.shape)
            nmse = (diff * m).sum() / (tpow * m).sum()
            nmse_total += nmse.item()
            n_batches += 1
    avg_nmse = nmse_total / max(n_batches, 1)

    # ── 写入 TensorBoard (epoch 从 1 开始计数，所以 ep-1) ──
    writer.add_scalar("nmse/val", avg_nmse, ep - 1)
    print(f"    epoch {ep}: NMSE = {avg_nmse:.4f}")

writer.close()
print("Done. 已追加到 TensorBoard.")
