"""
从训练日志恢复 TensorBoard 事件 (loss/val, lr)。

用法:
    python scripts/restore_tb_from_log.py --log train_log/small_25k.log --tb_dir experiments/tensorboard/small_25k
"""

import re, os, sys, argparse
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--log', required=True)
parser.add_argument('--tb_dir', required=True)
args = parser.parse_args()

# 解析日志
pattern = re.compile(
    r"Epoch\s+(\d+)/\d+\s+\|\s+train=([\d.]+)\s+\|\s+val=([\d.]+)\s+\|\s+lr=([\d.e+-]+)"
)

epochs, train_losses, val_losses, lrs = [], [], [], []

with open(args.log) as f:
    for line in f:
        m = pattern.search(line)
        if m:
            epochs.append(int(m.group(1)))
            train_losses.append(float(m.group(2)))
            val_losses.append(float(m.group(3)))
            lrs.append(float(m.group(4)))

print(f"解析到 {len(epochs)} 个 epoch 记录")

# 写入 TensorBoard
os.makedirs(args.tb_dir, exist_ok=True)
writer = SummaryWriter(log_dir=args.tb_dir, flush_secs=5)

for ep, tr, va, lr in zip(epochs, train_losses, val_losses, lrs):
    step = ep - 1  # 0-indexed
    writer.add_scalar("loss/train", tr, step)
    writer.add_scalar("loss/val", va, step)
    writer.add_scalar("lr", lr, step)

writer.close()
print(f"已写入 {args.tb_dir}")
