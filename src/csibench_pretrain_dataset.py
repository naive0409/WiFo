"""
CSI-Bench 无监督预训练数据集
- 遍历所有 H5 文件，加载 CSI_amps
- 插值统一到 (500, 232) — 遵循 "Scale What Counts" 预处理哲学
- 输出 (1, 500, 232) 幅度张量，用于 MAE 自监督预训练
"""

import os
import json
import random
import h5py
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader


CSI_BENCH_ROOT = "/mnt/DataDrive164/wr/dataset_wifo/csibench"
CACHE_PATH = "./experiments/csibench/h5_filelist.json"
TARGET_TIME = 500
TARGET_FREQ = 232


def load_file_list(max_files=None, seed=42):
    """从缓存加载 H5 文件列表。

    当 max_files 指定时，从全集中随机抽样（而非取前 N 个），
    保证各数据子集能覆盖所有场景/设备/用户。
    """
    if not os.path.exists(CACHE_PATH):
        raise FileNotFoundError(
            f"缓存文件不存在: {CACHE_PATH}\n"
            f"请先运行: python scripts/build_h5_cache.py"
        )
    with open(CACHE_PATH) as f:
        files = json.load(f)
    if max_files and max_files < len(files):
        rng = random.Random(seed)
        files = rng.sample(files, max_files)
    return files


def load_csi(filepath, t_len=TARGET_TIME, f_len=TARGET_FREQ):
    """
    读取 H5 → 插值到 (1, t_len, f_len) → Z-score

    原始 CSI_amps 形状: (F, T, 1)
    流程: (F,T,1) → squeeze → (1,T,F) → interpolate → (1,t_len,f_len) → Z-score
    """
    try:
        with h5py.File(filepath, "r") as f:
            raw = f["CSI_amps"][:]          # (F, T, 1)
    except Exception as e:
        print(f"  [跳过] {filepath}: {e}")
        return None

    if raw.size == 0:
        return None

    raw = np.squeeze(raw, axis=-1)           # (F, T)
    tensor = torch.from_numpy(raw).float()   # (F, T)

    # permute → (1, T, F)  (1 = channel dim)
    tensor = tensor.unsqueeze(0)             # (1, F, T)
    tensor = tensor.permute(0, 2, 1)         # (1, T, F)

    orig_t, orig_f = tensor.shape[1], tensor.shape[2]

    # 时间轴插值
    if orig_t != t_len:
        tensor = tensor.unsqueeze(0)         # (1, 1, T, F)
        tensor = F.interpolate(tensor, size=(t_len, orig_f), mode="bilinear", align_corners=False)
        tensor = tensor.squeeze(0)           # (1, t_len, orig_f)

    # 频率轴插值
    if orig_f != f_len:
        tensor = tensor.unsqueeze(0)         # (1, 1, t_len, orig_f)
        tensor = F.interpolate(tensor, size=(t_len, f_len), mode="bilinear", align_corners=False)
        tensor = tensor.squeeze(0)           # (1, t_len, f_len)

    # Z-score (per-sample)
    mu, std_ = tensor.mean(), tensor.std()
    if std_ > 1e-8:
        tensor = (tensor - mu) / std_
    else:
        tensor = tensor - mu

    return tensor  # (1, 500, 232)


class CSIBenchPretrainDataset(Dataset):
    """CSI-Bench 无监督预训练数据集"""

    def __init__(self, max_samples=None, file_list=None):
        if file_list is not None:
            self.files = file_list
        else:
            print("加载 H5 文件列表...")
            self.files = load_file_list(max_samples)
        print(f"  共 {len(self.files)} 个样本")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        tensor = load_csi(self.files[idx])
        if tensor is None:
            # 出错时取下一个
            return self.__getitem__((idx + 1) % len(self.files))
        return tensor


def create_pretrain_loader(batch_size=128, max_samples=None, num_workers=4,
                           val_ratio=0.0, test_ratio=0.0, shuffle=True, seed=42):
    """
    创建 train / val / test DataLoader。

    Args:
        val_ratio: 验证集比例 (0.0 = 不创建)
        test_ratio: 测试集比例 (0.0 = 不创建)
        seed: 随机打乱种子 (仅用于划分子集)
    Returns:
        根据参数返回 1~3 个 loader:
        val_ratio=0, test_0        → train_loader
        val_ratio>0, test_ratio=0  → (train_loader, val_loader)
        val_ratio>0, test_ratio>0  → (train_loader, val_loader, test_loader)
    """
    all_files = load_file_list(max_samples, seed=seed)
    n_total = len(all_files)

    # 随机打乱后按比例切分
    rng = random.Random(seed)
    shuffled = all_files[:]
    rng.shuffle(shuffled)

    n_test = max(1, int(n_total * test_ratio)) if test_ratio > 0 else 0
    n_val = max(1, int(n_total * val_ratio)) if val_ratio > 0 else 0
    n_train = n_total - n_val - n_test

    train_files = shuffled[:n_train]
    val_files = shuffled[n_train:n_train + n_val] if n_val > 0 else []
    test_files = shuffled[n_train + n_val:] if n_test > 0 else []

    print(f"  分割: train={n_train} ({100*n_train/n_total:.0f}%), "
          f"val={n_val} ({100*n_val/n_total:.0f}%), "
          f"test={n_test} ({100*n_test/n_total:.0f}%)")

    train_ds = CSIBenchPretrainDataset(file_list=train_files)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle,
                              num_workers=num_workers, pin_memory=True, drop_last=True)

    result = [train_loader]

    if n_val > 0:
        val_ds = CSIBenchPretrainDataset(file_list=val_files)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=True, drop_last=False)
        result.append(val_loader)

    if n_test > 0:
        test_ds = CSIBenchPretrainDataset(file_list=test_files)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                                 num_workers=num_workers, pin_memory=True, drop_last=False)
        result.append(test_loader)

    return result[0] if len(result) == 1 else tuple(result)


if __name__ == "__main__":
    loader = create_pretrain_loader(batch_size=4, max_samples=50, num_workers=0)
    for x in loader:
        print(f"Batch shape: {x.shape}")   # (4, 1, 500, 232)
        print(f"  mean={x.mean():.4f}, std={x.std():.4f}")
        break
