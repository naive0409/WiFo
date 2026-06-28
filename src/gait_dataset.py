"""
GaitID 数据集下游分类微调用 (User Identification)
- 数据来自: /mnt/DataDrive164/wr/dataset_wifo/Gait_Dataset/CSI_Gait/
- 文件格式: {user}-{track}-{rep}-r{receiver}.dat
  - user: user1~user11 (11 个用户)
  - track: 1~4 (行走轨迹)
  - rep: 重复编号
  - receiver: r1~r6 (6 个 Wi-Fi 接收器)
- 日期目录 = 环境（20190627~20190718 = Room#1 教室, 20190719 = Room#2 大厅）
- 原始 CSI: int8 复数 (3 Tx × 3 Rx × 30 子载波 × I/Q)
- 预处理: 幅度平均 → 每个接收器文件独立处理 → 滑动窗口 (500 时间步, 步长 250) → 频域插值到 232
- 标准化: 全局标准化（基于训练集统计数据，保留用户间幅度差异）
"""

import os
import re
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

GAIT_ROOT = "/mnt/DataDrive164/wr/dataset_wifo/Gait_Dataset/CSI_Gait"
TARGET_TIME = 500
TARGET_FREQ = 232
WINDOW_SIZE = 500
WINDOW_STRIDE = 250

# 11 个用户, 标签 0~10
USERS = [f"user{i}" for i in range(1, 12)]
USER_TO_IDX = {u: i for i, u in enumerate(USERS)}
N_USERS = len(USERS)


def _find_csi_offset(raw_bytes):
    """找到 CSI 数据的起始偏移（剩余字节数能被 540 整除）"""
    for offset in range(min(2001, len(raw_bytes))):
        remaining = len(raw_bytes) - offset
        if remaining > 0 and remaining % 540 == 0:
            return offset
    return -1


def _load_single_receiver(filepath):
    """
    加载单个 .dat 文件（一个接收器的一次记录）
    返回: (T, 30) float32 numpy, T 为时间步数, 30 为子载波数
    """
    try:
        with open(filepath, 'rb') as f:
            raw = f.read()
    except Exception:
        return None

    offset = _find_csi_offset(raw)
    if offset < 0:
        return None

    csi_bytes = np.frombuffer(raw[offset:], dtype=np.int8)
    n_packets = len(csi_bytes) // 540
    if n_packets == 0:
        return None
    csi_bytes = csi_bytes[:n_packets * 540].reshape(-1, 3, 3, 30, 2)

    csi_complex = csi_bytes[:, :, :, :, 0].astype(np.float32) + \
                  1j * csi_bytes[:, :, :, :, 1].astype(np.float32)
    csi_amp = np.abs(csi_complex).mean(axis=(1, 2))  # (T, 30)

    return csi_amp


def _parse_filename(fname):
    """解析文件名, 返回 (user, track, rep, receiver) 或 None"""
    m = re.match(r'(user\d+)-(\d+)-(\d+)-r(\d+)\.dat$', fname)
    if not m:
        return None
    return m.group(1), m.group(2), m.group(3), int(m.group(4))


def _load_all_windows():
    """加载所有 GaitID 样本（每接收器文件 → 滑动窗口展开）"""
    all_windows = []
    for date_dir in sorted(os.listdir(GAIT_ROOT)):
        date_path = os.path.join(GAIT_ROOT, date_dir)
        if not os.path.isdir(date_path) or date_dir.endswith('_sup'):
            continue

        for user_dir in sorted(os.listdir(date_path)):
            user_path = os.path.join(date_path, user_dir)
            if not os.path.isdir(user_path) or not user_dir.startswith('user'):
                continue

            for fname in sorted(os.listdir(user_path)):
                parsed = _parse_filename(fname)
                if parsed is None:
                    continue
                user, track, rep, rx = parsed
                if user not in USER_TO_IDX:
                    continue
                fpath = os.path.join(user_path, fname)

                try:
                    with open(fpath, 'rb') as f:
                        raw = f.read()
                    off = _find_csi_offset(raw)
                    if off < 0:
                        continue
                    T = (len(raw) - off) // 540
                except Exception:
                    continue

                if T < WINDOW_SIZE:
                    continue

                n_windows = (T - WINDOW_SIZE) // WINDOW_STRIDE + 1
                for w in range(n_windows):
                    all_windows.append({
                        "user": user,
                        "label": USER_TO_IDX[user],
                        "track": int(track),
                        "rep": int(rep),
                        "date": date_dir,
                        "receiver": rx,
                        "filepath": fpath,
                        "window_start": w * WINDOW_STRIDE,
                    })

    return all_windows


class GaitIDFinetuneDataset(Dataset):
    """GaitID 下游分类数据集 (User Identification, 11 类)"""

    def __init__(self, split="train", held_out=None, seed=42):
        """
        split: "train" / "val" / "test"
        held_out:
          - None: in-domain 随机 80/10/10 划分
          - "track_4" (或 track_1~4): cross-track, 留出指定 track
          - "env_20190719": cross-env, 留出 Room#2 (Hall)
        """
        self.split = split
        self.held_out = held_out
        self.seed = seed
        self.num_classes = N_USERS

        all_windows = _load_all_windows()
        total = len(all_windows)

        if held_out is not None:
            if held_out.startswith("track_"):
                held_track = held_out.replace("track_", "")
                print(f"  [GaitID] Cross-track: held out track {held_track}")
                held_out_set = [s for s in all_windows if str(s["track"]) == held_track]
                train_val_set = [s for s in all_windows if str(s["track"]) != held_track]

            elif held_out.startswith("env_"):
                held_env = held_out.replace("env_", "")
                print(f"  [GaitID] Cross-env: held out date {held_env}")
                train_val_set = [s for s in all_windows if s["date"] != held_env]
                train_users = set(s["user"] for s in train_val_set)
                held_out_set = [s for s in all_windows if s["date"] == held_env
                                and s["user"] in train_users]
                print(f"    训练集用户: {sorted(train_users)}")
                print(f"    测试集用户 (cross-env 同用户): {sorted(set(s['user'] for s in held_out_set))}")
            else:
                raise ValueError(f"Unknown held_out: {held_out}")

            rng = random.Random(seed)
            rng.shuffle(train_val_set)
            n_val = max(1, int(len(train_val_set) * 0.1))
            train_set = train_val_set[n_val:]
            val_set = train_val_set[:n_val]
            print(f"    训练: {len(train_set)}, 验证: {len(val_set)}, 测试: {len(held_out_set)}")
            split_map = {"train": train_set, "val": val_set, "test": held_out_set}
            self.data = split_map.get(split, [])

        else:
            rng = random.Random(seed)
            rng.shuffle(all_windows)
            n_total = len(all_windows)
            n_train = int(n_total * 0.8)
            n_val = int(n_total * 0.1)
            split_map = {
                "train": all_windows[:n_train],
                "val": all_windows[n_train:n_train + n_val],
                "test": all_windows[n_train + n_val:],
            }
            self.data = split_map.get(split, [])

        print(f"  [GaitID/{split}] 选中 windows: {len(self.data)}")

        # 全局标准化参数
        self.global_mean = None
        self.global_std = None
        if split == "train":
            self._compute_global_stats()
        else:
            self.global_mean = getattr(GaitIDFinetuneDataset, '_global_mean', torch.tensor(78.0))
            self.global_std = getattr(GaitIDFinetuneDataset, '_global_std', torch.tensor(6.0))

        print(f"  [GaitID/{split}] global_mean={self.global_mean:.2f}, global_std={self.global_std:.2f}")

    def _compute_global_stats(self):
        """从训练集样本估算全局均值和标准差"""
        means, stds = [], []
        n_samples = min(100, len(self.data))
        for i in range(n_samples):
            sample = self.data[i]
            raw = _load_single_receiver(sample["filepath"])
            if raw is None:
                continue
            ws = sample["window_start"]
            seg = raw[ws:ws + WINDOW_SIZE]
            if seg.shape[0] < WINDOW_SIZE:
                continue
            means.append(seg.mean())
            stds.append(seg.std())
        if means:
            self.global_mean = torch.tensor(np.mean(means))
            self.global_std = torch.tensor(np.mean(stds))
        else:
            self.global_mean = torch.tensor(78.0)
            self.global_std = torch.tensor(6.0)
        GaitIDFinetuneDataset._global_mean = self.global_mean
        GaitIDFinetuneDataset._global_std = self.global_std
        print(f"  [GaitID] 全局标准化参数: mean={self.global_mean:.2f}, std={self.global_std:.2f}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        raw = _load_single_receiver(sample["filepath"])
        if raw is None:
            return self.__getitem__((idx + 1) % len(self.data))

        ws = sample["window_start"]
        seg = raw[ws:ws + WINDOW_SIZE]
        if seg.shape[0] < WINDOW_SIZE:
            return self.__getitem__((idx + 1) % len(self.data))

        tensor = torch.from_numpy(seg).float().unsqueeze(0).unsqueeze(0)
        tensor = F.interpolate(tensor, size=(WINDOW_SIZE, TARGET_FREQ),
                               mode="bilinear", align_corners=False)
        tensor = tensor.squeeze(0)

        if self.global_std is not None and self.global_std > 1e-8:
            tensor = (tensor - self.global_mean) / self.global_std

        return tensor, sample["label"]


if __name__ == "__main__":
    print("=== GaitID Dataset Test ===")
    ds = GaitIDFinetuneDataset(split="train")
    print(f"  in-domain train: {len(ds)}")
    for i in range(3):
        csi, label = ds[i]
        print(f"  sample {i}: csi shape={csi.shape}, label={label}")
    print("\n--- Cross-track test ---")
    ds = GaitIDFinetuneDataset(split="test", held_out="track_4")
    print(f"  cross-track test: {len(ds)}")
    print("\n--- Cross-env test ---")
    ds = GaitIDFinetuneDataset(split="test", held_out="env_20190719")
    print(f"  cross-env test: {len(ds)}")
