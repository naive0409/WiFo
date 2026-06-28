"""
XRF55 数据集下游分类微调用
- 数据来自: /mnt/DataDrive164/wr/dataset_wifo/xrf55/
- 原始形状: (270, 1000) = (子载波/天线, 时间步)
- 预处理: 转置 → 插值为 (1, 500, 232) 与预训练模型对齐
- 55 类活动 (01~55), 4 个场景 (Scene1~Scene4)
- 15 个 subject (由文件名第一段标识)
"""

import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

XRF55_ROOT = "/mnt/DataDrive164/wr/dataset_wifo/xrf55"
TARGET_TIME = 500
TARGET_FREQ = 232

SCENES = ["Scene1", "Scene2", "Scene3", "Scene4"]


def load_and_preprocess(npy_path):
    """加载 XRF55 .npy 并预处理为 (1, 500, 232)"""
    try:
        raw = np.load(npy_path).astype(np.float32)  # (270, 1000)
    except Exception:
        return None
    tensor = torch.from_numpy(raw).float().unsqueeze(0)
    tensor = tensor.permute(0, 2, 1)                     # (1, 1000, 270)
    tensor = tensor.unsqueeze(0)                         # (1, 1, 1000, 270)
    tensor = F.interpolate(
        tensor, size=(TARGET_TIME, TARGET_FREQ),
        mode="bilinear", align_corners=False,
    )
    tensor = tensor.squeeze(0)
    mu = tensor.mean(dim=2, keepdim=True)
    std = tensor.std(dim=2, keepdim=True) + 1e-8
    tensor = (tensor - mu) / std
    return tensor


def _load_all_samples():
    """加载 XRF55 所有 WiFi 样本"""
    samples = []
    for scene in SCENES:
        wifi_dir = os.path.join(XRF55_ROOT, scene, scene, "WiFi")
        if not os.path.isdir(wifi_dir):
            continue
        for fname in sorted(os.listdir(wifi_dir)):
            if not fname.endswith(".npy"):
                continue
            parts = fname.replace(".npy", "").split("_")
            if len(parts) < 3:
                continue
            subject, activity = parts[0], parts[1]
            label = int(activity) - 1
            fpath = os.path.join(wifi_dir, fname)
            if not os.path.exists(fpath):
                continue
            samples.append({
                "path": fpath,
                "label": label,
                "activity_num": int(activity),
                "subject": subject,
                "scene": scene,
            })
    return samples


def _stratified_split(samples, train_ratio=0.8, val_ratio=0.1, seed=42):
    """按类别分层抽样"""
    from collections import defaultdict
    by_label = defaultdict(list)
    for s in samples:
        by_label[s["label"]].append(s)
    rng = random.Random(seed)
    train, val, test = [], [], []
    for _, group in by_label.items():
        rng.shuffle(group)
        n = len(group)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train.extend(group[:n_train])
        val.extend(group[n_train:n_train + n_val])
        test.extend(group[n_train + n_val:])
    rng.shuffle(train)
    return train, val, test


class XRF55FinetuneDataset(Dataset):
    """XRF55 下游分类数据集, 55 类活动"""

    def __init__(self, split="train", held_out=None, seed=42):
        """
        split: "train" / "val" / "test"
        held_out: None (in-domain 随机划分)
                  或 "env_Scene1"~"env_Scene4"（留一场景）
                  或 "subject_01"~"subject_31"（留一 subject）
        """
        self.split = split
        self.held_out = held_out
        self.seed = seed
        self.num_classes = 55

        all_samples = _load_all_samples()
        total = len(all_samples)

        if held_out is not None:
            # cross-domain: 训练集排除 held-out domain
            if held_out.startswith("env_"):
                held_env = held_out.replace("env_", "")
                held_out_samples = [s for s in all_samples if s["scene"] == held_env]
                train_val_samples = [s for s in all_samples if s["scene"] != held_env]
            elif held_out.startswith("subject_"):
                held_subj = held_out.replace("subject_", "")
                held_out_samples = [s for s in all_samples if s["subject"] == held_subj]
                train_val_samples = [s for s in all_samples if s["subject"] != held_subj]
            else:
                raise ValueError(f"Unknown held_out: {held_out}")

            # 从 train_val 中按 90/10 分出 train/val
            rng = random.Random(seed)
            rng.shuffle(train_val_samples)
            n_val = max(1, int(len(train_val_samples) * 0.1))
            train_set = train_val_samples[n_val:]
            val_set = train_val_samples[:n_val]
            test_set = held_out_samples
            print(f"  [XRF55/held_out={held_out}] 总样本: {total}")
            print(f"    训练: {len(train_set)}, 验证: {len(val_set)}, 测试(held-out): {len(test_set)}")

            split_map = {"train": train_set, "val": val_set, "test": test_set}
            self.data = split_map.get(split, [])

        else:
            # in-domain: 随机 80/10/10
            print(f"  [XRF55] 总样本: {total}")
            train, val, test = _stratified_split(all_samples, seed=seed)
            split_map = {"train": train, "val": val, "test": test}
            self.data = split_map.get(split, [])

        print(f"  [XRF55/{split}] 选中样本: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        csi = load_and_preprocess(sample["path"])
        if csi is None:
            return self.__getitem__((idx + 1) % len(self.data))
        return csi, sample["label"]


if __name__ == "__main__":
    # 测试 in-domain
    ds = XRF55FinetuneDataset(split="train")
    print(f"  in-domain train: {len(ds)}")
    # 测试 cross-env
    ds = XRF55FinetuneDataset(split="test", held_out="env_Scene1")
    print(f"  cross-env Scene1 test: {len(ds)}")
    # 测试 cross-subject
    ds = XRF55FinetuneDataset(split="test", held_out="subject_01")
    print(f"  cross-subject 01 test: {len(ds)}")
