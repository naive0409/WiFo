"""
WiMANS 数据集下游分类微调用
- 数据来自: /mnt/DataDrive164/wr/dataset_wifo/wimans/
- 原始形状: (2901, 3, 3, 30) = (时间, Rx, Tx, 子载波)
- 预处理: 取单 Tx (索引0) → (T, 3, 30) → 平均3 Rx → (T, 30)
          → 插值为 (1, 500, 232) 与 CSI-Bench 预训练模型输入对齐
- 9 类活动, 5 个 subject (a~e), 3 个环境
"""

import os
import csv
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

WIMANS_ROOT = "/mnt/DataDrive164/wr/dataset_wifo/wimans"
TARGET_TIME = 500
TARGET_FREQ = 232

ACTIVITIES = [
    "nothing", "walk", "rotation", "jump", "wave",
    "lie_down", "pick_up", "sit_down", "stand_up",
]
LABEL_TO_IDX = {a: i for i, a in enumerate(ACTIVITIES)}
SUBJECTS = ["a", "b", "c", "d", "e"]
ENVIRONMENTS = ["classroom", "meeting_room", "empty_room"]


def load_and_preprocess(npy_path):
    try:
        raw = np.load(npy_path).astype(np.float32)
    except Exception:
        return None
    x = raw[:, :, 0, :].mean(axis=1)
    tensor = torch.from_numpy(x).float().unsqueeze(0)
    tensor = tensor.unsqueeze(0)
    tensor = F.interpolate(tensor, size=(TARGET_TIME, TARGET_FREQ),
                           mode="bilinear", align_corners=False)
    tensor = tensor.squeeze(0)
    mu = tensor.mean(dim=2, keepdim=True)
    std = tensor.std(dim=2, keepdim=True) + 1e-8
    tensor = (tensor - mu) / std
    return tensor


def _load_all_samples():
    ann_path = os.path.join(WIMANS_ROOT, "annotation.csv")
    with open(ann_path) as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)

    samples = []
    for r in all_rows:
        label_str = r["user_1_activity"].strip()
        if label_str not in LABEL_TO_IDX:
            continue
        subj = r["user_1_location"].strip()
        if subj not in SUBJECTS:
            continue
        npy_path = os.path.join(WIMANS_ROOT, "wifi_csi", "amp", r["label"] + ".npy")
        if not os.path.exists(npy_path):
            continue
        samples.append({
            "path": npy_path,
            "label": LABEL_TO_IDX[label_str],
            "subject": subj,
            "environment": r["environment"].strip(),
        })
    return samples


class WiMANSFinetuneDataset(Dataset):
    def __init__(self, split="train", held_out=None, seed=42):
        """
        held_out: None (in-domain 随机 80/10/10)
                  或 "env_classroom"/"env_meeting_room"/"env_empty_room"
                  或 "subject_a"~"subject_e"
        """
        self.split = split
        self.held_out = held_out
        self.seed = seed
        self.num_classes = len(ACTIVITIES)

        all_samples = _load_all_samples()
        total = len(all_samples)

        if held_out is not None:
            if held_out.startswith("env_"):
                held_env = held_out.replace("env_", "")
                held_out_set = [s for s in all_samples if s["environment"] == held_env]
                train_val_set = [s for s in all_samples if s["environment"] != held_env]
            elif held_out.startswith("subject_"):
                held_subj = held_out.replace("subject_", "")
                held_out_set = [s for s in all_samples if s["subject"] == held_subj]
                train_val_set = [s for s in all_samples if s["subject"] != held_subj]
            else:
                raise ValueError(f"Unknown held_out: {held_out}")

            rng = random.Random(seed)
            rng.shuffle(train_val_set)
            n_val = max(1, int(len(train_val_set) * 0.1))
            train_set = train_val_set[n_val:]
            val_set = train_val_set[:n_val]
            print(f"  [WiMANS/held_out={held_out}] 训练: {len(train_set)}, 验证: {len(val_set)}, 测试(held-out): {len(held_out_set)}")
            split_map = {"train": train_set, "val": val_set, "test": held_out_set}
            self.data = split_map.get(split, [])
        else:
            rng = random.Random(seed)
            rng.shuffle(all_samples)
            n_total = len(all_samples)
            n_train = int(n_total * 0.8)
            n_val = int(n_total * 0.1)
            print(f"  [WiMANS] 总样本: {total}")
            split_map = {
                "train": all_samples[:n_train],
                "val": all_samples[n_train:n_train + n_val],
                "test": all_samples[n_train + n_val:],
            }
            self.data = split_map.get(split, [])

        print(f"  [WiMANS/{split}] 选中样本: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        csi = load_and_preprocess(sample["path"])
        if csi is None:
            return self.__getitem__((idx + 1) % len(self.data))
        return csi, sample["label"]


if __name__ == "__main__":
    ds = WiMANSFinetuneDataset(split="train")
    print(f"  in-domain train: {len(ds)}")
    ds = WiMANSFinetuneDataset(split="test", held_out="env_classroom")
    print(f"  cross-env classroom test: {len(ds)}")
    ds = WiMANSFinetuneDataset(split="test", held_out="subject_a")
    print(f"  cross-subject a test: {len(ds)}")
