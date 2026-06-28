"""
CSI-Bench 下游分类微调数据集
- 加载 Multitask 任务的 metadata + splits
- 标签映射 + 跨域 split 支持
- 与预训练保持一致的预处理流水线
"""

import os
import json
import pandas as pd
import h5py
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader

CSI_BENCH_ROOT = "/mnt/DataDrive164/wr/dataset_wifo/csibench"
TARGET_TIME = 500
TARGET_FREQ = 232

# Multitask 子任务
MULTITASK_TASKS = {
    "HumanActivityRecognition": os.path.join(CSI_BENCH_ROOT, "Multitask", "HumanActivityRecognition"),
    "HumanIdentification":      os.path.join(CSI_BENCH_ROOT, "Multitask", "HumanIdentification"),
    "ProximityRecognition":     os.path.join(CSI_BENCH_ROOT, "Multitask", "ProximityRecognition"),
}

# 可用 split 类型
SPLIT_TYPES = {
    "id":             "test_id.json",
    "cross_user":     "test_cross_user.json",
    "cross_env":      "test_cross_env.json",
    "cross_device":   "test_cross_device.json",
}


def resolve_h5_path(meta_filepath):
    """解析 metadata 中的相对路径为绝对路径"""
    # metadata 中路径形如 ../../sub_Human_h5/...
    # 相对于 Multitask/HumanActivityRecognition/metadata/
    parts = meta_filepath.replace("\\", "/").lstrip("/")
    # 去掉 "../" 前缀
    while parts.startswith("../"):
        parts = parts[3:]
    return os.path.join(CSI_BENCH_ROOT, "Multitask", parts)


def load_csi_from_path(filepath, t_len=TARGET_TIME, f_len=TARGET_FREQ):
    """加载单个 CSI 样本，插值到 (1, t_len, f_len)"""
    try:
        with h5py.File(filepath, "r") as f:
            raw = f["CSI_amps"][:]  # (F, T, 1)
    except Exception:
        return None

    raw = np.squeeze(raw, axis=-1)                     # (F, T)
    tensor = torch.from_numpy(raw).float()
    tensor = tensor.unsqueeze(0).permute(0, 2, 1)      # (1, T, F)

    ot, of = tensor.shape[1], tensor.shape[2]
    if ot != t_len:
        tensor = tensor.unsqueeze(0)
        tensor = F.interpolate(tensor, size=(t_len, of), mode="bilinear", align_corners=False)
        tensor = tensor.squeeze(0)
    if of != f_len:
        tensor = tensor.unsqueeze(0)
        tensor = F.interpolate(tensor, size=(t_len, f_len), mode="bilinear", align_corners=False)
        tensor = tensor.squeeze(0)

    mu, s = tensor.mean(), tensor.std()
    if s > 1e-8:
        tensor = (tensor - mu) / s
    else:
        tensor = tensor - mu
    return tensor


class CSIBenchFinetuneDataset(Dataset):
    """CSI-Bench 下游分类数据集"""

    def __init__(self, task_name, split="train_id"):
        """
        task_name: "HumanActivityRecognition" / "HumanIdentification" / "ProximityRecognition"
        split: "train_id" / "val_id" / "test_id" / "test_cross_user" / "test_cross_env" / "test_cross_device"
        """
        assert task_name in MULTITASK_TASKS, f"Unknown task: {task_name}"
        self.task_dir = MULTITASK_TASKS[task_name]
        self.task_name = task_name

        # 加载 metadata
        meta_path = os.path.join(self.task_dir, "metadata", "sample_metadata.csv")
        self.meta = pd.read_csv(meta_path)
        print(f"  [{task_name}] metadata 总样本: {len(self.meta)}")

        # 加载 split
        split_path = os.path.join(self.task_dir, "splits", f"{split}.json")
        if not os.path.exists(split_path):
            raise FileNotFoundError(f"Split not found: {split_path}")
        with open(split_path) as f:
            split_ids = set(json.load(f))

        # 过滤
        id_col = "id" if "id" in self.meta.columns else "sample_id"
        self.data = self.meta[self.meta[id_col].isin(split_ids)].reset_index(drop=True)
        print(f"  [{task_name}/{split}] 选中样本: {len(self.data)}")

        # 标签映射
        mapping_path = os.path.join(self.task_dir, "metadata", "label_mapping.json")
        if os.path.exists(mapping_path):
            with open(mapping_path) as f:
                mapping = json.load(f)
            self.label_to_idx = mapping["label_to_idx"]
        else:
            # 动态创建
            labels = sorted(self.meta["label"].unique())
            self.label_to_idx = {l: i for i, l in enumerate(labels)}

        self.num_classes = len(self.label_to_idx)
        print(f"  [{task_name}] 类别数: {self.num_classes}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        # 加载 CSI
        h5_path = resolve_h5_path(row["file_path"])
        csi = load_csi_from_path(h5_path)
        if csi is None:
            return self.__getitem__((idx + 1) % len(self.data))

        # 标签
        label_str = str(row["label"])
        label = self.label_to_idx.get(label_str, 0)

        return csi, label


def create_finetune_loaders(task_name, batch_size=32, num_workers=4):
    """创建 train / val / test 三个 DataLoader"""
    loaders = {}
    for split_name in ["train_id", "val_id"]:
        ds = CSIBenchFinetuneDataset(task_name, split=split_name)
        shuffle = (split_name == "train_id")
        loaders[split_name] = DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=True,
        )
    return loaders


if __name__ == "__main__":
    # 测试
    ds = CSIBenchFinetuneDataset("HumanActivityRecognition", split="train_id")
    for i in range(3):
        csi, label = ds[i]
        print(f"  sample {i}: csi shape={csi.shape}, label={label}")
    print(f"  总样本: {len(ds)}, 类别数: {ds.num_classes}")
