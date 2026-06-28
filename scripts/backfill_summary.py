"""
从日志文件恢复 summary.json 到已完成的预训练实验目录。
"""
import re, os, json, sys

RUNS = {
    "debug_small_bs256_full": {
        "args": {
            "model_type": "wifo2d", "size": "small",
            "batch_size": 256, "total_epoches": 300,
            "lr": 0.001, "weight_decay": 0.05,
            "mask_ratio": 0.8, "patch_size_time": 25, "patch_size_freq": 8,
            "csibench_max": None, "run_name": "debug_small_bs256_full",
        },
        "log": "train_log/debug_small_bs256_full.log",
    },
    "small_25k": {
        "args": {
            "model_type": "wifo2d", "size": "small",
            "batch_size": 256, "total_epoches": 200,
            "lr": 0.001, "weight_decay": 0.05,
            "mask_ratio": 0.8, "patch_size_time": 25, "patch_size_freq": 8,
            "csibench_max": 25540, "run_name": "small_25k",
        },
        "log": "train_log/small_25k.log",
    },
    "tiny_full": {
        "args": {
            "model_type": "wifo2d", "size": "tiny",
            "batch_size": 256, "total_epoches": 200,
            "lr": 0.001, "weight_decay": 0.05,
            "mask_ratio": 0.8, "patch_size_time": 25, "patch_size_freq": 8,
            "csibench_max": None, "run_name": "tiny_full",
        },
        "log": "train_log/tiny_full.log",
    },
    "small_63k": {
        "args": {
            "model_type": "wifo2d", "size": "small",
            "batch_size": 256, "total_epoches": 200,
            "lr": 0.001, "weight_decay": 0.05,
            "mask_ratio": 0.8, "patch_size_time": 25, "patch_size_freq": 8,
            "csibench_max": 63850, "run_name": "small_63k",
        },
        "log": "train_log/small_63k.log",
    },
}

for name, cfg in RUNS.items():
    log_path = cfg["log"]
    run_dir = f"experiments/csibench/{name}"
    summary_path = f"{run_dir}/summary.json"

    if not os.path.exists(run_dir):
        print(f"[跳过] {run_dir} 不存在")
        continue

    # 解析日志：找 best val loss, test loss, test nmse, 最后 epoch
    best_val = None
    test_mse = None
    test_nmse = None
    last_epoch = None

    with open(log_path) as f:
        for line in f:
            # Best val loss
            m = re.search(r"Best val loss=([\d.]+)", line)
            if m:
                best_val = float(m.group(1))
            # Test loss 和 NMSE (可能在同一行)
            m = re.search(r"Test\s+loss=([\d.]+)", line)
            if m:
                test_mse = float(m.group(1))
            m = re.search(r"NMSE=([\d.]+)", line)
            if m:
                test_nmse = float(m.group(1))
            # Last epoch completed
            m = re.search(r"Epoch\s+(\d+)/\d+\s+\|", line)
            if m:
                last_epoch = int(m.group(1))

    # 确定状态
    if test_mse is not None:
        status = "completed"
    elif last_epoch is not None:
        status = f"interrupted_at_epoch_{last_epoch}"
    else:
        status = "unknown"

    summary = {
        "args": cfg["args"],
        "status": status,
    }
    if best_val is not None:
        summary.setdefault("results", {})["best_val_loss"] = round(best_val, 6)
    if test_mse is not None:
        summary.setdefault("results", {})["test_mse"] = round(test_mse, 6)
    if test_nmse is not None:
        summary.setdefault("results", {})["test_nmse"] = round(test_nmse, 4)

    # 原子写入
    tmp = summary_path + ".tmp"
    with open(tmp, 'w') as f:
        json.dump(summary, f, indent=2)
    os.replace(tmp, summary_path)
    print(f"[{name}] status={status}, best_val={best_val}, test_mse={test_mse}")
