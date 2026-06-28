# coding=utf-8
import argparse
import os
import random

import setproctitle
import torch

from model import WiFo2D_model, WiFo_model
from train import TrainLoop

try:
    from DataLoader import data_load_main
except ImportError:
    data_load_main = None  # WiFo original dataloader not needed for CSIBench
import numpy as np
import torch as th
from torch.utils.tensorboard import SummaryWriter

from csibench_pretrain_dataset import create_pretrain_loader
from utils import *


def setup_init(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.backends.cudnn.benchmark = False
    th.backends.cudnn.deterministic = True


def dev(device_id="0"):
    """
    Get the device to use for torch.distributed.
    #"""
    if th.cuda.is_available():
        return th.device("cuda:{}".format(device_id))
    return th.device("cpu")


def create_argparser():
    defaults = dict(
        # experimental settings
        note="",
        task="short",
        file_load_path="",
        dataset="DS1",
        used_data="",
        process_name="process_name",
        his_len=6,
        pred_len=6,
        few_ratio=0.5,
        stage=0,
        # model settings
        mask_ratio=0.5,
        patch_size=4,
        t_patch_size=2,
        size="middle",
        no_qkv_bias=0,
        pos_emb="SinCos",
        conv_num=3,
        # pretrain settings
        random=True,
        mask_strategy="random",
        mask_strategy_random="batch",  # ['none','batch']
        # training parameters
        lr=1e-3,
        min_lr=1e-5,
        early_stop=5,
        weight_decay=0.05,
        batch_size=256,
        log_interval=5,
        total_epoches=10000,
        device_id="1",
        machine="machine_name",
        clip_grad=0.05,  # 0.05
        lr_anneal_steps=200,
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    # 新增: 2D model 参数 (CSIBench)
    parser.add_argument(
        "--run_name",
        type=str,
        default="",
        help="Unique run name for saving weights (default: auto timestamp)",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="wifo",
        choices=["wifo", "wifo2d"],
        help="wifo (original 3D) or wifo2d (2D for CSI-Bench)",
    )
    parser.add_argument(
        "--patch_size_time", type=int, default=25, help="Time patch size for WiFo2D"
    )
    parser.add_argument(
        "--patch_size_freq", type=int, default=8, help="Frequency patch size for WiFo2D"
    )
    # CSIBench 预训练参数
    parser.add_argument(
        "--csibench", action="store_true", help="Use CSI-Bench pretrain dataset"
    )
    parser.add_argument(
        "--csibench_max",
        type=int,
        default=None,
        help="Max CSIBench samples (for testing)",
    )
    return parser


torch.multiprocessing.set_sharing_strategy("file_system")


def main():

    th.autograd.set_detect_anomaly(True)

    args = create_argparser().parse_args()
    setproctitle.setproctitle("{}-{}".format(args.process_name, args.device_id))
    setup_init(100)  # 随机种子设定100

    # CUDA_VISIBLE_DEVICES 会重映射 GPU 编号，所以始终用 cuda:0
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.csibench:
        # ── CSI-Bench 预训练模式 ──
        n_epochs = min(args.total_epoches, 2000)
        print(f"Loading CSI-Bench data (max={args.csibench_max}), epochs={n_epochs}...")

        # 创建目录
        save_dir = "./experiments/csibench/"
        os.makedirs(save_dir, exist_ok=True)
        if args.run_name:
            run_id = args.run_name
        else:
            from datetime import datetime

            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(save_dir, run_id)
        os.makedirs(run_dir, exist_ok=True)
        print(f"Run directory: {run_dir}/")

        # ── 训练一开始就写入 args 到 summary.json ──
        import json
        summary = {
            "args": {
                "model_type": "wifo2d",
                "size": args.size,
                "batch_size": args.batch_size,
                "total_epoches": n_epochs,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "mask_ratio": args.mask_ratio,
                "patch_size_time": args.patch_size_time,
                "patch_size_freq": args.patch_size_freq,
                "csibench_max": args.csibench_max,
                "run_name": run_id,
            },
            "status": "running",
        }
        with open(os.path.join(run_dir, "summary.json"), 'w') as f:
            json.dump(summary, f, indent=2)

        # TensorBoard — 集中存放 experiments/tensorboard/<run_id>/，方便对比
        tb_dir = os.path.join("./experiments/tensorboard", run_id)
        os.makedirs(tb_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=tb_dir, flush_secs=5)

        # DataLoader (80% 训练, 10% 验证, 10% 测试 — 随机分割)
        train_loader, val_loader, test_loader = create_pretrain_loader(
            batch_size=args.batch_size,
            max_samples=args.csibench_max,
            num_workers=4,
            val_ratio=0.10,
            test_ratio=0.10,
        )

        model = WiFo2D_model(args=args).to(device)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"WiFo2D parameters: {total_params}")

        optimizer = th.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        scheduler = th.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
        best_val_loss = float("inf")

        import time

        for epoch in range(n_epochs):
            # ── Train ──
            model.train()
            total_loss, n_batches = 0, 0
            t0 = time.time()
            for batch in train_loader:
                batch = batch.to(device, non_blocking=True)
                loss, _, _, _, _ = model(batch, mask_ratio=args.mask_ratio)
                optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1
            scheduler.step()
            avg_train_loss = total_loss / max(n_batches, 1)
            elapsed = time.time() - t0
            lr_now = scheduler.get_last_lr()[0]

            # ── Validation ──
            model.eval()
            val_loss_total, val_nmse_total, val_batches = 0, 0, 0
            with th.no_grad():
                for batch in val_loader:
                    batch = batch.to(device, non_blocking=True)
                    loss, _, pred, target, mask = model(batch, mask_ratio=args.mask_ratio)
                    val_loss_total += loss.item()
                    # NMSE on masked patches: mean|pred-target|² / mean|target|²
                    diff = (pred - target).norm(dim=-1) ** 2   # (B, L)
                    tpow = target.norm(dim=-1) ** 2            # (B, L)
                    m = mask.view(diff.shape)
                    nmse = (diff * m).sum() / (tpow * m).sum()
                    val_nmse_total += nmse.item()
                    val_batches += 1
            avg_val_loss = val_loss_total / max(val_batches, 1)
            avg_val_nmse = val_nmse_total / max(val_batches, 1)

            # ── Log ──
            print(
                f"Epoch {epoch + 1:3d}/{n_epochs} | "
                f"train={avg_train_loss:.6f} | val={avg_val_loss:.6f} | "
                f"NMSE={avg_val_nmse:.4f} | "
                f"lr={lr_now:.2e} | {elapsed:.0f}s"
            )

            writer.add_scalar("loss/train", avg_train_loss, epoch)
            writer.add_scalar("loss/val", avg_val_loss, epoch)
            writer.add_scalar("nmse/val", avg_val_nmse, epoch)
            writer.add_scalar("lr", lr_now, epoch)

            # ── 保存 best (按 val loss) ──
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                th.save(
                    model.state_dict(),
                    os.path.join(run_dir, f"wifo2d_{args.size}_best.pth"),
                )
            if (epoch + 1) % 20 == 0:
                th.save(
                    model.state_dict(),
                    os.path.join(run_dir, f"wifo2d_{args.size}_ep{epoch + 1}.pth"),
                )

        th.save(
            model.state_dict(), os.path.join(run_dir, f"wifo2d_{args.size}_final.pth")
        )

        # ── 测试集评估 ──
        model.eval()
        test_loss_total, test_nmse_total, test_batches = 0, 0, 0
        with th.no_grad():
            for batch in test_loader:
                batch = batch.to(device, non_blocking=True)
                loss, _, pred, target, mask = model(batch, mask_ratio=args.mask_ratio)
                test_loss_total += loss.item()
                diff = (pred - target).norm(dim=-1) ** 2
                tpow = target.norm(dim=-1) ** 2
                m = mask.view(diff.shape)
                nmse = (diff * m).sum() / (tpow * m).sum()
                test_nmse_total += nmse.item()
                test_batches += 1
        avg_test_loss = test_loss_total / max(test_batches, 1)
        avg_test_nmse = test_nmse_total / max(test_batches, 1)
        print(f"Test  loss={avg_test_loss:.6f}, NMSE={avg_test_nmse:.4f}")

        writer.close()

        # ── 更新 summary.json ──
        summary_path = os.path.join(run_dir, "summary.json")
        with open(summary_path) as f:
            summary = json.load(f)
        summary["results"] = {
            "best_val_loss": round(best_val_loss, 6),
            "test_mse": round(avg_test_loss, 6),
            "test_nmse": round(avg_test_nmse, 4),
        }
        summary["status"] = "completed"
        tmp = summary_path + ".tmp"
        with open(tmp, 'w') as f:
            json.dump(summary, f, indent=2)
        os.replace(tmp, summary_path)
        print(f"Done. Best val loss={best_val_loss:.6f}, test loss={avg_test_loss:.6f}")

    else:
        # ── 原始 WiFo 模式 ──
        if data_load_main is None:
            raise ImportError("DataLoader not available (hdf5storage missing)")
        test_data = data_load_main(args)

        args.folder = "Dataset_{}_Task_{}_FewRatio_{}_{}_{}/".format(
            args.dataset, args.task, args.few_ratio, args.size, args.note
        )
        args.folder = "Test_" + args.folder

        if args.mask_strategy_random != "batch":
            args.folder = (
                "{}_{}".format(args.mask_strategy, args.mask_ratio) + args.folder
            )
        args.model_path = "./experiments/{}".format(args.folder)
        logdir = "./logs/{}".format(args.folder)
        if not os.path.exists(args.model_path):
            os.makedirs(args.model_path)
            os.makedirs(args.model_path + "model_save/")

        writer = SummaryWriter(log_dir=logdir, flush_secs=5)

        model = WiFo_model(args=args).to(device)

        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total number of parameters: {total_params}")
        if args.file_load_path != "":
            model.load_state_dict(
                torch.load("{}.pkl".format(args.file_load_path), map_location=device),
                strict=False,
            )
            print("pretrained model loaded" + args.file_load_path)
        TrainLoop(
            args=args,
            writer=writer,
            model=model,
            test_data=test_data,
            device=device,
            early_stop=args.early_stop,
        ).run_loop()


if __name__ == "__main__":
    main()
