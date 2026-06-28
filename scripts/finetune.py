"""
CSI-Bench Multitask 下游分类微调
"""
import os, sys, argparse, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# 记录原始命令行
raw_command = ' '.join(sys.argv)

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

from Embed import get_2d_sincos_pos_embed
from model import WiFo2D_model, WiFo2D
from csibench_finetune_dataset import CSIBenchFinetuneDataset
from wimans_dataset import WiMANSFinetuneDataset
from xrf55_dataset import XRF55FinetuneDataset
from gait_dataset import GaitIDFinetuneDataset
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='csibench', choices=['csibench', 'wimans', 'xrf55', 'gait'])
parser.add_argument('--pretrained', type=str, default=None)
parser.add_argument('--task', type=str, default='HumanActivityRecognition')
parser.add_argument('--mode', type=str, default='ft', choices=['ft', 'lp', 'supervised'])
parser.add_argument('--size', type=str, default='small')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--held_out', type=str, default=None,
                    help='Cross-domain: held-out domain key (e.g. env_Scene1, subject_01 for xrf55)')
parser.add_argument('--run_name', type=str, default='')
args = parser.parse_args()
parsed_args = vars(args)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
run_id = args.run_name or f"{args.task}_{args.mode}_{args.size}"
save_dir = f"./experiments/finetune/{run_id}"
os.makedirs(save_dir, exist_ok=True)

# ── 日志记录：同时输出到控制台和 .log 文件 ──
class Tee:
    """将 stdout 同时写入文件和终端"""
    def __init__(self, log_path):
        self.file = open(log_path, 'w', encoding='utf-8')
        self.stdout = sys.stdout
        sys.stdout = self
    def write(self, text):
        self.stdout.write(text)
        self.file.write(text)
        self.file.flush()
    def flush(self):
        self.stdout.flush()
        self.file.flush()
    def close(self):
        sys.stdout = self.stdout
        self.file.close()

log_path = os.path.join(save_dir, f"{run_id}.log")
_tee = Tee(log_path)

print(f"[Log] 日志文件: {log_path}")
print(f"[Config] 命令行: {raw_command}")
print(f"[Config] 解析参数: {json.dumps(parsed_args, indent=2)}")

# TensorBoard — 集中存放 experiments/tensorboard/<run_id>/
tb_dir = os.path.join("./experiments/tensorboard", run_id)
os.makedirs(tb_dir, exist_ok=True)
writer = SummaryWriter(log_dir=tb_dir, flush_secs=5)
print(f"[TB] TensorBoard log: {tb_dir}")

# ── 数据 ──
print("Loading data...")

if args.dataset == 'csibench':
    train_ds = CSIBenchFinetuneDataset(args.task, split="train_id")
    val_ds   = CSIBenchFinetuneDataset(args.task, split="val_id")
    num_classes = train_ds.num_classes

    test_splits = ["test_id", "test_cross_user", "test_cross_env", "test_cross_device"]
    test_dss = {s: CSIBenchFinetuneDataset(args.task, split=s) for s in test_splits}

elif args.dataset == 'wimans':
    if args.held_out:
        train_ds = WiMANSFinetuneDataset(split="train", held_out=args.held_out)
        val_ds   = WiMANSFinetuneDataset(split="val",   held_out=args.held_out)
        test_dss = {"test_held_out": WiMANSFinetuneDataset(split="test", held_out=args.held_out)}
        test_splits = ["test_held_out"]
    else:
        train_ds = WiMANSFinetuneDataset(split="train")
        val_ds   = WiMANSFinetuneDataset(split="val")
        test_dss = {"test_id": WiMANSFinetuneDataset(split="test")}
        test_splits = ["test_id"]
    num_classes = train_ds.num_classes
    print(f"  [WiMANS] 测试集: {test_splits}")

elif args.dataset == 'xrf55':
    if args.held_out:
        train_ds = XRF55FinetuneDataset(split="train", held_out=args.held_out)
        val_ds   = XRF55FinetuneDataset(split="val",   held_out=args.held_out)
        test_dss = {"test_held_out": XRF55FinetuneDataset(split="test", held_out=args.held_out)}
        test_splits = ["test_held_out"]
    else:
        train_ds = XRF55FinetuneDataset(split="train")
        val_ds   = XRF55FinetuneDataset(split="val")
        test_dss = {"test_id": XRF55FinetuneDataset(split="test")}
        test_splits = ["test_id"]
    num_classes = train_ds.num_classes
    print(f"  [XRF55] 测试集: {test_splits}")

elif args.dataset == 'gait':
    if args.held_out:
        train_ds = GaitIDFinetuneDataset(split="train", held_out=args.held_out)
        val_ds   = GaitIDFinetuneDataset(split="val",   held_out=args.held_out)
        test_dss = {"test_held_out": GaitIDFinetuneDataset(split="test", held_out=args.held_out)}
        test_splits = ["test_held_out"]
    else:
        train_ds = GaitIDFinetuneDataset(split="train")
        val_ds   = GaitIDFinetuneDataset(split="val")
        test_dss = {"test_id": GaitIDFinetuneDataset(split="test")}
        test_splits = ["test_id"]
    num_classes = train_ds.num_classes
    print(f"  [GaitID] 测试集: {test_splits}")

train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=False)
val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=False)

test_loaders = {s: DataLoader(d, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=False)
                for s, d in test_dss.items()}

# ── 分类器构建 ──
if args.mode == 'supervised':
    model_args = type('Args', (), {
        'size': args.size,
        'patch_size_time': 25,
        'patch_size_freq': 8,
    })()
    encoder = WiFo2D_model(args=model_args)
    state_dict = None
else:
    model_args = type('Args', (), {
        'size': args.size,
        'patch_size_time': 25,
        'patch_size_freq': 8,
    })()
    encoder = WiFo2D_model(args=model_args)
    state_dict = torch.load(args.pretrained, map_location=device)

class ClassifierHead(nn.Module):
    def __init__(self, encoder, num_classes):
        super().__init__()
        self.encoder = encoder  # keep for patchify
        self.patch_embed = encoder.patch_embed
        self.blocks = encoder.blocks
        self.norm = encoder.norm
        self.embed_dim = encoder.embed_dim
        self.patch_size = encoder.patch_size
        self.head = nn.Linear(self.embed_dim, num_classes)

    def forward(self, x):
        B, C, T, F = x.shape
        p_t, p_f = self.patch_size
        n_t, n_f = T // p_t, F // p_f

        # patchify + embed
        x = self.encoder.patchify(x)
        x = self.patch_embed(x)

        # SinCos positional encoding
        pos = torch.from_numpy(
            get_2d_sincos_pos_embed(self.embed_dim, n_t, n_f)
        ).float().unsqueeze(0).to(x.device)
        x = x + pos

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = x.mean(dim=1)          # global average pooling
        x = self.head(x)
        return x

model = ClassifierHead(encoder, num_classes).to(device)

if state_dict is not None:
    # 只加载 encoder 部分的权重
    own_state = model.state_dict()
    for name, param in state_dict.items():
        if name in own_state and 'head' not in name:
            own_state[name].copy_(param)
    model.load_state_dict(own_state)
    print("Pretrained encoder weights loaded.")

if args.mode == 'lp':
    for name, param in model.named_parameters():
        if 'head' not in name:
            param.requires_grad = False
    print("Linear probing: encoder frozen.")

# ── 训练 ──
optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=args.lr, weight_decay=1e-5
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
best_val_acc = 0.0

import time
for epoch in range(args.epochs):
    model.train()
    total_loss, n_batches = 0.0, 0
    t0 = time.time()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = nn.functional.cross_entropy(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    scheduler.step()
    avg_train_loss = total_loss / max(n_batches, 1)
    elapsed = time.time() - t0
    lr_now = scheduler.get_last_lr()[0]

    # Val
    model.eval()
    correct, total = 0, 0
    val_loss_total, val_batches = 0.0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            val_loss_total += nn.functional.cross_entropy(logits, y).item()
            val_batches += 1
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    val_acc = correct / total
    avg_val_loss = val_loss_total / max(val_batches, 1)

    # ── Log ──
    print(f"Epoch {epoch+1:3d}/{args.epochs} | train_loss={avg_train_loss:.4f} | val_loss={avg_val_loss:.4f} | val_acc={val_acc:.4f} | lr={lr_now:.2e} | {elapsed:.0f}s")
    writer.add_scalar("loss/train", avg_train_loss, epoch)
    writer.add_scalar("loss/val", avg_val_loss, epoch)
    writer.add_scalar("acc/val", val_acc, epoch)
    writer.add_scalar("lr", lr_now, epoch)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), f"{save_dir}/best.pth")

# ── 测试 ──
model.load_state_dict(torch.load(f"{save_dir}/best.pth"))
model.eval()
results = {}
for split_name, loader in test_loaders.items():
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    acc = correct / total
    results[split_name] = round(acc, 4)
    print(f"  {split_name}: {acc:.4f} ({correct}/{total})")

# ── 将测试结果写入 TensorBoard ──
for split_name, acc in results.items():
    writer.add_scalar(f"test_acc/{split_name}", acc, 0)

output = {
    "dataset": args.dataset,
    "mode": args.mode,
    "task": args.task,
    "pretrained_weight": os.path.basename(args.pretrained) if args.pretrained else None,
    "command": raw_command,
    "parsed_args": {
        k: v for k, v in parsed_args.items()
    },
    "results": results,
}
with open(f"{save_dir}/results.json", 'w') as f:
    json.dump(output, f, indent=2)
print(f"Results saved to {save_dir}/results.json")

writer.close()

# 关闭日志重定向
_tee.close()