# WiFo2D — CSI-Bench 预训练命令

> 基于 WiFo 代码库改造的 2D MAE 模型，遵循 "Scale What Counts" 架构
> 数据: CSI-Bench (~255K H5 样本)
> 目标形状: (1, 500, 232)，patch=(25, 8) → 580 tokens

---

## 环境

```bash
conda activate wifo_csibench
# 预缓存 H5 文件列表（只需执行一次）
python scripts/build_h5_cache.py
```

---

## 调试命令（少量数据验证）

```bash
# 500 样本 × 5 epoch，快速验证 pipeline
CUDA_VISIBLE_DEVICES=0 python src/main.py \
    --model_type wifo2d \
    --size tiny \
    --csibench \
    --csibench_max 500 \
    --batch_size 16 \
    --total_epoches 5 \
    --lr 1e-3 \
    --mask_ratio 0.80 \
    --patch_size_time 25 \
    --patch_size_freq 8 \
    --run_name debug_500

# 5000 样本 × 20 epoch，验证 loss 收敛趋势
CUDA_VISIBLE_DEVICES=0 python src/main.py \
    --model_type wifo2d \
    --size tiny \
    --csibench \
    --csibench_max 5000 \
    --batch_size 128 \
    --total_epoches 20 \
    --lr 1e-3 \
    --mask_ratio 0.80 \
    --patch_size_time 25 \
    --patch_size_freq 8 \
    --run_name debug_5k
```

```bash
CUDA_VISIBLE_DEVICES=1 python src/main.py     --model_type wifo2d     --size tiny     --csibench     --csibench_max 5000     --batch_size 128     --total_epoches 20     --lr 1e-3     --mask_ratio 0.80     --patch_size_time 25     --patch_size_freq 8     --run_name debug_5k

CUDA_VISIBLE_DEVICES=1 python src/main.py     --model_type wifo2d     --size small     --csibench     --csibench_max 5000     --batch_size 256     --total_epoches 20     --lr 1e-3     --mask_ratio 0.80     --patch_size_time 25     --patch_size_freq 8     --run_name debug_small_bs256_5k
```

---

## 全量预训练命令

### ViT-Tiny (3.5M 参数，主实验)

```bash
# 全量 ~255K 样本，100 epoch
CUDA_VISIBLE_DEVICES=0 python src/main.py \
    --model_type wifo2d \
    --size tiny \
    --csibench \
    --batch_size 128 \
    --total_epoches 100 \
    --lr 1e-3 \
    --mask_ratio 0.80 \
    --patch_size_time 25 \
    --patch_size_freq 8 \
    --run_name tiny_full
```

### ViT-Small (15M 参数，对比实验)

```bash
CUDA_VISIBLE_DEVICES=1 python src/main.py \
    --model_type wifo2d \
    --size small \
    --csibench \
    --batch_size 128 \
    --total_epoches 100 \
    --lr 1e-3 \
    --mask_ratio 0.80 \
    --patch_size_time 25 \
    --patch_size_freq 8 \
    --run_name small_full
```

### ViT-Base (86M 参数，大规模实验)

```bash
CUDA_VISIBLE_DEVICES=2 python src/main.py \
    --model_type wifo2d \
    --size base \
    --csibench \
    --batch_size 64 \
    --total_epoches 100 \
    --lr 1e-3 \
    --mask_ratio 0.80 \
    --patch_size_time 25 \
    --patch_size_freq 8 \
    --run_name base_full
```

---

## 数据规模实验（scaling analysis）

```bash
# 1% 数据 (~2550 样本)
CUDA_VISIBLE_DEVICES=0 python src/main.py --model_type wifo2d --size tiny --csibench --csibench_max 2550 --batch_size 128 --total_epoches 100 --lr 1e-3 --mask_ratio 0.80 --patch_size_time 25 --patch_size_freq 8 --run_name tiny_1pct

# 5% 数据 (~12770 样本)
CUDA_VISIBLE_DEVICES=0 python src/main.py --model_type wifo2d --size tiny --csibench --csibench_max 12770 --batch_size 128 --total_epoches 100 --lr 1e-3 --mask_ratio 0.80 --patch_size_time 25 --patch_size_freq 8 --run_name tiny_5pct

# 10% 数据 (~25540 样本)
CUDA_VISIBLE_DEVICES=0 python src/main.py --model_type wifo2d --size tiny --csibench --csibench_max 25540 --batch_size 128 --total_epoches 100 --lr 1e-3 --mask_ratio 0.80 --patch_size_time 25 --patch_size_freq 8 --run_name tiny_10pct

# 25% 数据 (~63850 样本)
CUDA_VISIBLE_DEVICES=0 python src/main.py --model_type wifo2d --size tiny --csibench --csibench_max 63850 --batch_size 128 --total_epoches 100 --lr 1e-3 --mask_ratio 0.80 --patch_size_time 25 --patch_size_freq 8 --run_name tiny_25pct

# 50% 数据 (~127700 样本)
CUDA_VISIBLE_DEVICES=0 python src/main.py --model_type wifo2d --size tiny --csibench --csibench_max 127700 --batch_size 128 --total_epoches 100 --lr 1e-3 --mask_ratio 0.80 --patch_size_time 25 --patch_size_freq 8 --run_name tiny_50pct
```

```bash
0611 small 25k 300e (no random sanmple)
CUDA_VISIBLE_DEVICES=1 python src/main.py     --model_type wifo2d     --size small     --csibench     --csibench_max 25540     --batch_size 256     --total_epoches 200     --lr 1e-3     --mask_ratio 0.80     --patch_size_time 25     --patch_size_freq 8     --run_name debug_small_bs256_25k

0612 small 63k 300e (no random sanmple)
CUDA_VISIBLE_DEVICES=1 python src/main.py     --model_type wifo2d     --size small     --csibench     --csibench_max 63850     --batch_size 256     --total_epoches 300     --lr 1e-3     --mask_ratio 0.80     --patch_size_time 25     --patch_size_freq 8     --run_name debug_small_bs256_63k

0613 small full 300e
CUDA_VISIBLE_DEVICES=1 python src/main.py     --model_type wifo2d     --size small     --csibench                              --batch_size 256     --total_epoches 300     --lr 1e-3     --mask_ratio 0.80     --patch_size_time 25     --patch_size_freq 8     --run_name debug_small_bs256_full

0615 small 63k 200e
CUDA_VISIBLE_DEVICES=1 python src/main.py     --model_type wifo2d     --size small     --csibench     --csibench_max 63850     --batch_size 256     --total_epoches 200     --lr 1e-3     --mask_ratio 0.80     --patch_size_time 25     --patch_size_freq 8     --run_name small_63k

0616 small 25k 200e
CUDA_VISIBLE_DEVICES=1 python src/main.py     --model_type wifo2d     --size small     --csibench     --csibench_max 25540     --batch_size 256     --total_epoches 200     --lr 1e-3     --mask_ratio 0.80     --patch_size_time 25     --patch_size_freq 8     --run_name small_25k

0616 tiny full 200e
CUDA_VISIBLE_DEVICES=1 python src/main.py     --model_type wifo2d     --size tiny     --csibench                              --batch_size 256     --total_epoches 200     --lr 1e-3     --mask_ratio 0.80     --patch_size_time 25     --patch_size_freq 8     --run_name tiny_full

```

---

## 重构可视化

```bash
# 默认 (使用 experiments/csibench/wifo2d_tiny_best.pth)
CUDA_VISIBLE_DEVICES=0 python scripts/visualize_reconstruction.py

# 指定权重路径
CUDA_VISIBLE_DEVICES=1 python scripts/visualize_reconstruction.py \
    --model_path ./experiments/csibench/debug_5k/wifo2d_tiny_best.pth \
    --size tiny \
    --mask_ratio 0.80 \
    --n_samples 8

CUDA_VISIBLE_DEVICES=1 python scripts/visualize_reconstruction.py \
    --model_path ./experiments/csibench/debug_small_bs256_5k/wifo2d_small_best.pth \
    --size small \
    --mask_ratio 0.80 \
    --n_samples 8

CUDA_VISIBLE_DEVICES=1 python scripts/visualize_reconstruction.py \
    --model_path ./experiments/csibench/debug_small_bs256_25k/wifo2d_small_best.pth \
    --size small \
    --mask_ratio 0.80 \
    --n_samples 8

# 参数说明:
#   --model_path  模型权重路径 (默认: ./experiments/csibench/wifo2d_tiny_best.pth)
#   --size        模型尺寸, tiny/small/base (默认: tiny)
#   --mask_ratio  mask 比例 (默认: 0.80)
#   --n_samples   可视化样本数 (默认: 8)
#   --save_dir    输出目录 (默认: ./visualizations)
```

输出: `./visualizations/reconstruction_{size}_{weight_name}.png`

---

## 训练监控

TensorBoard 日志集中存放在 `experiments/tensorboard/<run_name>/`，方便对比不同实验：

```bash
# 启动 TensorBoard（在项目根目录），加载所有实验
tensorboard --logdir ./experiments/tensorboard --port 6006
```

记录指标:
- `loss/train` — 每个 epoch 的训练 loss
- `loss/val` — 每个 epoch 的验证 loss (10% 数据)
- `lr` — 学习率

验证集用于选择 best 模型（按 val loss 最优保存），也可监控过拟合（train loss 降但 val loss 升）。
训练结束后在测试集 (10% 数据) 上计算最终 test loss。

---

---

## 下游分类微调

在 CSI-Bench Multitask 任务上微调预训练模型，评估跨域泛化。

### 命令格式

```bash
# 全量微调 (FT) — 更新全部参数
CUDA_VISIBLE_DEVICES=0 python scripts/finetune.py \
    --dataset csibench \
    --pretrained ./experiments/csibench/debug_small_bs256_full/wifo2d_small_best.pth \
    --task HumanActivityRecognition \
    --mode ft \
    --size small \
    --batch_size 128 \
    --epochs 50

# 线性探测 (LP) — 冻结编码器，只训练分类头
CUDA_VISIBLE_DEVICES=0 python scripts/finetune.py \
    --dataset csibench \
    --pretrained ./experiments/csibench/debug_small_bs256_full/wifo2d_small_best.pth \
    --task HumanActivityRecognition \
    --mode lp \
    --size small \
    --batch_size 1024 \
    --epochs 50

# 有监督基线 (supervised) — 不加载预训练权重，从头训练
CUDA_VISIBLE_DEVICES=0 python scripts/finetune.py \
    --dataset csibench \
    --task HumanActivityRecognition \
    --mode supervised \
    --size small \
    --batch_size 128 \
    --epochs 50
```

### 参数说明

| 参数 | 说明 |
|------|------|
| `--pretrained` | 预训练权重路径 (scratch 模式不需要) |
| `--task` | HumanActivityRecognition / HumanIdentification / ProximityRecognition |
| `--mode` | ft (全量微调) / lp (线性探测) / supervised (有监督基线) |
| `--size` | tiny / small / base (必须与预训练一致) |
| `--batch_size` | 默认 128（FT/supervised 推荐 128，LP 推荐 1024） |
| `--epochs` | 默认 50 |
| `--held_out` | 跨域评估: 留出某个 domain（如 env_Scene1, subject_01），不指定则为 in-domain |
| `--run_name` | 自定义名称，默认自动生成 |

### 测试集自动评估

训练完成后自动在以下 split 上评估准确率:
- `test_id` — 同域
- `test_cross_user` — 跨用户
- `test_cross_env` — 跨环境
- `test_cross_device` — 跨设备

结果保存在 `./experiments/finetune/<run_name>/results.json`

### 推荐实验顺序

```bash
# 1. 有监督基线 (supervised)
CUDA_VISIBLE_DEVICES=0 python scripts/finetune.py --dataset csibench --task HumanActivityRecognition --mode supervised --size small --batch_size 128 --epochs 50 --run_name har_supervised

# 2. 线性探测 (small_25k 预训练)
CUDA_VISIBLE_DEVICES=0 python scripts/finetune.py --dataset csibench --pretrained ./experiments/csibench/small_25k/wifo2d_small_best.pth --task HumanActivityRecognition --mode lp --size small --batch_size 1024 --epochs 50 --run_name har_lp_25k

# 3. 全量微调 (small_25k 预训练)
CUDA_VISIBLE_DEVICES=0 python scripts/finetune.py --dataset csibench --pretrained ./experiments/csibench/small_25k/wifo2d_small_best.pth --task HumanActivityRecognition --mode ft --size small --batch_size 128 --epochs 50 --run_name har_ft_25k

# 4. 全量微调 (small_63k 预训练)
CUDA_VISIBLE_DEVICES=0 python scripts/finetune.py --dataset csibench --pretrained ./experiments/csibench/small_63k/wifo2d_small_best.pth --task HumanActivityRecognition --mode ft --size small --batch_size 128 --epochs 50 --run_name har_ft_63k

# 5. 全量微调 (full 预训练)
CUDA_VISIBLE_DEVICES=0 python scripts/finetune.py --dataset csibench --pretrained ./experiments/csibench/debug_small_bs256_full/wifo2d_small_best.pth --task HumanActivityRecognition --mode ft --size small --batch_size 128 --epochs 50 --run_name har_ft_full

# 6. 同样的流程可换到 HumanIdentification 和 ProximityRecognition
```

---

### WiMANS 九组完整实验命令

```bash
# ═══════════════════════════════════════════════
# 有监督基线 (supervised) — 不加载预训练权重
# ═══════════════════════════════════════════════

# supervised in-domain
CUDA_VISIBLE_DEVICES=0 python scripts/finetune.py \
    --dataset wimans --mode supervised --size small --batch_size 128 --epochs 50 \
    --run_name har_wimans_sup_id

# supervised cross-env (留 classroom 出)
CUDA_VISIBLE_DEVICES=0 python scripts/finetune.py \
    --dataset wimans --mode supervised --size small --batch_size 128 --epochs 50 \
    --held_out env_classroom --run_name har_wimans_sup_cross_env

# supervised cross-subject (留 subject a 出)
CUDA_VISIBLE_DEVICES=0 python scripts/finetune.py \
    --dataset wimans --mode supervised --size small --batch_size 128 --epochs 50 \
    --held_out subject_a --run_name har_wimans_sup_cross_subject


# ═══════════════════════════════════════════════
# 线性探测 (LP) — 冻结编码器，只训练分类头
# ═══════════════════════════════════════════════

# LP in-domain
CUDA_VISIBLE_DEVICES=0 python scripts/finetune.py \
    --dataset wimans --mode lp --size small --batch_size 1024 --epochs 50 \
    --pretrained ./experiments/csibench/debug_small_bs256_full/wifo2d_small_best.pth \
    --run_name har_wimans_lp_id

# LP cross-env (留 classroom 出)
CUDA_VISIBLE_DEVICES=0 python scripts/finetune.py \
    --dataset wimans --mode lp --size small --batch_size 1024 --epochs 50 \
    --pretrained ./experiments/csibench/debug_small_bs256_full/wifo2d_small_best.pth \
    --held_out env_classroom --run_name har_wimans_lp_cross_env

# LP cross-subject (留 subject a 出)
CUDA_VISIBLE_DEVICES=0 python scripts/finetune.py \
    --dataset wimans --mode lp --size small --batch_size 1024 --epochs 50 \
    --pretrained ./experiments/csibench/debug_small_bs256_full/wifo2d_small_best.pth \
    --held_out subject_a --run_name har_wimans_lp_cross_subject


# ═══════════════════════════════════════════════
# 全量微调 (FT) — 更新全部参数
# ═══════════════════════════════════════════════

# FT in-domain
CUDA_VISIBLE_DEVICES=0 python scripts/finetune.py \
    --dataset wimans --mode ft --size small --batch_size 128 --epochs 50 \
    --pretrained ./experiments/csibench/debug_small_bs256_full/wifo2d_small_best.pth \
    --run_name har_wimans_ft_id

# FT cross-env (留 classroom 出)
CUDA_VISIBLE_DEVICES=0 python scripts/finetune.py \
    --dataset wimans --mode ft --size small --batch_size 128 --epochs 50 \
    --pretrained ./experiments/csibench/debug_small_bs256_full/wifo2d_small_best.pth \
    --held_out env_classroom --run_name har_wimans_ft_cross_env

# FT cross-subject (留 subject a 出)
CUDA_VISIBLE_DEVICES=0 python scripts/finetune.py \
    --dataset wimans --mode ft --size small --batch_size 128 --epochs 50 \
    --pretrained ./experiments/csibench/debug_small_bs256_full/wifo2d_small_best.pth \
    --held_out subject_a --run_name har_wimans_ft_cross_subject
```

### WiMANS 数据集说明

| 属性 | 值 |
|------|-----|
| 来源 | WiMANS: A Benchmark Dataset for WiFi-based Multi-user Activity Sensing |
| 样本数 | 11,286（有效 ~4,752） |
| 类别 | 9 类: nothing, walk, rotation, jump, wave, lie_down, pick_up, sit_down, stand_up |
| 环境 | classroom / meeting_room / empty_room（各 1,584 条，均衡）|
| Subject | a~e（a: 1,296 条, b~e: 各 864 条）|
| 原始形状 | (2901, 3, 3, 30) → 预处理为 (1, 500, 232) |

---

### XRF55 九组完整实验命令

```bash
# ═══════════════════════════════════════════════
# 有监督基线 (supervised) — 不加载预训练权重
# ═══════════════════════════════════════════════

# supervised in-domain
CUDA_VISIBLE_DEVICES=0 python scripts/finetune.py \
    --dataset xrf55 --mode supervised --size small --batch_size 128 --epochs 50 \
    --run_name har_xrf55_sup_id

# supervised cross-env (留 Scene2 出)
CUDA_VISIBLE_DEVICES=0 python scripts/finetune.py \
    --dataset xrf55 --mode supervised --size small --batch_size 128 --epochs 50 \
    --held_out env_Scene2 --run_name har_xrf55_sup_cross_env

# supervised cross-subject (留 subject 01 出)
CUDA_VISIBLE_DEVICES=0 python scripts/finetune.py \
    --dataset xrf55 --mode supervised --size small --batch_size 128 --epochs 50 \
    --held_out subject_01 --run_name har_xrf55_sup_cross_subject


# ═══════════════════════════════════════════════
# 线性探测 (LP) — 冻结编码器，只训练分类头
# ═══════════════════════════════════════════════

# LP in-domain
CUDA_VISIBLE_DEVICES=0 python scripts/finetune.py \
    --dataset xrf55 --mode lp --size small --batch_size 1024 --epochs 50 \
    --pretrained ./experiments/csibench/debug_small_bs256_full/wifo2d_small_best.pth \
    --run_name har_xrf55_lp_id

# LP cross-env (留 Scene2 出)
CUDA_VISIBLE_DEVICES=0 python scripts/finetune.py \
    --dataset xrf55 --mode lp --size small --batch_size 1024 --epochs 50 \
    --pretrained ./experiments/csibench/debug_small_bs256_full/wifo2d_small_best.pth \
    --held_out env_Scene2 --run_name har_xrf55_lp_cross_env

# LP cross-subject (留 subject 01 出)
CUDA_VISIBLE_DEVICES=0 python scripts/finetune.py \
    --dataset xrf55 --mode lp --size small --batch_size 1024 --epochs 50 \
    --pretrained ./experiments/csibench/debug_small_bs256_full/wifo2d_small_best.pth \
    --held_out subject_01 --run_name har_xrf55_lp_cross_subject


# ═══════════════════════════════════════════════
# 全量微调 (FT) — 更新全部参数
# ═══════════════════════════════════════════════

# FT in-domain
CUDA_VISIBLE_DEVICES=1 python scripts/finetune.py \
    --dataset xrf55 --mode ft --size small --batch_size 128 --epochs 50 \
    --pretrained ./experiments/csibench/debug_small_bs256_full/wifo2d_small_best.pth \
    --run_name har_xrf55_ft_id

# FT cross-env (留 Scene2 出)
CUDA_VISIBLE_DEVICES=1 python scripts/finetune.py \
    --dataset xrf55 --mode ft --size small --batch_size 128 --epochs 50 \
    --pretrained ./experiments/csibench/debug_small_bs256_full/wifo2d_small_best.pth \
    --held_out env_Scene2 --run_name har_xrf55_ft_cross_env

# FT cross-subject (留 subject 01 出)
CUDA_VISIBLE_DEVICES=1 python scripts/finetune.py \
    --dataset xrf55 --mode ft --size small --batch_size 128 --epochs 50 \
    --pretrained ./experiments/csibench/debug_small_bs256_full/wifo2d_small_best.pth \
    --held_out subject_01 --run_name har_xrf55_ft_cross_subject
```

### XRF55 数据集说明

| 属性 | 值 |
|------|-----|
| 来源 | XRF55: A Comprehensive Dataset for RF-based Human Activity Recognition |
| 样本数 | 22,000（仅 WiFi 模态）|
| 类别 | **55 类**（活动 01~55）|
| 场景 | 4（Scene1: 11 subject, 12,100 样本; Scene2~4: 各 3 subject, 3,300 样本）|
| Subject | 15 个 |
| 原始形状 | (270, 1000) → 预处理为 (1, 500, 232) |

### GaitID 九组完整实验命令 (User Identification)

运行前请确保已激活 conda 环境：
```bash
conda activate wifo_csibench
```

```bash
# ═══════════════════════════════════════════════
# 有监督基线 (supervised) — 不加载预训练权重
# ═══════════════════════════════════════════════

# supervised in-domain
CUDA_VISIBLE_DEVICES=0 python scripts/finetune.py \
    --dataset gait --mode supervised --size small --batch_size 128 --epochs 50 \
    --run_name uid_gait_sup_id

# supervised cross-track (留 track 4 出)
CUDA_VISIBLE_DEVICES=2 python scripts/finetune.py \
    --dataset gait --mode supervised --size small --batch_size 128 --epochs 50 \
    --held_out track_4 --run_name uid_gait_sup_cross_track

# supervised cross-env (留 Room#2 / 20190719 出)
CUDA_VISIBLE_DEVICES=0 python scripts/finetune.py \
    --dataset gait --mode supervised --size small --batch_size 128 --epochs 50 \
    --held_out env_20190719 --run_name uid_gait_sup_cross_env


# ═══════════════════════════════════════════════
# 线性探测 (LP) — 冻结编码器，只训练分类头
# ═══════════════════════════════════════════════

# LP in-domain
CUDA_VISIBLE_DEVICES=0 python scripts/finetune.py \
    --dataset gait --mode lp --size small --batch_size 1024 --epochs 50 \
    --pretrained ./experiments/csibench/debug_small_bs256_full/wifo2d_small_best.pth \
    --run_name uid_gait_lp_id

# LP cross-track (留 track 4 出)
CUDA_VISIBLE_DEVICES=0 python scripts/finetune.py \
    --dataset gait --mode lp --size small --batch_size 1024 --epochs 50 \
    --pretrained ./experiments/csibench/debug_small_bs256_full/wifo2d_small_best.pth \
    --held_out track_4 --run_name uid_gait_lp_cross_track

# LP cross-env (留 Room#2 / 20190719 出)
CUDA_VISIBLE_DEVICES=0 python scripts/finetune.py \
    --dataset gait --mode lp --size small --batch_size 1024 --epochs 50 \
    --pretrained ./experiments/csibench/debug_small_bs256_full/wifo2d_small_best.pth \
    --held_out env_20190719 --run_name uid_gait_lp_cross_env


# ═══════════════════════════════════════════════
# 全量微调 (FT) — 更新全部参数
# 注意: FT 使用较低学习率 (5e-5) 以避免灾难性遗忘，
# 因为预训练模型是在 CSI-Bench（而非 14 数据集）上训练的，
# 预训练特征与 GaitID 的步态识别任务有领域差异。
# ═══════════════════════════════════════════════

# FT in-domain
CUDA_VISIBLE_DEVICES=0 python scripts/finetune.py \
    --dataset gait --mode ft --size small --batch_size 128 --epochs 50 --lr 5e-5 \
    --pretrained ./experiments/csibench/debug_small_bs256_full/wifo2d_small_best.pth \
    --run_name uid_gait_ft_id

# FT cross-track (留 track 4 出)
CUDA_VISIBLE_DEVICES=1 python scripts/finetune.py \
    --dataset gait --mode ft --size small --batch_size 128 --epochs 50 --lr 5e-5 \
    --pretrained ./experiments/csibench/debug_small_bs256_full/wifo2d_small_best.pth \
    --held_out track_4 --run_name uid_gait_ft_cross_track

# FT cross-env (留 Room#2 / 20190719 出)
CUDA_VISIBLE_DEVICES=0 python scripts/finetune.py \
    --dataset gait --mode ft --size small --batch_size 128 --epochs 50 --lr 5e-5 \
    --pretrained ./experiments/csibench/debug_small_bs256_full/wifo2d_small_best.pth \
    --held_out env_20190719 --run_name uid_gait_ft_cross_env
```

### GaitID 数据集说明

| 属性 | 值 |
|------|-----|
| 来源 | GaitID: Robust Wi-Fi Based Gait Recognition (WASA'2020) |
| 任务 | User Identification (UID) |
| 用户数 | 11 类 (user1~user11) |
| 总 trials | ~3,652 (每 trial = 6 接收器) |
| 环境 | Room#1 教室 (20190627~20190718) / Room#2 大厅 (20190719) |
| 行走轨迹 | 4 条 (track 1~4, 不同位置/方向) |
| 跨域方式 | `--held_out track_4` (cross-track) / `--held_out env_20190719` (cross-env) |
| 原始形状 | (T, 3, 3, 30) int8 复数 → 预处理为 (1, 500, 232) |

**跨域说明**:
- **cross-track**: 留出 track 4 作为测试集，训练集使用 track 1~3。全部 11 个用户可用。
- **cross-env**: 留出 20190719 (Room #2 大厅) 作为测试环境，训练集使用 Room #1 (教室, 20190627~20190718)。测试集仅保留训练集中已出现的用户（user1, user2），避免跨环境+跨用户混杂。

**FT 性能低于 supervised 的说明**:
预训练模型仅在 CSI-Bench（单数据集）上训练，而原文使用了 14 个数据集。CSI-Bench 主要是 HAR 任务，与 GaitID 的 UID 任务存在领域差异。FT 时默认学习率 1e-4 会过快覆盖预训练特征（灾难性遗忘）。解决方案：
1. **降低 FT 学习率**: FT 命令已改为 `--lr 5e-5`，更温和地微调
2. **更少 epochs**: 可选 `--epochs 30` 配合早停（训练完成后选择最佳 val 权重）
3. **两阶段微调**: 先 LP（冻结编码器训练分类头 10 epoch），再 FT（解冻全部用低 LR 微调 20 epoch），需手动分步执行

**关于准确率无法达到论文水平的进一步分析**:
即使 supervised（与预训练无关）也远低于论文的 92.7%（cross-track）/ 50.4%（cross-env），核心原因是**数据预处理差异**：
1. **静态路径干扰**: 原始 CSI 幅度包含大量静态环境分量（墙壁、家具），步态的时变信号非常微弱（仅占总幅度的 ~2%）。模型学到的是 Room #1 的静态特征而非用户步态特征。
2. **DC 去除修复**: 已添加 `csi_amp -= csi_amp.mean(axis=0)` 去除每个子载波的时间均值，仅保留时变运动分量。kNN 基线从 16% → 27%（cross-env），证明了该方法有效。
3. **跨环境难度: Room #1→Room #2**: 我们的 cross-env 是两个不同房间（教室→大厅），原文可能使用同房间不同 session 作为"环境"，难度小得多。这是跨房间测试，更具挑战性。
4. **跨设备/跨环境差异**: Room #1 和 Room #2 的 CSI 幅度分布差异大，DC 去除后 kNN 提升明显，但完全消除领域漂移仍需更大规模预训练数据。

---

### 已支持的数据集一览

| `--dataset` | 任务 | 类别数 | 总样本 | 跨域方式 |
|:-----------:|------|:-----:|:------:|:--------:|
| `csibench` | HAR / UID / Prox | 5~6 | 55,684 (HAR) | 内置 split（`train_id` 自动排除跨域测试集）|
| `wimans` | HAR | 9 | ~4,752 | `--held_out env_*` / `--held_out subject_*` |
| `xrf55` | HAR | 55 | 22,000 | `--held_out env_*` / `--held_out subject_*` |
| `gait` | UID | **11** | ~3,652 | `--held_out track_*` (cross-track) / `--held_out env_20190719` (cross-env) |

**注意**: `csibench` 使用预定义的 split JSON 文件自动保证跨域测试集与训练集互斥；`wimans`、`xrf55` 和 `gait` 需要通过 `--held_out` 参数指定留出哪个 domain，每次运行训练一个 leave-one-out fold。

---

## 模型保存位置

```
./experiments/csibench/
├── wifo2d_tiny_best.pth     # 验证 loss 最优
├── wifo2d_tiny_final.pth    # 最终 epoch
├── wifo2d_tiny_ep20.pth     # 每 20 epoch 的检查点
├── wifo2d_tiny_ep40.pth
├── wifo2d_small_best.pth
├── wifo2d_base_best.pth
└── h5_filelist.json          # H5 文件缓存
```
