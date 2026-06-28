import h5py, numpy as np, os, pandas as pd, sys

base = '/mnt/DataDrive164/wr/dataset_wifo/csibench'

# 1. 从metadata读取file_path来检查H5形状 —— 快速，只读metadata中引用的文件
print("=" * 70)
print("【从metadata引用的H5文件检查形状】")
print("=" * 70)

tasks_meta = [
    ('Multitask/HumanActivityRecognition', 'Multitask'),
    ('Multitask/HumanIdentification', 'Multitask'),
    ('Multitask/ProximityRecognition', 'Multitask'),
    ('FallDetection', 'Single'),
    ('MotionSourceRecognition', 'Single'),
    ('BreathingDetection', 'Single'),
    ('Localization', 'Single'),
]

total_labeled = 0
all_shapes = {}

for task_rel, task_type in tasks_meta:
    meta_path = os.path.join(base, task_rel, 'metadata', 'sample_metadata.csv')
    if not os.path.exists(meta_path):
        print(f"  {task_rel}: metadata不存在")
        continue
    
    df = pd.read_csv(meta_path)
    n_total = len(df)
    
    # 前50个文件的形状统计
    shapes = {}
    n_checked = 0
    for _, row in df.head(50).iterrows():
        # 解析file_path (相对路径，需还原)
        fpath_rel = row['file_path']
        # 尝试解析绝对路径
        candidates = [
            os.path.join(base, task_rel, fpath_rel),
            os.path.join(base, fpath_rel),
            os.path.join(base, task_rel.split('/')[0], fpath_rel) if '/' in task_rel else None
        ]
        resolved = None
        for c in candidates:
            if c and os.path.exists(c):
                resolved = c
                break
        
        if resolved is None:
            # 从路径提取信息试另一个方式
            if 'sub_Human_h5' in fpath_rel or 'sub_Human' in fpath_rel:
                # Try walking up from base
                fpath_abs = os.path.join(base, fpath_rel if not fpath_rel.startswith('../') else fpath_rel.replace('../../', ''))
                if os.path.exists(fpath_abs):
                    resolved = fpath_abs
        
        if resolved and os.path.exists(resolved):
            try:
                with h5py.File(resolved, 'r') as hf:
                    data = hf['CSI_amps']
                    shape = data.shape
                    s = str(shape)
                    shapes[s] = shapes.get(s, 0) + 1
                    n_checked += 1
            except Exception as e:
                pass
    
    print(f"\n  {task_rel} ({task_type}):")
    print(f"    metadata总样本: {n_total}")
    for s, c in shapes.items():
        print(f"    H5形状 {s}: 检查到{c}个")
    total_labeled += n_total

print(f"\n  有标签样本总计: {total_labeled}")

# 2. WiFo数据集的形状对比
print("\n" + "=" * 70)
print("【WiFo 数据集与 CSI-Bench 形状对比】")
print("=" * 70)
print("""
WiFo (当前代码):
  输入: (N, 2, T, H, W) — 2通道(实部+虚部), T=时间, H=频域子载波, W=空间天线
  示例: (256, 2, 12, 16, 8) — 12时间步 × 16子载波 × 8天线
  格式: .mat 复数矩阵 → 拆分实部虚部

CSI-Bench:
  输入: (N_subcarriers, N_time, 1) — 幅度值 (float32)
  示例: (232, 500, 1) — 232子载波 × 500时间步 × 1通道
  格式: .h5 'CSI_amps' 键, 已预处理为幅度
  
核心差异:
  - WiFo: 3维 (时/频/空) + 复数 → (2, T, H, W)
  - CSI-Bench: 2维 (时×子载波) + 单通道幅度 → (N_subcarriers, N_time)
  - 需要将 CSI-Bench 从 (F, T, 1) → permute → (1, T, F) 作为2D图像输入
""")
