import h5py, numpy as np, os, pandas as pd

base = '/mnt/DataDrive164/wr/dataset_wifo/csibench'

# 1. 统计每个子任务的所有H5文件形状
print("=" * 70)
print("【CSI-Bench 各子任务H5文件形状统计】")
print("=" * 70)

for d in sorted(os.listdir(base)):
    dpath = os.path.join(base, d)
    if not os.path.isdir(dpath) or d in ('RawContinuousRecording',):
        continue
    shapes = {}
    count = 0
    for root, dirs, files in os.walk(dpath):
        for f in files:
            if not f.endswith('.h5'):
                continue
            fpath = os.path.join(root, f)
            try:
                with h5py.File(fpath, 'r') as hf:
                    data = hf['CSI_amps'][:]
                    shape = data.shape
                    key = str(shape)
                    shapes[key] = shapes.get(key, 0) + 1
                    count += 1
            except:
                pass
            if count >= 20:
                break
        if count >= 20:
            break
    
    print(f"\n--- {d} ({'检查' if count < 20 else '样本'}: {count}个文件) ---")
    for s, c in shapes.items():
        print(f"  形状 {s}: {c}个文件")

# 2. 总H5文件数
print("\n" + "=" * 70)
print("【各子任务总H5文件数】")
print("=" * 70)
total_all = 0
for d in sorted(os.listdir(base)):
    dpath = os.path.join(base, d)
    if not os.path.isdir(dpath) or d in ('RawContinuousRecording',):
        continue
    count = sum(1 for _, _, fs in os.walk(dpath) for f in fs if f.endswith('.h5'))
    total_all += count
    print(f"  {d}: {count}")

print(f"\n  总计: {total_all}")

# 3. 各子任务metadata
print("\n" + "=" * 70)
print("【Metadata 有标签样本统计】")
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

for task_rel, task_type in tasks_meta:
    meta_path = os.path.join(base, task_rel, 'metadata', 'sample_metadata.csv')
    if os.path.exists(meta_path):
        df = pd.read_csv(meta_path)
        print(f"\n  {task_rel} ({task_type}):")
        print(f"    样本数: {len(df)}")
        if 'label' in df.columns:
            print(f"    类别: {df['label'].nunique()}")
            print(f"    分布: {df['label'].value_counts().to_dict()}")

# 4. Multitask的子载波和时间统一性检查
print("\n" + "=" * 70)
print("【Multitask H5样本形状一致性检查】")
print("=" * 70)
mt_path = os.path.join(base, 'Multitask', 'sub_Human_h5')
h5_count = 0
shapes_seen = {}
for root, dirs, files in os.walk(mt_path):
    for f in files:
        if f.endswith('.h5'):
            fpath = os.path.join(root, f)
            try:
                with h5py.File(fpath, 'r') as hf:
                    s = str(hf['CSI_amps'].shape)
                    shapes_seen[s] = shapes_seen.get(s, 0) + 1
                    h5_count += 1
            except:
                pass

print(f"  Multitask sub_Human_h5 总数: {h5_count}")
for s, c in sorted(shapes_seen.items(), key=lambda x: -x[1]):
    print(f"    {s}: {c}个 ({100*c/h5_count:.1f}%)")
