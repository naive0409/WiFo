import h5py, os, pandas as pd

base = '/mnt/DataDrive164/wr/dataset_wifo/csibench'

# 各任务metadata样本数
tasks = {
    'Multitask/HAR': ('Multitask/HumanActivityRecognition', None),
    'Multitask/UID': ('Multitask/HumanIdentification', None),
    'Multitask/Prox': ('Multitask/ProximityRecognition', None),
    'FallDetection': ('FallDetection', None),
    'MotionSource': ('MotionSourceRecognition', None),
    'Breathing': ('BreathingDetection', None),
    'Localization': ('Localization', None),
}

# metadata样本数
print("【各任务有标签样本数】")
total_labeled = 0
for name, (path, _) in tasks.items():
    meta = os.path.join(base, path, 'metadata', 'sample_metadata.csv')
    if os.path.exists(meta):
        df = pd.read_csv(meta)
        total_labeled += len(df)
        print(f"  {name:20s}: {len(df):>7d}")

print(f"  {'TOTAL':20s}: {total_labeled:>7d}")

# Multitask H5 形状统计 (更多样本)
print("\n【Multitask H5 全部形状统计】")
mt_h5 = os.path.join(base, 'Multitask', 'sub_Human_h5')
shapes = {}
total = 0
for root, dirs, files in os.walk(mt_h5):
    for f in files:
        if f.endswith('.h5'):
            fpath = os.path.join(root, f)
            try:
                with h5py.File(fpath, 'r') as hf:
                    s = str(hf['CSI_amps'].shape)
                    shapes[s] = shapes.get(s, 0) + 1
                    total += 1
            except:
                pass

for s, c in sorted(shapes.items(), key=lambda x: -x[1]):
    print(f"  {s:>15s}: {c:>6d}个 ({100*c/total:.1f}%)")
print(f"  {'TOTAL':>15s}: {total:>6d}个")

# 根据freq字段分析子载波数
print("\n【Multitask 不同子载波数的样本分布】")
freq_shapes = {}
for root, dirs, files in os.walk(mt_h5):
    for f in files:
        if f.endswith('.h5') and '__freq' in f:
            freq = f.split('__freq')[1].split('.')[0]
            freq_shapes[freq] = freq_shapes.get(freq, 0) + 1

for freq, count in sorted(freq_shapes.items(), key=lambda x: -x[1]):
    print(f"  freq={freq:>3s} (子载波): {count:>6d}个 ({100*count/total:.1f}%)")
