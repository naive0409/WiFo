import h5py, os

base = '/mnt/DataDrive164/wr/dataset_wifo/csibench/Multitask/sub_Human_h5'

# Multitask H5 的形状
print("【Multitask H5文件形状检查】")
shapes = {}
count = 0
for root, dirs, files in os.walk(base):
    for f in files:
        if f.endswith('.h5'):
            fpath = os.path.join(root, f)
            try:
                with h5py.File(fpath, 'r') as hf:
                    s = str(hf['CSI_amps'].shape)
                    shapes[s] = shapes.get(s, 0) + 1
                    count += 1
            except:
                pass
            if count >= 100:
                break
    if count >= 100:
        break

print(f"检查了 {count} 个Multitask H5文件")
for s, c in sorted(shapes.items(), key=lambda x: -x[1]):
    print(f"  形状 {s}: {c}个")

# 检查Multitask的metadata文件路径解析
print("\n【Multitask metadata 文件路径测试】")
import pandas as pd
meta = pd.read_csv('/mnt/DataDrive164/wr/dataset_wifo/csibench/Multitask/HumanActivityRecognition/metadata/sample_metadata.csv')
for i in range(5):
    fpath_rel = meta.iloc[i]['file_path']
    print(f"  metadata路径: {fpath_rel}")
    # 尝试解析
    candidates = [
        os.path.join('/mnt/DataDrive164/wr/dataset_wifo/csibench/Multitask/HumanActivityRecognition', fpath_rel),
        os.path.join('/mnt/DataDrive164/wr/dataset_wifo/csibench', fpath_rel.replace('../../', '')),
        os.path.normpath(os.path.join('/mnt/DataDrive164/wr/dataset_wifo/csibench/Multitask/HumanActivityRecognition/metadata', fpath_rel)),
    ]
    for c in candidates:
        exists = os.path.exists(c)
        print(f"    -> {c}: {'EXISTS' if exists else 'NOT FOUND'}")
        if exists:
            with h5py.File(c, 'r') as hf:
                print(f"      形状: {hf['CSI_amps'].shape}")
            break

# 检查各task metadata中label字段
print("\n【各任务标签字段】")
for task in ['HumanActivityRecognition', 'HumanIdentification', 'ProximityRecognition']:
    meta_path = f'/mnt/DataDrive164/wr/dataset_wifo/csibench/Multitask/{task}/metadata/sample_metadata.csv'
    df = pd.read_csv(meta_path)
    print(f"\n  {task}:")
    print(f"    列: {list(df.columns)}")
    if 'label' in df.columns:
        print(f"    标签值: {df['label'].unique()}")
