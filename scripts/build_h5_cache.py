"""一次性扫描所有 H5 文件、验证有效性、保存路径列表到缓存"""
import os, json
import h5py

CSI_BENCH_ROOT = "/mnt/DataDrive164/wr/dataset_wifo/csibench"
CACHE_PATH = "./experiments/csibench/h5_filelist.json"

PRETRAIN_DIRS = [
    "Multitask/sub_Human_h5",
    "FallDetection/sub_Human",
    "MotionSourceRecognition/sub_Human",
    "MotionSourceRecognition/sub_Pet",
    "MotionSourceRecognition/sub_IRobot",
    "MotionSourceRecognition/sub_Fan",
    "BreathingDetection/sub_Human",
    "Localization/sub_Human",
]

files = []
bad_files = 0
for sub in PRETRAIN_DIRS:
    path = os.path.join(CSI_BENCH_ROOT, sub)
    if not os.path.isdir(path):
        print(f"[跳过] {path}")
        continue
    count = 0
    for dirpath, _, fnames in os.walk(path):
        for f in fnames:
            if not f.endswith(".h5"):
                continue
            fpath = os.path.join(dirpath, f)
            # 验证 H5 文件是否有 CSI_amps key
            try:
                with h5py.File(fpath, "r") as hf:
                    if "CSI_amps" not in hf:
                        bad_files += 1
                        print(f"  [无效] {fpath}: 缺少 CSI_amps")
                        continue
                files.append(fpath)
                count += 1
            except Exception as e:
                bad_files += 1
                print(f"  [无效] {fpath}: {e}")
    print(f"  {sub}: {count} 个有效文件")

os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
with open(CACHE_PATH, 'w') as f:
    json.dump(files, f)

print(f"\n总计: {len(files)} 个有效 H5 文件, {bad_files} 个无效文件已排除")
print(f"缓存已保存: {CACHE_PATH}")
