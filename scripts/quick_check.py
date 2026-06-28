"""Quick data check for XRF55 cross-env"""
import sys, os
sys.path.insert(0, 'src')
os.environ['PYTHONPATH'] = 'src'

# Must use conda env: source activate wifo_csibench
from xrf55_dataset import XRF55FinetuneDataset, _load_all_samples
from collections import Counter

all_s = _load_all_samples()
print(f'Total samples: {len(all_s)}')

scene_cnt = Counter(s['scene'] for s in all_s)
for s, c in sorted(scene_cnt.items()):
    print(f'  {s}: {c}')

ds_train = XRF55FinetuneDataset(split='train', held_out='env_Scene2')
ds_test  = XRF55FinetuneDataset(split='test',  held_out='env_Scene2')
print(f'\ncross-env Scene2: train={len(ds_train)}, test={len(ds_test)}')

train_labels = set(ds_train.data[i]['label'] for i in range(len(ds_train)))
test_labels  = set(ds_test.data[i]['label'] for i in range(len(ds_test)))
print(f'  train classes: {len(train_labels)}, test classes: {len(test_labels)}')

csi, label = ds_test[0]
print(f'\nSample: shape={csi.shape}, label={label}')
print(f'  range: {csi.min().item():.3f} ~ {csi.max().item():.3f}')
print(f'  mean={csi.mean().item():.3f}, std={csi.std().item():.3f}')

# Check test samples are from Scene2
scene2_subjects = set(s['subject'] for s in all_s if s['scene'] == 'Scene2')
print(f'\nScene2 subjects: {scene2_subjects}')
test_subjects = set(s['subject'] for s in ds_test.data)
print(f'Test subjects: {test_subjects}')
