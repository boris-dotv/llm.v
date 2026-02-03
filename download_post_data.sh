#!/bin/bash
cd /public_hw/share/cit_ztyu/cz/nanochat
source .venv/bin/activate

export HF_ENDPOINT=https://hf-mirror.com
export HF_DATASETS_CACHE=/public_hw/share/cit_ztyu/cz/nanochat/base_data/hf_datasets

echo "=== Downloading ARC dataset ==="
python -c "
from datasets import load_dataset

# ARC-Easy
ds_easy_train = load_dataset('allenai/ai2_arc', 'ARC-Easy', split='train', cache_dir='/public_hw/share/cit_ztyu/cz/nanochat/base_data/hf_datasets')
print(f'ARC-Easy train: {len(ds_easy_train)} rows')
ds_easy_train.save_to_disk('/public_hw/share/cit_ztyu/cz/nanochat/base_data/arc_easy_train')

ds_easy_validation = load_dataset('allenai/ai2_arc', 'ARC-Easy', split='validation', cache_dir='/public_hw/share/cit_ztyu/cz/nanochat/base_data/hf_datasets')
print(f'ARC-Easy validation: {len(ds_easy_validation)} rows')
ds_easy_validation.save_to_disk('/public_hw/share/cit_ztyu/cz/nanochat/base_data/arc_easy_validation')

# ARC-Challenge
ds_challenge_train = load_dataset('allenai/ai2_arc', 'ARC-Challenge', split='train', cache_dir='/public_hw/share/cit_ztyu/cz/nanochat/base_data/hf_datasets')
print(f'ARC-Challenge train: {len(ds_challenge_train)} rows')
ds_challenge_train.save_to_disk('/public_hw/share/cit_ztyu/cz/nanochat/base_data/arc_challenge_train')

ds_challenge_validation = load_dataset('allenai/ai2_arc', 'ARC-Challenge', split='validation', cache_dir='/public_hw/share/cit_ztyu/cz/nanochat/base_data/hf_datasets')
print(f'ARC-Challenge validation: {len(ds_challenge_validation)} rows')
ds_challenge_validation.save_to_disk('/public_hw/share/cit_ztyu/cz/nanochat/base_data/arc_challenge_validation')

print('ARC dataset downloaded!')
"

echo "=== Checking identity_conversations.jsonl ==="
if [ -f "/public_hw/share/cit_ztyu/cz/nanochat/identity_conversations.jsonl" ]; then
    echo "identity_conversations.jsonl exists!"
    wc -l /public_hw/share/cit_ztyu/cz/nanochat/identity_conversations.jsonl
else
    echo "WARNING: identity_conversations.jsonl NOT found!"
fi

echo ""
echo "=== Verifying all datasets ==="
python -c "
from datasets import load_from_disk
import os

datasets = [
    ('base_data/arc_easy_train', 'ARC-Easy train'),
    ('base_data/arc_challenge_train', 'ARC-Challenge train'),
    ('base_data/gsm8k_main_train', 'GSM8K train'),
    ('base_data/smoltalk_train', 'SmolTalk train'),
]

for ds_path, name in datasets:
    try:
        full_path = '/public_hw/share/cit_ztyu/cz/nanochat/' + ds_path
        ds = load_from_disk(full_path)
        print(f'{name} ({ds_path}): {len(ds)} rows')
    except Exception as e:
        print(f'{name} ({ds_path}): ERROR - {e}')
"