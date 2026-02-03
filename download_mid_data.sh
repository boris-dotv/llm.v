#!/bin/bash
cd /public_hw/share/cit_ztyu/cz/nanochat
source .venv/bin/activate

export HF_ENDPOINT=https://hf-mirror.com
export HF_DATASETS_CACHE=/public_hw/share/cit_ztyu/cz/nanochat/base_data/hf_datasets

python -c "
from datasets import load_dataset

# SmolTalk
ds1 = load_dataset('HuggingFaceTB/smol-smoltalk', split='train', cache_dir='/public_hw/share/cit_ztyu/cz/nanochat/base_data/hf_datasets')
print(f'SmolTalk train: {len(ds1)} rows')
ds1.save_to_disk('/public_hw/share/cit_ztyu/cz/nanochat/base_data/smoltalk_train')

ds2 = load_dataset('HuggingFaceTB/smol-smoltalk', split='test', cache_dir='/public_hw/share/cit_ztyu/cz/nanochat/base_data/hf_datasets')
print(f'SmolTalk test: {len(ds2)} rows')
ds2.save_to_disk('/public_hw/share/cit_ztyu/cz/nanochat/base_data/smoltalk_test')

# MMLU
ds3 = load_dataset('cais/mmlu', 'auxiliary_train', split='train', cache_dir='/public_hw/share/cit_ztyu/cz/nanochat/base_data/hf_datasets')
print(f'MMLU auxiliary_train: {len(ds3)} rows')
ds3.save_to_disk('/public_hw/share/cit_ztyu/cz/nanochat/base_data/mmlu_aux_train')

ds4 = load_dataset('cais/mmlu', 'all', split='test', cache_dir='/public_hw/share/cit_ztyu/cz/nanochat/base_data/hf_datasets')
print(f'MMLU test: {len(ds4)} rows[:1000]')
ds4.save_to_disk('/public_hw/share/cit_ztyu/cz/nanochat/base_data/mmlu_test')

# GSM8K
ds5 = load_dataset('openai/gsm8k', 'main', split='train', cache_dir='/public_hw/share/cit_ztyu/cz/nanochat/base_data/hf_datasets')
print(f'GSM8K train: {len(ds5)} rows')
ds5.save_to_disk('/public_hw/share/cit_ztyu/cz/nanochat/base_data/gsm8k_train')

ds6 = load_dataset('openai/gsm8k', 'main', split='test', cache_dir='/public_hw/share/cit_ztyu/cz/nanochat/base_data/hf_datasets')
print(f'GSM8K test: {len(ds6)} rows[:1000]')
ds6.save_to_disk('/public_hw/share/cit_ztyu/cz/nanochat/base_data/gsm8k_test')

print('Done!')
"