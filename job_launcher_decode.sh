#!/bin/sh

echo "Starting plus DECODE decode_weight=0.5"
python scripts/training/train_text_generation.py --config_path scripts/training/task_configs/dialog/gpt2_ppo_plus_decode0.5.yml --experiment_name gpt2_ppo_plus_decode0.5


echo "Starting plus DECODE decode_weight=0.25"
python scripts/training/train_text_generation.py --config_path scripts/training/task_configs/dialog/gpt2_ppo_plus_decode0.25.yml --experiment_name gpt2_ppo_plus_decode0.25

echo "Starting plus DECODE decode_weight=0.75"
python scripts/training/train_text_generation.py --config_path scripts/training/task_configs/dialog/gpt2_ppo_plus_decode0.75.yml --experiment_name gpt2_ppo_plus_decode0.75

