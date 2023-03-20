#!/bin/sh

echo "Starting baseline max-length=40"
python scripts/training/train_text_generation.py --config_path scripts/training/task_configs/dialog/gpt2_ppo_max_40.yml --experiment_name gpt2_ppo_max_40 

echo "Starting min-length=10 max-length=40"
python scripts/training/train_text_generation.py --config_path scripts/training/task_configs/dialog/gpt2_ppo_max_40_min_10.yml --experiment_name gpt2_ppo_max_40_min_10

echo "Starting topk4=40 max-length=40"
python scripts/training/train_text_generation.py --config_path scripts/training/task_configs/dialog/gpt2_ppo_max_40_topk_40.yml --experiment_name gpt2_ppo_max_40_topk_40

echo "Starting min-length=10 topk=40 max-length=40"
python scripts/training/train_text_generation.py --config_path scripts/training/task_configs/dialog/gpt2_ppo_max_40_topk_40_min_10.yml --experiment_name gpt2_ppo_max_40_topk_40_min_10


