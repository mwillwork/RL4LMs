#!/bin/sh

echo "Starting noise 0.25 factor 0.5"
python scripts/training/train_text_generation.py --config_path scripts/training/task_configs/dialog/gpt2_ppo_noisy0.25_factor0.5.yml --experiment_name gpt2_ppo_noisy0.25_factor0.5 

echo "Starting noise 0.25 factor 1.0"
python scripts/training/train_text_generation.py --config_path scripts/training/task_configs/dialog/gpt2_ppo_noisy0.25_factor1.0.yml --experiment_name gpt2_ppo_noisy0.25_factor1.0

echo "Starting noise 0.25 factor 2.0"
python scripts/training/train_text_generation.py --config_path scripts/training/task_configs/dialog/gpt2_ppo_noisy0.25_factor2.0.yml --experiment_name gpt2_ppo_noisy0.25_factor2.0

echo "Starting min length 10"
python scripts/training/train_text_generation.py --config_path scripts/training/task_configs/dialog/gpt2_ppo_min_10.yml --experiment_name gpt2_ppo_min_10

echo "Starting topk 40"
python scripts/training/train_text_generation.py --config_path scripts/training/task_configs/dialog/gpt2_ppo_topk_40.yml --experiment_name gpt2_ppo_topk_40

echo "Starting min length 10 topk 40"
python scripts/training/train_text_generation.py --config_path scripts/training/task_configs/dialog/gpt2_ppo_min_10_topk_40.yml --experiment_name gpt2_ppo_min_10_topk_40.yml


