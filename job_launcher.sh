#!/bin/sh
echo "Starting noise 0.75 factor 1.0"
python scripts/training/train_text_generation.py --config_path scripts/training/task_configs/dialog/gpt2_ppo_noisy0.75_factor1.0.yml --experiment_name gpt2_ppo_noisy0.75_factor1.0

echo "Starting noise 0.75 factor 2.0"
python scripts/training/train_text_generation.py --config_path scripts/training/task_configs/dialog/gpt2_ppo_noisy0.75_factor2.0.yml --experiment_name gpt2_ppo_noisy0.75_factor2.0

echo "Starting noise 0.5 factor 0.5"
python scripts/training/train_text_generation.py --config_path scripts/training/task_configs/dialog/gpt2_ppo_noisy0.5_factor0.5.yml --experiment_name gpt2_ppo_noisy0.5_factor0.5

echo "Starting noise 0.5 factor 1.0"
python scripts/training/train_text_generation.py --config_path scripts/training/task_configs/dialog/gpt2_ppo_noisy0.5_factor1.0.yml --experiment_name gpt2_ppo_noisy0.5_factor1.0

echo "Starting noise 0.5 factor 2.0"
python scripts/training/train_text_generation.py --config_path scripts/training/task_configs/dialog/gpt2_ppo_noisy0.5_factor2.0.yml --experiment_name gpt2_ppo_noisy0.5_factor2.0

echo "Starting noise 1.0 factor 0.5"
python scripts/training/train_text_generation.py --config_path scripts/training/task_configs/dialog/gpt2_ppo_noisy1.0_factor0.5.yml --experiment_name gpt2_ppo_noisy1.0_factor0.5

echo "Starting noise 1.0 factor 1.0"
python scripts/training/train_text_generation.py --config_path scripts/training/task_configs/dialog/gpt2_ppo_noisy1.0_factor1.0.yml --experiment_name gpt2_ppo_noisy1.0_factor1.0

echo "Starting noise 1.0 factor 2.0"
python scripts/training/train_text_generation.py --config_path scripts/training/task_configs/dialog/gpt2_ppo_noisy1.0_factor2.0.yml --experiment_name gpt2_ppo_noisy1.0_factor2.0

