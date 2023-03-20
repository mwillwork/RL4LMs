#!/bin/sh

echo "Starting min reward 0.5 max prob threshold 0.5"
python scripts/training/train_text_generation.py --config_path scripts/training/task_configs/dialog/gpt2_ppo_minreward0.5_max_prob0.5.yml --experiment_name gpt2_ppo_minreward0.5_max_prob0.5

echo "Starting min reward 0.5 max prob threshold 0.75"
python scripts/training/train_text_generation.py --config_path scripts/training/task_configs/dialog/gpt2_ppo_minreward0.5_max_prob0.75.yml --experiment_name gpt2_ppo_minreward0.5_max_prob0.75 

echo "Starting min reward 0.0 max prob threshold 0.5"
python scripts/training/train_text_generation.py --config_path scripts/training/task_configs/dialog/gpt2_ppo_minreward0.0_max_prob0.5.yml --experiment_name gpt2_ppo_minreward0.0_max_prob0.5 

echo "Starting min reward 0.0 max prob threshold 0.75"
python scripts/training/train_text_generation.py --config_path scripts/training/task_configs/dialog/gpt2_ppo_minreward0.0_max_prob0.75.yml --experiment_name gpt2_ppo_minreward0.0_max_prob0.75 
