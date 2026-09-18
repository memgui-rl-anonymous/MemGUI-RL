#!/usr/bin/env bash
# FARPO ablation cell E1F0: reward-decoupled estimator, natural (size-matched) sampling.
# All four cells share the SFT initialisation, the verifier, 100 optimizer steps, G = 5 and the
# reward weights (0.1, 0.4, 0.4, 0.1); they differ only in the advantage estimator and in the
# span-to-step exposure of the training prompts (Section 5.3 of the paper).
set -e

# 1. Weights & Biases (metrics only; set WANDB_MODE=offline to skip the login)
if [ "${WANDB_MODE:-online}" != "offline" ]; then
  : "${WANDB_API_KEY:?set WANDB_API_KEY or export WANDB_MODE=offline}"
  python3 -c 'import os, wandb; wandb.login(key=os.environ["WANDB_API_KEY"])'
fi

# 2. Run from the repository root
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

# 3. Paths: MemGUI-3K-Verl JSON files (https://huggingface.co/datasets/memgui-rl/MemGUI-3K-Verl)
#    and the SFT initialisation (https://huggingface.co/memgui-agent-anonymous/MemGUI-8B-SFT)
DATA_DIR="${MEMGUI3K_VERL_DIR:-data/MemGUI-3K-Verl}"
SFT_MODEL="${MEMGUI_SFT_MODEL:-checkpoints/MemGUI-8B-SFT}"

EXPERIMENT_NAME=qwen3_vl_8b_farpo_ablation_e1f0_normagg_natural_seed1
CHECKPOINT_DIR="checkpoints/memgui-rl/${EXPERIMENT_NAME}"
if [ -e "$CHECKPOINT_DIR" ]; then
  echo "error: checkpoint directory already exists: $CHECKPOINT_DIR"
  echo "choose another EXPERIMENT_NAME or resume the old run manually."
  exit 1
fi

bash examples/memgui_8b_farpo.sh \
  --train_data_path "$DATA_DIR/memgui3k_train_verl.json" \
  --train_folding_filter_mode valid_natural_matched \
  --span_step_mix_ratio 9:1 \
  --val_data_path "$DATA_DIR/memgui3k_test_verl.json" \
  --model_path "$SFT_MODEL" \
  --algorithm gdpo \
  --component_reward_keys "format,action_type,action_params,folding" \
  --n_gpus_per_node 8 \
  --total_epochs 1 \
  --max_steps 100 \
  --val_freq 20 \
  --val_generations_to_log 0 \
  --train_generations_to_log 0 \
  --reward_weights "0.1,0.4,0.4,0.1" \
  --folding_include_summary false \
  --folding_depth_bonus 0.0 \
  --wandb_log_generations false \
  --train_dataloader_num_workers 0 \
  --val_dataloader_num_workers 0 \
  --dataloader_prefetch_factor 1 \
  --dataloader_persistent_workers false \
  --seed 1 \
  --split_seed 1 \
  --filter_seed 1 \
  --data_seed 1 \
  --rollout_seed 1 \
  --worker_seed 1 \
  --save_freq 20 \
  --save_limit 2 \
  --find_last_checkpoint false \
  --project_name memgui-rl \
  --experiment_name "$EXPERIMENT_NAME"
