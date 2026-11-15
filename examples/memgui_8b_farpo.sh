#!/bin/bash
# MemGUI-RL training recipe for Qwen3-VL-8B (MemGUI-8B-SFT -> MemGUI-8B-RL).
# Trains a ConAct policy on MemGUI-3K annotated states with GRPO, the matched scalar
# baseline (aggregate-then-normalise) or the reward-decoupled estimator used by FARPO.
#
# Usage:
#   # 1. separate training and validation files (recommended)
#   bash examples/memgui_8b_farpo.sh \
#       --train_data_path /path/to/train.json \
#       --val_data_path /path/to/val.json
#
#   # 1b. several training / validation JSON files (comma separated)
#   bash examples/memgui_8b_farpo.sh \
#       --train_data_path /path/to/train1.json,/path/to/train2.json \
#       --val_data_path /path/to/val1.json,/path/to/val2.json
#
#   # 2. reward-decoupled estimator (FARPO; the option keeps its historical name `gdpo`)
#   bash examples/memgui_8b_farpo.sh \
#       --train_data_path /path/to/train.json \
#       --val_data_path /path/to/val.json \
#       --algorithm gdpo
#
#   # 3. one file, split by task_id (legacy mode)
#   bash examples/memgui_8b_farpo.sh \
#       --train_data_path /path/to/data.json \
#       --train_task_count 100 \
#       --val_task_count 15
#
#   # 4. custom training options
#   bash examples/memgui_8b_farpo.sh \
#       --train_data_path /path/to/train.json \
#       --val_data_path /path/to/val.json \
#       --total_epochs 20 \
#       --val_freq 2 \
#       --val_generations_to_log 5 \
#       --train_generations_to_log 5
#
#   # 5. debugging with N trajectories (not N steps)
#   bash examples/memgui_8b_farpo.sh \
#       --train_data_path /path/to/train.json \
#       --val_data_path /path/to/val.json \
#       --train_max_trajectories 32 \
#       --val_max_trajectories 16
#
#   # 6. folding-aware sampling: span folds only, or span:step = 9:1
#   bash examples/memgui_8b_farpo.sh \
#       --train_data_path /path/to/train.json \
#       --val_data_path /path/to/val.json \
#       --train_folding_filter_mode span_only
#   bash examples/memgui_8b_farpo.sh \
#       --train_data_path /path/to/train.json \
#       --val_data_path /path/to/val.json \
#       --train_folding_filter_mode span_with_step_mix \
#       --span_step_mix_ratio 9:1

set -eo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Keep wandb lightweight: metrics are still logged, but code snapshots,
# model artifacts, and console streams are disabled unless the caller overrides.
export WANDB_DISABLE_CODE="${WANDB_DISABLE_CODE:-true}"
export WANDB_LOG_MODEL="${WANDB_LOG_MODEL:-false}"
export WANDB_CONSOLE="${WANDB_CONSOLE:-off}"

# defaults
MODEL_PATH="checkpoints/MemGUI-8B-SFT"   # https://huggingface.co/memgui-agent-anonymous/MemGUI-8B-SFT
TRAIN_DATA_PATH=""
VAL_DATA_PATH=""
DATA_LOADING_MODE=3  # 1: medium only, 2: medium+easy, 3: all (incl. unknown)
POSITIVE_ONLY=true
ALGORITHM="grpo"      # grpo | scalar_grpo_matched | gdpo (reward-decoupled estimator of FARPO)
EXPERIMENT_NAME=""    # derived from the algorithm when empty
PROJECT_NAME="memgui-rl"

# single-file split options (only when VAL_DATA_PATH is empty)
TRAIN_VAL_SPLIT=0.9
TRAIN_TASK_COUNT=-1   # training trajectories; -1 uses TRAIN_VAL_SPLIT
VAL_TASK_COUNT=-1     # validation trajectories; -1 uses TRAIN_VAL_SPLIT
SPLIT_SEED=42         # seed of the single-file split
EXPERIMENT_SEED=1     # default seed for driver, workers, rollout and DataLoader
FILTER_SEED=""        # defaults to EXPERIMENT_SEED
DATA_SEED=""          # defaults to EXPERIMENT_SEED
ROLLOUT_SEED=""       # defaults to EXPERIMENT_SEED
WORKER_SEED=""        # defaults to EXPERIMENT_SEED

# folding-aware sampling and debug subsets (also apply in separate-file mode)
TRAIN_FOLDING_FILTER_MODE="all"  # all | span_only | span_with_step_mix | valid_natural_matched
SPAN_STEP_MIX_RATIO="9:1"        # span folds : step folds (rho in the paper)
TRAIN_FOLDING_FILTER_OUTPUT_DIR="${TMPDIR:-/tmp}/memgui_verl_folding_filtered"
TRAIN_MAX_TRAJECTORIES=-1
TRAIN_SUBSET_OUTPUT_DIR="${TMPDIR:-/tmp}/memgui_verl_train_subsets"
VAL_MAX_TRAJECTORIES=-1
VAL_SUBSET_OUTPUT_DIR="${TMPDIR:-/tmp}/memgui_verl_val_subsets"

# training options
N_GPUS_PER_NODE=8
TOTAL_EPOCHS=4
MAX_STEPS="null"      # overrides total_epochs when set; the paper uses 100
VAL_FREQ=1
VAL_GENERATIONS_TO_LOG=1
TRAIN_GENERATIONS_TO_LOG=0
WANDB_LOG_GENERATIONS=false
GENERATION_LOG_MAX_CHARS=4000
SAVE_FREQ=5
SAVE_LIMIT=-1
FIND_LAST_CHECKPOINT=true

# Multimodal batches go through a multiprocessing queue in /dev/shm; with small container
# shared memory, prefetching workers die with SIGBUS, so data are loaded in the main process
# by default. Raise the worker counts (1/2/4) only when /dev/shm is large enough.
TRAIN_DATALOADER_NUM_WORKERS=0
VAL_DATALOADER_NUM_WORKERS=0
DATALOADER_PREFETCH_FACTOR=1
DATALOADER_PERSISTENT_WORKERS=false

# Reward weights w (overall reward, and the aggregation weights of both estimators)
# format: "format,action_type,action_params,folding"; default "0.1,0.4,0.4,0.1"
#   - format: output format (10%)
#   - action_type: action type (40%)
#   - action_params: action arguments (40%)
#   - folding: context-folding directive (10%)
REWARD_WEIGHTS="0.1,0.4,0.4,0.1"

# Whether the folding reward scores the summary text
# true : folding reward = 0.3 * range match + 0.7 * summary similarity
# false: folding reward = range match only (paper setting)
FOLDING_INCLUDE_SUMMARY=true

# Depth bonus for folds at least as deep as the annotation (IoU > 0.5); 0.0 disables it (paper setting)
FOLDING_DEPTH_BONUS=0.0

# Component keys of the estimators (the `gdpo_` option names are kept for compatibility).
# scalar_grpo_matched aggregates then normalises; gdpo (FARPO) standardises each component first.
GDPO_REWARD_KEYS="format,action_type,action_params,folding"

# command-line options
while [[ $# -gt 0 ]]; do
    case $1 in
        --train_data_path)
            TRAIN_DATA_PATH="$2"
            shift 2
            ;;
        --val_data_path)
            VAL_DATA_PATH="$2"
            shift 2
            ;;
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --data_loading_mode)
            DATA_LOADING_MODE="$2"
            shift 2
            ;;
        --train_val_split)
            TRAIN_VAL_SPLIT="$2"
            shift 2
            ;;
        --train_task_count)
            TRAIN_TASK_COUNT="$2"
            shift 2
            ;;
        --train_max_trajectories|--debug_train_trajectories)
            TRAIN_MAX_TRAJECTORIES="$2"
            shift 2
            ;;
        --train_subset_output_dir)
            TRAIN_SUBSET_OUTPUT_DIR="$2"
            shift 2
            ;;
        --train_folding_filter_mode|--folding_filter_mode)
            TRAIN_FOLDING_FILTER_MODE="$2"
            shift 2
            ;;
        --span_step_mix_ratio|--span_step_ratio)
            SPAN_STEP_MIX_RATIO="$2"
            shift 2
            ;;
        --train_folding_filter_output_dir)
            TRAIN_FOLDING_FILTER_OUTPUT_DIR="$2"
            shift 2
            ;;
        --val_max_trajectories|--debug_val_trajectories)
            VAL_MAX_TRAJECTORIES="$2"
            shift 2
            ;;
        --val_subset_output_dir)
            VAL_SUBSET_OUTPUT_DIR="$2"
            shift 2
            ;;
        --val_task_count)
            VAL_TASK_COUNT="$2"
            shift 2
            ;;
        --split_seed)
            SPLIT_SEED="$2"
            shift 2
            ;;
        --seed|--experiment_seed)
            EXPERIMENT_SEED="$2"
            shift 2
            ;;
        --filter_seed)
            FILTER_SEED="$2"
            shift 2
            ;;
        --data_seed)
            DATA_SEED="$2"
            shift 2
            ;;
        --rollout_seed)
            ROLLOUT_SEED="$2"
            shift 2
            ;;
        --worker_seed)
            WORKER_SEED="$2"
            shift 2
            ;;
        --positive_only)
            POSITIVE_ONLY="$2"
            shift 2
            ;;
        --algorithm)
            ALGORITHM="$2"
            shift 2
            ;;
        --gdpo_reward_keys|--component_reward_keys)
            GDPO_REWARD_KEYS="$2"
            shift 2
            ;;
        --project_name)
            PROJECT_NAME="$2"
            shift 2
            ;;
        --experiment_name)
            EXPERIMENT_NAME="$2"
            shift 2
            ;;
        --n_gpus_per_node)
            N_GPUS_PER_NODE="$2"
            shift 2
            ;;
        --total_epochs)
            TOTAL_EPOCHS="$2"
            shift 2
            ;;
        --max_steps)
            MAX_STEPS="$2"
            shift 2
            ;;
        --val_freq)
            VAL_FREQ="$2"
            shift 2
            ;;
        --val_generations_to_log)
            VAL_GENERATIONS_TO_LOG="$2"
            shift 2
            ;;
        --train_generations_to_log)
            TRAIN_GENERATIONS_TO_LOG="$2"
            shift 2
            ;;
        --wandb_log_generations)
            WANDB_LOG_GENERATIONS="$2"
            shift 2
            ;;
        --generation_log_max_chars)
            GENERATION_LOG_MAX_CHARS="$2"
            shift 2
            ;;
        --save_freq)
            SAVE_FREQ="$2"
            shift 2
            ;;
        --save_limit)
            SAVE_LIMIT="$2"
            shift 2
            ;;
        --find_last_checkpoint)
            FIND_LAST_CHECKPOINT="$2"
            shift 2
            ;;
        --train_dataloader_num_workers|--train_num_workers)
            TRAIN_DATALOADER_NUM_WORKERS="$2"
            shift 2
            ;;
        --val_dataloader_num_workers|--val_num_workers)
            VAL_DATALOADER_NUM_WORKERS="$2"
            shift 2
            ;;
        --dataloader_prefetch_factor)
            DATALOADER_PREFETCH_FACTOR="$2"
            shift 2
            ;;
        --dataloader_persistent_workers)
            DATALOADER_PERSISTENT_WORKERS="$2"
            shift 2
            ;;
        --reward_weights)
            # reward weights, format "format,action_type,action_params,folding"
            # e.g. --reward_weights "0.1,0.4,0.4,0.1"
            REWARD_WEIGHTS="$2"
            shift 2
            ;;
        --folding_include_summary)
            FOLDING_INCLUDE_SUMMARY="$2"
            shift 2
            ;;
        --folding_depth_bonus)
            FOLDING_DEPTH_BONUS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

FILTER_SEED="${FILTER_SEED:-$EXPERIMENT_SEED}"
DATA_SEED="${DATA_SEED:-$EXPERIMENT_SEED}"
ROLLOUT_SEED="${ROLLOUT_SEED:-$EXPERIMENT_SEED}"
WORKER_SEED="${WORKER_SEED:-$EXPERIMENT_SEED}"

for seed_value in "$EXPERIMENT_SEED" "$FILTER_SEED" "$DATA_SEED" "$ROLLOUT_SEED" "$WORKER_SEED" "$SPLIT_SEED"; do
    if ! [[ "$seed_value" =~ ^[0-9]+$ ]]; then
        echo "Error: all seed values must be non-negative integers, got: $seed_value"
        exit 1
    fi
done

if [[ "$MAX_STEPS" != "null" ]] && ! [[ "$MAX_STEPS" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: --max_steps must be a positive integer or null"
    exit 1
fi
if ! [[ "$SAVE_FREQ" =~ ^-?[0-9]+$ ]]; then
    echo "Error: --save_freq must be an integer"
    exit 1
fi
if ! [[ "$SAVE_LIMIT" =~ ^-?[0-9]+$ ]]; then
    echo "Error: --save_limit must be an integer"
    exit 1
fi
if [[ "$FIND_LAST_CHECKPOINT" != "true" && "$FIND_LAST_CHECKPOINT" != "false" ]]; then
    echo "Error: --find_last_checkpoint must be true or false"
    exit 1
fi

if ! [[ "$TRAIN_DATALOADER_NUM_WORKERS" =~ ^[0-9]+$ ]]; then
    echo "Error: --train_dataloader_num_workers must be a non-negative integer"
    exit 1
fi
if ! [[ "$VAL_DATALOADER_NUM_WORKERS" =~ ^[0-9]+$ ]]; then
    echo "Error: --val_dataloader_num_workers must be a non-negative integer"
    exit 1
fi
if ! [[ "$DATALOADER_PREFETCH_FACTOR" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: --dataloader_prefetch_factor must be a positive integer"
    exit 1
fi
if [[ "$DATALOADER_PERSISTENT_WORKERS" != "true" && "$DATALOADER_PERSISTENT_WORKERS" != "false" ]]; then
    echo "Error: --dataloader_persistent_workers must be true or false"
    exit 1
fi

# reward weights
IFS=',' read -ra REWARD_WEIGHTS_ARRAY <<< "$REWARD_WEIGHTS"
if [ ${#REWARD_WEIGHTS_ARRAY[@]} -ne 4 ]; then
    echo "Error: --reward_weights format should be 'format,action_type,action_params,folding'"
    exit 1
fi
REWARD_FORMAT_WEIGHT=${REWARD_WEIGHTS_ARRAY[0]}
REWARD_ACTION_TYPE_WEIGHT=${REWARD_WEIGHTS_ARRAY[1]}
REWARD_ACTION_PARAMS_WEIGHT=${REWARD_WEIGHTS_ARRAY[2]}
REWARD_FOLDING_WEIGHT=${REWARD_WEIGHTS_ARRAY[3]}

# algorithm
ALGORITHM=$(echo "$ALGORITHM" | tr '[:upper:]' '[:lower:]')
if [[ "$ALGORITHM" != "grpo" && "$ALGORITHM" != "scalar_grpo_matched" && "$ALGORITHM" != "gdpo" ]]; then
    echo "Error: --algorithm must be grpo, scalar_grpo_matched, or gdpo; got: $ALGORITHM"
    exit 1
fi

# experiment name derived from the algorithm when not given
if [ -z "$EXPERIMENT_NAME" ]; then
    EXPERIMENT_NAME="memgui_8b_${ALGORITHM}"
fi

validate_data_paths() {
    local label="$1"
    local paths="$2"
    local count=0
    local data_path

    IFS=',' read -ra DATA_PATH_ARRAY <<< "$paths"
    for data_path in "${DATA_PATH_ARRAY[@]}"; do
        data_path="${data_path#"${data_path%%[![:space:]]*}"}"
        data_path="${data_path%"${data_path##*[![:space:]]}"}"
        if [ -z "$data_path" ]; then
            continue
        fi
        if [ ! -f "$data_path" ]; then
            echo "Error: ${label} data file not found: $data_path"
            exit 1
        fi
        count=$((count + 1))
    done

    if [ "$count" -eq 0 ]; then
        echo "Error: ${label} data path is empty"
        exit 1
    fi
}

# required options
if [ -z "$TRAIN_DATA_PATH" ]; then
    echo "Error: --train_data_path is required"
    echo "Usage: bash $0 --train_data_path /path/to/train1.json,/path/to/train2.json --val_data_path /path/to/val1.json,/path/to/val2.json [--algorithm grpo|scalar_grpo_matched|gdpo]"
    exit 1
fi

validate_data_paths "Train" "$TRAIN_DATA_PATH"
ACTUAL_TRAIN_PATH="$TRAIN_DATA_PATH"

# validation data: separate file when --val_data_path is given, otherwise a task_id split
# of the training file (train_files == val_files)
if [ -n "$VAL_DATA_PATH" ]; then
    validate_data_paths "Val" "$VAL_DATA_PATH"
    ACTUAL_VAL_PATH="$VAL_DATA_PATH"
    SPLIT_MODE="separate-files"
else
    ACTUAL_VAL_PATH="$TRAIN_DATA_PATH"
    SPLIT_MODE="single-file-split"
fi

if [ "$TRAIN_FOLDING_FILTER_MODE" != "all" ]; then
    FOLD_FILTER_SCRIPT="${REPO_ROOT}/scripts/filter_mobilese_by_folding.py"
    if [ ! -f "$FOLD_FILTER_SCRIPT" ]; then
        echo "Error: folding filter script not found: $FOLD_FILTER_SCRIPT"
        exit 1
    fi
    mkdir -p "$TRAIN_FOLDING_FILTER_OUTPUT_DIR"
    FILTER_INPUT_HASH=$(printf "%s" "$ACTUAL_TRAIN_PATH" | sha1sum | awk '{print substr($1,1,10)}')
    FILTER_RATIO_SAFE=$(printf "%s" "$SPAN_STEP_MIX_RATIO" | tr ':,/' '___')
    TRAIN_FOLDING_FILTERED_PATH="${TRAIN_FOLDING_FILTER_OUTPUT_DIR}/${EXPERIMENT_NAME}_train_${TRAIN_FOLDING_FILTER_MODE}_ratio${FILTER_RATIO_SAFE}_seed${FILTER_SEED}_${FILTER_INPUT_HASH}.json"
    python3 "$FOLD_FILTER_SCRIPT" \
        --input "$ACTUAL_TRAIN_PATH" \
        --output "$TRAIN_FOLDING_FILTERED_PATH" \
        --mode "$TRAIN_FOLDING_FILTER_MODE" \
        --span-step-ratio "$SPAN_STEP_MIX_RATIO" \
        --seed "$FILTER_SEED" \
        --positive-only "$POSITIVE_ONLY"
    ACTUAL_TRAIN_PATH="$TRAIN_FOLDING_FILTERED_PATH"
fi

if [ "$TRAIN_MAX_TRAJECTORIES" -gt 0 ] 2>/dev/null; then
    SUBSET_SCRIPT="${REPO_ROOT}/scripts/subset_mobilese_by_trajectory.py"
    if [ ! -f "$SUBSET_SCRIPT" ]; then
        echo "Error: subset script not found: $SUBSET_SCRIPT"
        exit 1
    fi
    mkdir -p "$TRAIN_SUBSET_OUTPUT_DIR"
    TRAIN_PATH_HASH=$(printf "%s" "$ACTUAL_TRAIN_PATH" | sha1sum | awk '{print substr($1,1,10)}')
    TRAIN_SUBSET_PATH="${TRAIN_SUBSET_OUTPUT_DIR}/${EXPERIMENT_NAME}_train_traj${TRAIN_MAX_TRAJECTORIES}_seed${FILTER_SEED}_${TRAIN_PATH_HASH}.json"
    python3 "$SUBSET_SCRIPT" \
        --input "$ACTUAL_TRAIN_PATH" \
        --output "$TRAIN_SUBSET_PATH" \
        --max-trajectories "$TRAIN_MAX_TRAJECTORIES" \
        --seed "$FILTER_SEED"
    ACTUAL_TRAIN_PATH="$TRAIN_SUBSET_PATH"
fi

if [ "$VAL_MAX_TRAJECTORIES" -gt 0 ] 2>/dev/null; then
    SUBSET_SCRIPT="${REPO_ROOT}/scripts/subset_mobilese_by_trajectory.py"
    if [ ! -f "$SUBSET_SCRIPT" ]; then
        echo "Error: subset script not found: $SUBSET_SCRIPT"
        exit 1
    fi
    mkdir -p "$VAL_SUBSET_OUTPUT_DIR"
    VAL_PATH_HASH=$(printf "%s" "$ACTUAL_VAL_PATH" | sha1sum | awk '{print substr($1,1,10)}')
    VAL_SUBSET_PATH="${VAL_SUBSET_OUTPUT_DIR}/${EXPERIMENT_NAME}_val_traj${VAL_MAX_TRAJECTORIES}_seed${FILTER_SEED}_${VAL_PATH_HASH}.json"
    python3 "$SUBSET_SCRIPT" \
        --input "$ACTUAL_VAL_PATH" \
        --output "$VAL_SUBSET_PATH" \
        --max-trajectories "$VAL_MAX_TRAJECTORIES" \
        --seed "$FILTER_SEED"
    ACTUAL_VAL_PATH="$VAL_SUBSET_PATH"
fi

echo "============================================"
echo "MemGUI Training - Qwen3-VL-8B ${ALGORITHM^^}"
echo "============================================"
echo "Model: $MODEL_PATH"
echo "Train Data: $TRAIN_DATA_PATH"
if [ "$TRAIN_FOLDING_FILTER_MODE" != "all" ]; then
    echo "Train Folding Filter: $TRAIN_FOLDING_FILTER_MODE"
    echo "Span:Step Mix Ratio: $SPAN_STEP_MIX_RATIO"
fi
if [ "$ACTUAL_TRAIN_PATH" != "$TRAIN_DATA_PATH" ]; then
    echo "Effective Train Data: $ACTUAL_TRAIN_PATH"
    echo "Train Trajectory Limit: $TRAIN_MAX_TRAJECTORIES"
fi
echo "Val Data: $ACTUAL_VAL_PATH"
if [ "$VAL_MAX_TRAJECTORIES" -gt 0 ] 2>/dev/null; then
    echo "Val Trajectory Limit: $VAL_MAX_TRAJECTORIES"
fi
echo "Split Mode: $SPLIT_MODE"
echo "Algorithm: ${ALGORITHM^^}"
if [[ "$ALGORITHM" == "gdpo" || "$ALGORITHM" == "scalar_grpo_matched" ]]; then
    echo "  Component Reward Keys: $GDPO_REWARD_KEYS"
    echo "  Component Reward Weights: $REWARD_WEIGHTS"
fi
echo "Data Loading Mode: $DATA_LOADING_MODE (1:medium, 2:medium+easy, 3:all)"
echo "Positive Only: $POSITIVE_ONLY"
if [ "$SPLIT_MODE" == "single-file-split" ]; then
    if [ "$TRAIN_TASK_COUNT" -gt 0 ] 2>/dev/null || [ "$VAL_TASK_COUNT" -gt 0 ] 2>/dev/null; then
        echo "Train/Val Split: by count (train=${TRAIN_TASK_COUNT}, val=${VAL_TASK_COUNT} trajectories)"
    else
        echo "Train/Val Split: by fraction ($TRAIN_VAL_SPLIT)"
    fi
    echo "Split Seed: $SPLIT_SEED"
fi
echo "Experiment: $EXPERIMENT_NAME"
echo "Project: $PROJECT_NAME"
echo "-- Reward / Estimator Weights --"
echo "  format: $REWARD_FORMAT_WEIGHT"
echo "  action_type: $REWARD_ACTION_TYPE_WEIGHT"
echo "  action_params: $REWARD_ACTION_PARAMS_WEIGHT"
echo "  folding: $REWARD_FOLDING_WEIGHT"
echo "  folding_include_summary: $FOLDING_INCLUDE_SUMMARY"
echo "  folding_depth_bonus: $FOLDING_DEPTH_BONUS"
if [[ "$ALGORITHM" == "gdpo" || "$ALGORITHM" == "scalar_grpo_matched" ]]; then
    echo "  Component Reward Keys: $GDPO_REWARD_KEYS"
    echo "  Component Reward Weights: $REWARD_WEIGHTS"
fi
echo "-- Training Config --"
echo "  GPUs per node: $N_GPUS_PER_NODE"
echo "  Total epochs: $TOTAL_EPOCHS"
echo "  Max steps: $MAX_STEPS"
echo "  Val freq: $VAL_FREQ"
echo "  Val generations to log: $VAL_GENERATIONS_TO_LOG"
echo "  Train generations to log: $TRAIN_GENERATIONS_TO_LOG"
echo "  WandB generation table upload: $WANDB_LOG_GENERATIONS"
echo "  Generation log max chars: $GENERATION_LOG_MAX_CHARS"
echo "  Experiment/driver seed: $EXPERIMENT_SEED"
echo "  Filter seed: $FILTER_SEED"
echo "  DataLoader seed: $DATA_SEED"
echo "  Rollout seed: $ROLLOUT_SEED"
echo "  Worker seed: $WORKER_SEED"
echo "  Save freq / limit: $SAVE_FREQ / $SAVE_LIMIT"
echo "  Find last checkpoint: $FIND_LAST_CHECKPOINT"
echo "  Train DataLoader workers: $TRAIN_DATALOADER_NUM_WORKERS"
echo "  Val DataLoader workers: $VAL_DATALOADER_NUM_WORKERS"
echo "  DataLoader prefetch factor: $DATALOADER_PREFETCH_FACTOR"
echo "  DataLoader persistent workers: $DATALOADER_PERSISTENT_WORKERS"
echo "============================================"

# ConAct verifier and prompt templates
REWARD_FUNCTION="examples/reward_function/r1gui_memgui.py:compute_score"
SYSTEM_PROMPT="examples/format_prompt/r1gui_memgui_system.jinja"
FORMAT_PROMPT="examples/format_prompt/r1gui_memgui_user.jinja"

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files="${ACTUAL_TRAIN_PATH}" \
    data.val_files="${ACTUAL_VAL_PATH}" \
    data.use_mobilese_format=true \
    data.val_use_mobilese_format=true \
    data.data_loading_mode=${DATA_LOADING_MODE} \
    data.train_val_split=${TRAIN_VAL_SPLIT} \
    data.train_task_count=${TRAIN_TASK_COUNT} \
    data.val_task_count=${VAL_TASK_COUNT} \
    data.split_seed=${SPLIT_SEED} \
    data.seed=${DATA_SEED} \
    data.positive_only=${POSITIVE_ONLY} \
    data.train_dataloader_num_workers=${TRAIN_DATALOADER_NUM_WORKERS} \
    data.val_dataloader_num_workers=${VAL_DATALOADER_NUM_WORKERS} \
    data.dataloader_prefetch_factor=${DATALOADER_PREFETCH_FACTOR} \
    data.dataloader_persistent_workers=${DATALOADER_PERSISTENT_WORKERS} \
    data.system_prompt=${SYSTEM_PROMPT} \
    data.format_prompt=${FORMAT_PROMPT} \
    algorithm.adv_estimator=${ALGORITHM} \
    algorithm.gdpo_reward_keys="${GDPO_REWARD_KEYS}" \
    algorithm.gdpo_reward_weights="${REWARD_WEIGHTS}" \
    worker.seed=${WORKER_SEED} \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.tensor_parallel_size=1 \
    worker.rollout.enable_chunked_prefill=false \
    worker.rollout.seed=${ROLLOUT_SEED} \
    worker.reward.reward_function=${REWARD_FUNCTION} \
    worker.reward.reward_function_kwargs.format_weight=${REWARD_FORMAT_WEIGHT} \
    worker.reward.reward_function_kwargs.action_type_weight=${REWARD_ACTION_TYPE_WEIGHT} \
    worker.reward.reward_function_kwargs.action_params_weight=${REWARD_ACTION_PARAMS_WEIGHT} \
    worker.reward.reward_function_kwargs.folding_weight=${REWARD_FOLDING_WEIGHT} \
    worker.reward.reward_function_kwargs.folding_include_summary=${FOLDING_INCLUDE_SUMMARY} \
    worker.reward.reward_function_kwargs.folding_depth_bonus=${FOLDING_DEPTH_BONUS} \
    trainer.seed=${EXPERIMENT_SEED} \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.max_steps=${MAX_STEPS} \
    trainer.val_freq=${VAL_FREQ} \
    trainer.val_generations_to_log=${VAL_GENERATIONS_TO_LOG} \
    trainer.train_generations_to_log=${TRAIN_GENERATIONS_TO_LOG} \
    trainer.wandb_log_generations=${WANDB_LOG_GENERATIONS} \
    trainer.generation_log_max_chars=${GENERATION_LOG_MAX_CHARS} \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.save_limit=${SAVE_LIMIT} \
    trainer.find_last_checkpoint=${FIND_LAST_CHECKPOINT} \
    data.max_pixels=1258291 \
    data.max_prompt_length=16384 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=false \
    data.rollout_batch_size=128 \
    data.val_batch_size=13
