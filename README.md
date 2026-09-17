# MemGUI-RL: Reinforcement Learning for Proactive Context Management in Long-Horizon Mobile GUI Agents

Anonymous code release for the ICLR 2027 submission *MemGUI-RL*. The repository contains the
training code of **FARPO** (Folding-Aware Reward-decoupled Policy Optimization), the RL
recipe that turns [MemGUI-8B-SFT](https://huggingface.co/memgui-agent-anonymous/MemGUI-8B-SFT)
into **MemGUI-8B-RL**, together with the ConAct verifier, the folding-aware sampler and the
launch scripts of every training run reported in the paper.

* Project page: https://memgui-rl-anonymous.github.io/
* Model: https://huggingface.co/memgui-rl-anonymous/MemGUI-8B-RL
* Training data: https://huggingface.co/datasets/memgui-rl-anonymous/MemGUI-3K-Verl
* Evaluation logs: https://huggingface.co/datasets/memgui-rl-anonymous/MemGUI-RL-Eval

The trainer is a fork of [EasyR1](https://github.com/hiyouga/EasyR1) / [verl](https://github.com/volcengine/verl)
(Apache-2.0). Everything specific to this paper lives in the files listed below.

## What FARPO changes

A ConAct policy emits, in one response, a UI action *and* the context actions that manage
its own memory (a history fold `range` + `summary`, or a memory operation). The verifier
returns four rewards per response (format, action type, action parameters, folding). Applied
naively, GRPO (i) sums the components before normalising, so the component with the largest
within-group spread dominates the advantage, and (ii) samples annotated states at their
natural frequency, where span-level folds are only 22.7% of the annotated folds, so the
policy collapses to step-only folding. FARPO

1. **standardises each reward component within the group before weighting**
   (`--algorithm gdpo`, `verl/trainer/core_algos.py::compute_gdpo_outcome_advantage`), and
2. **controls the span-to-step exposure of the training prompts**
   (`--train_folding_filter_mode span_with_step_mix --span_step_mix_ratio 9:1`,
   `scripts/filter_mobilese_by_folding.py`).

Both mechanisms are ablated by the four launchers in `examples/farpo_ablation/`.

## Repository layout

| path | content |
| --- | --- |
| `examples/memgui_8b_farpo.sh` | main recipe: data filtering, estimator choice, reward weights, all hyper-parameters |
| `examples/farpo_ablation/run_e{0,1}f{0,1}_*.sh` | the 2x2 estimator x sampling ablation (`e1f1` = full FARPO = MemGUI-8B-RL) |
| `examples/reward_function/r1gui_memgui.py` | ConAct verifier (four reward components), see `docs/memgui_reward_function.md` |
| `examples/format_prompt/r1gui_memgui_{system,user}.jinja` | ConAct prompt templates |
| `scripts/filter_mobilese_by_folding.py` | folding-aware sampling (`span_only`, `span_with_step_mix`, `valid_natural_matched`) |
| `scripts/subset_mobilese_by_trajectory.py` | trajectory-level subsets for debugging |
| `scripts/model_merger.py` | merges FSDP checkpoints into a Hugging Face model directory |
| `verl/trainer/core_algos.py` | advantage estimators: `grpo`, `scalar_grpo_matched` (aggregate-then-normalise), `gdpo` (reward-decoupled) |
| `verl/trainer/ablation_metrics.py`, `verl/trainer/memgui3k_metrics.py` | logged diagnostics (zero-variance groups, deep-fold rate, offline memory metrics) |
| `verl/utils/dataset.py` | loader for the MemGUI-3K-Verl conversation format (`MobileSERLHFDataset`) |
| `tests/` | unit tests of the estimators, the sampler and the data pipeline |

The data format is called `mobilese` in option names and class names for historical
reasons; it is the MemGUI-3K conversation format described below.

## Installation

```bash
pip install -e .            # or use the EasyR1 docker image: hiyouga/verl:ngc-th2.6.0-cu120-vllm0.8.2
```

The reported runs use 8 GPUs (80 GB), vLLM rollouts, FSDP training and bf16.

## Data

1. Download [MemGUI-3K](https://huggingface.co/datasets/memgui-agent-anonymous/MemGUI-3K)
   (screenshots) and [MemGUI-3K-Verl](https://huggingface.co/datasets/memgui-rl-anonymous/MemGUI-3K-Verl)
   (annotated states in conversation format: `memgui3k_train_verl.json`, 73,807 states of
   2,661 trajectories; `memgui3k_test_verl.json`, 8,296 states of 295 held-out trajectories).
2. Image references in the JSON files are relative paths of the form `MemGUI-3K/images/<file>.png`.
   Point the loader at the directory that contains `MemGUI-3K/`:

```bash
export MEMGUI_IMAGE_ROOT=/path/to/data      # /path/to/data/MemGUI-3K/images/... must exist
export MEMGUI3K_VERL_DIR=/path/to/data/MemGUI-3K-Verl
```

Only positive (annotated-correct) states are used as RL prompts (`--positive_only true`,
default). The folding-aware sampler keeps every span fold and draws `m_rho =
min(N_step, ceil(N_span / rho))` step folds without replacement (fixed seed); `rho = 9`
in the final model. `valid_natural_matched` keeps the natural span:step ratio at the same
corpus size and is the control used in the paper.

## Training

Initialise from MemGUI-8B-SFT and run the full recipe (100 optimizer steps, G = 5,
temperature 1.0, 128 prompts per rollout batch, AdamW 1e-6, one PPO epoch, dual-clip PPO
with eps_lo = 0.2, eps_hi = 0.3, c = 3, KL beta = 0.01, w = (0.1, 0.4, 0.4, 0.1)):

```bash
export MEMGUI_SFT_MODEL=/path/to/MemGUI-8B-SFT
bash examples/farpo_ablation/run_e1f1_farpo_full.sh        # FARPO  -> MemGUI-8B-RL
bash examples/farpo_ablation/run_e1f0_normagg_natural.sh   # reward decoupling, natural sampling
bash examples/farpo_ablation/run_e0f1_scalar_foldaware.sh  # scalar estimator, folding-aware sampling
bash examples/farpo_ablation/run_e0f0_scalar_natural.sh    # GRPO recipe (scalar, natural)
```

Other span-to-step ratios of the development sweep:

```bash
bash examples/memgui_8b_farpo.sh --train_data_path $MEMGUI3K_VERL_DIR/memgui3k_train_verl.json \
  --val_data_path $MEMGUI3K_VERL_DIR/memgui3k_test_verl.json --model_path $MEMGUI_SFT_MODEL \
  --algorithm gdpo --train_folding_filter_mode span_with_step_mix --span_step_mix_ratio 4:1 \
  --total_epochs 1 --reward_weights "0.1,0.4,0.4,0.1" --folding_include_summary false --folding_depth_bonus 0.0
```

`--train_folding_filter_mode span_only` trains on span folds only; `--train_folding_filter_mode all`
uses the natural corpus. Checkpoints are written to `checkpoints/<project>/<experiment>/`;
merge the last FSDP checkpoint into a Hugging Face directory with

```bash
python3 scripts/model_merger.py --local_dir checkpoints/memgui-rl/<experiment>/global_step_100/actor
```

Weights & Biases is used for metrics only (`WANDB_MODE=offline` disables the login).
Validation runs greedy decoding on the 5,941 held-out states every 20 steps and logs the
offline memory metrics reported in the paper (memory-trigger F1, fold-range accuracy,
deep-fold rate, UI-type / match accuracy).

## Evaluation

Benchmark evaluation uses the public harnesses of
[MemGUI-Bench](https://arxiv.org/abs/2602.06075) (128 tasks, Pass@k and IRR) and
[MobileWorld](https://arxiv.org/abs/2512.19432) (117 GUI-only tasks) with the ConAct
prompt of MemGUI-Agent unchanged. The stored trajectories and judge outputs of every run in
the paper are released in the evaluation-log dataset linked above.

## Tests

```bash
pytest tests/test_core_algos_ablation.py tests/test_folding_filter.py tests/test_ablation_metrics.py
python3 examples/reward_function/r1gui_memgui.py
```

## License

Apache-2.0, inherited from EasyR1 / verl. See `LICENSE`.
