# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
PPO config
"""

import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Optional, Tuple

from ..utils.py_functional import get_abs_path
from ..workers.config import WorkerConfig


def recursive_post_init(dataclass_obj):
    if hasattr(dataclass_obj, "post_init"):
        dataclass_obj.post_init()

    for attr in fields(dataclass_obj):
        if is_dataclass(getattr(dataclass_obj, attr.name)):
            recursive_post_init(getattr(dataclass_obj, attr.name))


@dataclass
class DataConfig:
    train_files: str = ""
    val_files: str = ""
    prompt_key: str = "prompt"
    answer_key: str = "answer"
    image_key: str = "images"
    video_key: str = "videos"
    image_dir: Optional[str] = None
    video_fps: float = 2.0
    max_prompt_length: int = 512
    max_response_length: int = 512
    rollout_batch_size: int = 512
    mini_rollout_batch_size: Optional[int] = None
    val_batch_size: int = -1
    format_prompt: Optional[str] = None
    system_prompt: Optional[str] = None
    """system prompt file path (jinja2 template or plain text), will be added as system role message"""
    override_chat_template: Optional[str] = None
    shuffle: bool = True
    seed: int = 1
    min_pixels: Optional[int] = 262144
    max_pixels: Optional[int] = 4194304
    filter_overlong_prompts: bool = True
    filter_overlong_prompts_workers: int = 16
    train_dataloader_num_workers: int = 8
    """Training DataLoader workers; 0 loads in the main process (avoids SIGBUS on small /dev/shm)."""
    val_dataloader_num_workers: int = 8
    """Validation DataLoader workers; 0 loads in the main process."""
    dataloader_prefetch_factor: int = 2
    """Batches prefetched per DataLoader worker; ignored when num_workers=0."""
    dataloader_persistent_workers: bool = False
    """Keep DataLoader workers alive across epochs; ignored when num_workers=0."""

    # MemGUI-3K (MobileSE format) options
    use_mobilese_format: bool = False
    """Use the MemGUI-3K conversation format (train/val split by task_id when a single file is given)."""
    data_loading_mode: int = 3
    """Difficulty filter: 1 = medium only, 2 = medium + easy, 3 = all."""
    train_val_split: float = 0.9
    """Train fraction of the task_id split, used when train_task_count and val_task_count are both -1."""
    train_task_count: int = -1
    """Number of training trajectories; -1 uses train_val_split."""
    val_task_count: int = -1
    """Number of validation trajectories; -1 uses train_val_split."""
    split_seed: int = 42
    """Seed of the train/val split (identical splits across runs)."""
    positive_only: bool = True
    """Keep only positive (annotated-correct) samples."""

    # mixed-format options (MemGUI-3K training set + GUI-R1 validation set)
    val_image_dir: Optional[str] = None
    """Validation image directory (GUI-R1 format)."""
    val_use_mobilese_format: bool = True
    """Validation set in MemGUI-3K format; False uses the GUI-R1 format."""

    def post_init(self):
        if self.train_dataloader_num_workers < 0:
            raise ValueError("data.train_dataloader_num_workers must be >= 0")
        if self.val_dataloader_num_workers < 0:
            raise ValueError("data.val_dataloader_num_workers must be >= 0")
        if self.dataloader_prefetch_factor < 1:
            raise ValueError("data.dataloader_prefetch_factor must be >= 1")

        self.image_dir = get_abs_path(self.image_dir, prompt="Image directory")
        self.val_image_dir = get_abs_path(self.val_image_dir, prompt="Validation image directory")
        self.format_prompt = get_abs_path(self.format_prompt, prompt="Format prompt file")
        self.system_prompt = get_abs_path(self.system_prompt, prompt="System prompt file")
        self.override_chat_template = get_abs_path(self.override_chat_template, prompt="Chat template file")


@dataclass
class AlgorithmConfig:
    gamma: float = 1.0
    """discount factor for ppo gae advantage estimator"""
    lam: float = 1.0
    """lambda value for ppo gae advantage estimator"""
    adv_estimator: str = "grpo"
    """advantage estimator; paper ablations use `scalar_grpo_matched` and `gdpo` (FARPO)"""
    disable_kl: bool = False
    """disable reference model"""
    use_kl_loss: bool = False
    """use kl loss instead of kl in reward"""
    kl_penalty: str = "kl"
    """kl penalty type, support `kl`, `abs`, `mse`, `low_var_kl`, `full`"""
    kl_coef: float = 1e-3
    """kl coefficient"""
    kl_type: str = "fixed"
    """kl controller type, support `fixed`, `adaptive`"""
    kl_horizon: float = 10000.0
    """kl horizon for adaptive kl controller"""
    kl_target: float = 0.1
    """target kl for adaptive kl controller"""
    online_filtering: bool = False
    """use online filtering"""
    filter_key: str = "overall"
    """reward key for filtering samples"""
    filter_low: float = 0.01
    """filter out low reward samples if online filtering"""
    filter_high: float = 0.99
    """filter out high reward samples if online filtering"""
    # Component-reward estimator configuration. The gdpo names are retained for compatibility.
    gdpo_reward_keys: str = "format,action_type,action_params,folding"
    """Matched scalar GRPO and GDPO/FARPO reward component keys."""
    gdpo_reward_weights: str = "0.1,0.4,0.4,0.1"
    """Matched scalar GRPO and GDPO/FARPO component weights."""

    def post_init(self):
        component_estimators = {"gdpo", "scalar_grpo_matched"}
        if self.adv_estimator not in component_estimators:
            return

        keys = [key.strip() for key in self.gdpo_reward_keys.split(",") if key.strip()]
        weights = [weight.strip() for weight in self.gdpo_reward_weights.split(",") if weight.strip()]
        if not keys:
            raise ValueError("algorithm.gdpo_reward_keys must not be empty")
        if len(keys) != len(set(keys)):
            raise ValueError("algorithm.gdpo_reward_keys must not contain duplicates")
        if len(keys) != len(weights):
            raise ValueError(
                "algorithm.gdpo_reward_keys and algorithm.gdpo_reward_weights must have the same length"
            )
        try:
            [float(weight) for weight in weights]
        except ValueError as exc:
            raise ValueError("algorithm.gdpo_reward_weights must contain numeric values") from exc


@dataclass
class TrainerConfig:
    seed: int = 1
    """global driver seed; worker, rollout, data, and filter seeds are configured separately"""
    total_epochs: int = 15
    """total epochs for training"""
    max_steps: Optional[int] = None
    """max steps for training, if specified, total_epochs is ignored"""
    project_name: str = "easy_r1"
    """project name for logger"""
    experiment_name: str = "demo"
    """experiment name for logger"""
    logger: Tuple[str] = ("console", "wandb")
    """logger type, support `console`, `mlflow`, `swanlab`, `tensorboard`, `wandb`"""
    nnodes: int = 1
    """number of nodes for training"""
    n_gpus_per_node: int = 8
    """number of gpus per node for training"""
    max_try_make_batch: int = 20
    """max number of generations for online filtering, -1 means no limit"""
    critic_warmup: int = 0
    """critic warmup steps"""
    val_freq: int = -1
    """validation frequency, -1 means no validation"""
    val_before_train: bool = True
    """validate before training"""
    val_only: bool = False
    """validate only, skip training"""
    val_generations_to_log: int = 0
    """number of generations to log for validation, -1 means log all"""
    train_generations_to_log: int = 0
    """number of generations to log for training, -1 means log all"""
    wandb_log_generations: bool = False
    """whether to upload generation text tables to wandb; disabled by default to avoid large artifacts"""
    generation_log_max_chars: int = 4000
    """max characters per prompt/output/label in generation logs; <=0 means no truncation"""
    save_freq: int = -1
    """save frequency, -1 means no saving"""
    save_limit: int = -1
    """max number of checkpoints to save, -1 means no limit"""
    save_model_only: bool = False
    """save model only, no optimizer state dict"""
    save_checkpoint_path: Optional[str] = None
    """save checkpoint path, if not specified, use `checkpoints/project_name/experiment_name`"""
    load_checkpoint_path: Optional[str] = None
    """load checkpoint path"""
    ray_timeline: Optional[str] = None
    """file to save ray timeline"""
    find_last_checkpoint: bool = True
    """automatically find the last checkpoint in the save checkpoint path to resume training"""

    def post_init(self):
        if self.save_checkpoint_path is None:
            self.save_checkpoint_path = os.path.join("checkpoints", self.project_name, self.experiment_name)

        self.save_checkpoint_path = os.path.abspath(self.save_checkpoint_path)  # may be not exist
        self.load_checkpoint_path = get_abs_path(self.load_checkpoint_path, prompt="Model checkpoint")


@dataclass
class PPOConfig:
    data: DataConfig = field(default_factory=DataConfig)
    worker: WorkerConfig = field(default_factory=WorkerConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)

    def post_init(self):
        self.worker.rollout.prompt_length = self.data.max_prompt_length
        self.worker.rollout.response_length = self.data.max_response_length
        self.worker.rollout.trust_remote_code = self.worker.actor.model.trust_remote_code
        self.worker.actor.disable_kl = self.algorithm.disable_kl
        self.worker.actor.use_kl_loss = self.algorithm.use_kl_loss
        self.worker.actor.kl_penalty = self.algorithm.kl_penalty
        self.worker.actor.kl_coef = self.algorithm.kl_coef

    def deep_post_init(self):
        recursive_post_init(self)

    def to_dict(self):
        return asdict(self)
