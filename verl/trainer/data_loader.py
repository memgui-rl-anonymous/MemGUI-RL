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

from typing import Optional

import torch
from torch.utils.data import RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizer, ProcessorMixin

from ..utils.dataset import MobileSERLHFDataset, RLHFDataset, collate_fn
from .config import DataConfig


def _dataloader_worker_kwargs(
    num_workers: int,
    prefetch_factor: int,
    persistent_workers: bool,
) -> dict[str, object]:
    """Build worker-only kwargs without passing invalid options in single-process mode."""
    if num_workers < 0:
        raise ValueError(f"DataLoader num_workers must be >= 0, got {num_workers}")

    kwargs: dict[str, object] = {"num_workers": num_workers}
    if num_workers > 0:
        if prefetch_factor < 1:
            raise ValueError(f"DataLoader prefetch_factor must be >= 1, got {prefetch_factor}")
        kwargs["prefetch_factor"] = prefetch_factor
        kwargs["persistent_workers"] = persistent_workers

    return kwargs


def create_dataloader(config: DataConfig, tokenizer: PreTrainedTokenizer, processor: Optional[ProcessorMixin]) -> None:
    """
    Create the training and validation data loaders.

    Three modes:
    1. standard: RLHFDataset with separate train_files and val_files
    2. MemGUI-3K: MobileSERLHFDataset; with a single file the train/val split is made by task_id
    3. mixed: MobileSERLHFDataset for training, RLHFDataset (GUI-R1 format) for validation
    """

    if config.use_mobilese_format:
        # MemGUI-3K format for training
        print("[MobileSE] using the MemGUI-3K (MobileSE) data format")
        print(f"[MobileSE] train data: {config.train_files}")
        print(f"[MobileSE] loading mode: {config.data_loading_mode}")
        print(f"[MobileSE] positive only: {config.positive_only}")

        # separate-file mode when train_files and val_files differ
        use_separate_files = (
            config.val_use_mobilese_format
            and config.val_files
            and config.val_files != config.train_files
        )

        if use_separate_files:
            print("[MobileSE] separate files for training and validation")
            print(f"[MobileSE] train data: {config.train_files}")
            print(f"[MobileSE] validation data: {config.val_files}")

        train_dataset = MobileSERLHFDataset(
            data_path=config.train_files,
            tokenizer=tokenizer,
            processor=processor,
            is_train=True,
            data_loading_mode=config.data_loading_mode,
            train_val_split=config.train_val_split,
            split_seed=config.split_seed,
            positive_only=config.positive_only,
            train_task_count=config.train_task_count,
            val_task_count=config.val_task_count,
            load_all=use_separate_files,  # load everything in separate-file mode
            max_prompt_length=config.max_prompt_length,
            truncation="right",
            format_prompt=config.format_prompt,
            system_prompt=config.system_prompt,
            min_pixels=config.min_pixels,
            max_pixels=config.max_pixels,
            filter_overlong_prompts=config.filter_overlong_prompts,
        )

        # validation set: format chosen by the config
        if config.val_use_mobilese_format:
            if use_separate_files:
                # separate-file mode: load all of val_files
                print("[MobileSE] validation set: loading the whole separate file")
                val_dataset = MobileSERLHFDataset(
                    data_path=config.val_files,  # separate validation file
                    tokenizer=tokenizer,
                    processor=processor,
                    is_train=False,
                    data_loading_mode=config.data_loading_mode,
                    positive_only=config.positive_only,
                    load_all=True,  # load everything, no split
                    max_prompt_length=config.max_prompt_length,
                    truncation="right",
                    format_prompt=config.format_prompt,
                    system_prompt=config.system_prompt,
                    min_pixels=config.min_pixels,
                    max_pixels=config.max_pixels,
                    filter_overlong_prompts=config.filter_overlong_prompts,
                )
            else:
                # single-file split mode (backward compatible)
                if config.train_task_count > 0 or config.val_task_count > 0:
                    print(f"[MobileSE] validation set: split from the training file (train={config.train_task_count}, val={config.val_task_count} trajectories)")
                else:
                    print(f"[MobileSE] validation set: split from the training file ({1 - config.train_val_split:.0%})")
                val_dataset = MobileSERLHFDataset(
                    data_path=config.train_files,  # same file, split by task_id
                    tokenizer=tokenizer,
                    processor=processor,
                    is_train=False,
                    data_loading_mode=config.data_loading_mode,
                    train_val_split=config.train_val_split,
                    split_seed=config.split_seed,
                    positive_only=config.positive_only,
                    train_task_count=config.train_task_count,
                    val_task_count=config.val_task_count,
                    max_prompt_length=config.max_prompt_length,
                    truncation="right",
                    format_prompt=config.format_prompt,
                    system_prompt=config.system_prompt,
                    min_pixels=config.min_pixels,
                    max_pixels=config.max_pixels,
                    filter_overlong_prompts=config.filter_overlong_prompts,
                )
        else:
            # separate GUI-R1-format validation set
            # GUI-R1 field mapping:
            #   - prompt_key: "instruction" (not the default "problem")
            #   - image_key: "image" (singular)
            #   - no answer_key: gt_action / gt_bbox / gt_input_text are combined
            print("[mixed] validation set in GUI-R1 format")
            print(f"[mixed] validation data: {config.val_files}")
            print(f"[mixed] validation image dir: {config.val_image_dir}")
            val_dataset = RLHFDataset(
                data_path=config.val_files,
                tokenizer=tokenizer,
                processor=processor,
                prompt_key="instruction",  # GUI-R1 prompt field
                answer_key="gt_action",  # absent -> triggers the gt_bbox/gt_action/gt_input_text path
                image_key="image",  # GUI-R1 uses the singular key
                video_key=config.video_key,
                image_dir=config.val_image_dir,  # validation image directory
                video_fps=config.video_fps,
                max_prompt_length=config.max_prompt_length,
                truncation="right",
                format_prompt=config.format_prompt,
                system_prompt=config.system_prompt,
                min_pixels=config.min_pixels,
                max_pixels=config.max_pixels,
                filter_overlong_prompts=config.filter_overlong_prompts,
            )
    else:
        # standard mode: separate training and validation files
        train_dataset = RLHFDataset(
            data_path=config.train_files,
            tokenizer=tokenizer,
            processor=processor,
            prompt_key=config.prompt_key,
            answer_key=config.answer_key,
            image_key=config.image_key,
            video_key=config.video_key,
            image_dir=config.image_dir,
            video_fps=config.video_fps,
            max_prompt_length=config.max_prompt_length,
            truncation="right",
            format_prompt=config.format_prompt,
            system_prompt=config.system_prompt,
            min_pixels=config.min_pixels,
            max_pixels=config.max_pixels,
            filter_overlong_prompts=config.filter_overlong_prompts,
            filter_overlong_prompts_workers=config.filter_overlong_prompts_workers,
        )

        val_dataset = RLHFDataset(
            data_path=config.val_files,
            tokenizer=tokenizer,
            processor=processor,
            prompt_key=config.prompt_key,
            answer_key=config.answer_key,
            image_key=config.image_key,
            video_key=config.video_key,
            image_dir=config.image_dir,
            video_fps=config.video_fps,
            max_prompt_length=config.max_prompt_length,
            truncation="right",
            format_prompt=config.format_prompt,
            system_prompt=config.system_prompt,
            min_pixels=config.min_pixels,
            max_pixels=config.max_pixels,
            filter_overlong_prompts=config.filter_overlong_prompts,
        )

    # training dataloader
    if config.shuffle:
        train_dataloader_generator = torch.Generator()
        train_dataloader_generator.manual_seed(config.seed)
        sampler = RandomSampler(data_source=train_dataset, generator=train_dataloader_generator)
    else:
        sampler = SequentialSampler(data_source=train_dataset)

    if config.mini_rollout_batch_size is not None:
        train_batch_size = config.mini_rollout_batch_size
    else:
        train_batch_size = config.rollout_batch_size

    train_worker_kwargs = _dataloader_worker_kwargs(
        num_workers=config.train_dataloader_num_workers,
        prefetch_factor=config.dataloader_prefetch_factor,
        persistent_workers=config.dataloader_persistent_workers,
    )
    print(
        "Train DataLoader workers: "
        f"{config.train_dataloader_num_workers}, prefetch_factor: "
        f"{config.dataloader_prefetch_factor if config.train_dataloader_num_workers > 0 else 'disabled'}"
    )
    train_dataloader = StatefulDataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
        pin_memory=False,
        drop_last=True,
        **train_worker_kwargs,
    )

    # validation dataloader
    if config.val_batch_size == -1:
        val_batch_size = len(val_dataset)
    else:
        val_batch_size = config.val_batch_size

    val_worker_kwargs = _dataloader_worker_kwargs(
        num_workers=config.val_dataloader_num_workers,
        prefetch_factor=config.dataloader_prefetch_factor,
        persistent_workers=config.dataloader_persistent_workers,
    )
    print(
        "Val DataLoader workers: "
        f"{config.val_dataloader_num_workers}, prefetch_factor: "
        f"{config.dataloader_prefetch_factor if config.val_dataloader_num_workers > 0 else 'disabled'}"
    )
    val_dataloader = StatefulDataLoader(
        dataset=val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=False,
        drop_last=False,
        **val_worker_kwargs,
    )

    assert len(train_dataloader) >= 1, f"Train dataloader is empty! Dataset size: {len(train_dataset)}"
    assert len(val_dataloader) >= 1, f"Val dataloader is empty! Dataset size: {len(val_dataset)}"
    print(f"Size of train dataloader: {len(train_dataloader)}")
    print(f"Size of val dataloader: {len(val_dataloader)}")
    return train_dataloader, val_dataloader
