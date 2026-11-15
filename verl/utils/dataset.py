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

import base64
import json
import math
import os
import random
from collections import defaultdict
from io import BytesIO
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch
from datasets import Dataset as HFDataset
from datasets import load_dataset
from jinja2 import Template
from PIL import Image
from PIL.Image import Image as ImageObject
from qwen_vl_utils.vision_process import fetch_video
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from . import torch_functional as VF


def _split_data_paths(data_path: Union[str, List[str], Tuple[str, ...]]) -> List[str]:
    """Normalize a data path config into one or more file paths."""
    if isinstance(data_path, (list, tuple)):
        data_paths = list(data_path)
    else:
        data_paths = str(data_path).split(",")

    data_paths = [path.strip() for path in data_paths if str(path).strip()]
    if not data_paths:
        raise ValueError("data_path is empty")

    return data_paths


def collate_fn(features: list[dict[str, Any]]) -> dict[str, Any]:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)
    for feature in features:
        for key, value in feature.items():
            if isinstance(value, torch.Tensor):
                tensors[key].append(value)
            else:
                non_tensors[key].append(value)

    for key, value in tensors.items():
        tensors[key] = torch.stack(value, dim=0)

    for key, value in non_tensors.items():
        non_tensors[key] = np.array(value, dtype=object)

    return {**tensors, **non_tensors}


def process_image(
    image: Union[dict[str, Any], ImageObject, str], min_pixels: Optional[int], max_pixels: Optional[int]
) -> ImageObject:
    if isinstance(image, str):
        image = Image.open(image)
    elif isinstance(image, dict):
        image = Image.open(BytesIO(image["bytes"]))
    elif isinstance(image, bytes):
        image = Image.open(BytesIO(image))

    image.load()  # avoid "Too many open files" errors
    if max_pixels is not None and (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if min_pixels is not None and (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if image.mode != "RGB":
        image = image.convert("RGB")

    return image


def process_video(
    video: str, min_pixels: Optional[int], max_pixels: Optional[int], video_fps: float, return_fps: bool = False
) -> Union[list[ImageObject], tuple[list[ImageObject], list[float]]]:
    vision_info = {"video": video, "min_pixels": min_pixels, "max_pixels": max_pixels, "fps": video_fps}
    return fetch_video(vision_info, return_video_sample_fps=return_fps)


class RLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(
        self,
        data_path: Union[str, List[str], Tuple[str, ...]],
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        prompt_key: str = "prompt",
        answer_key: str = "answer",
        image_key: str = "images",
        video_key: str = "videos",
        image_dir: Optional[str] = None,
        video_fps: float = 2.0,
        max_prompt_length: int = 1024,
        truncation: str = "error",
        format_prompt: Optional[str] = None,
        system_prompt: Optional[str] = None,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        filter_overlong_prompts: bool = True,
        filter_overlong_prompts_workers: int = 16,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.prompt_key = prompt_key
        self.answer_key = answer_key
        self.image_key = image_key
        self.video_key = video_key
        self.image_dir = image_dir
        self.video_fps = video_fps
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

        data_paths = _split_data_paths(data_path)
        if len(data_paths) > 1:
            if any("@" in path for path in data_paths):
                raise ValueError("Multiple local data files do not support @split suffixes")
            if not all(os.path.isfile(path) for path in data_paths):
                raise ValueError(f"Multiple data files should all be local files: {data_paths}")
            file_type = os.path.splitext(data_paths[0])[-1][1:].replace("jsonl", "json")
            self.dataset = load_dataset(file_type, data_files=data_paths, split="train")
        else:
            data_path = data_paths[0]
            if "@" in data_path:
                data_path, data_split = data_path.split("@")
            else:
                data_split = "train"

            if os.path.isdir(data_path):
                # when we use dataset builder, we should always refer to the train split
                file_type = os.path.splitext(os.listdir(data_path)[0])[-1][1:].replace("jsonl", "json")
                self.dataset = load_dataset(file_type, data_dir=data_path, split=data_split)
            elif os.path.isfile(data_path):
                file_type = os.path.splitext(data_path)[-1][1:].replace("jsonl", "json")
                self.dataset = load_dataset(file_type, data_files=data_path, split=data_split)
            else:
                # load remote dataset from huggingface hub
                self.dataset = load_dataset(data_path, split=data_split)

        self.format_prompt = None
        if format_prompt:
            with open(format_prompt, encoding="utf-8") as f:
                self.format_prompt = f.read()

        # system prompt (jinja2 template or plain text)
        self.system_prompt = None
        if system_prompt:
            with open(system_prompt, encoding="utf-8") as f:
                self.system_prompt = f.read()

        if filter_overlong_prompts:
            self.dataset = self.dataset.filter(
                self._filter_overlong_prompts,
                desc="Filtering overlong prompts",
                num_proc=filter_overlong_prompts_workers,
            )

    def _build_messages(self, example: dict[str, Any]) -> list[dict[str, Any]]:
        messages = []

        # system message (when system_prompt is configured)
        if self.system_prompt:
            system_prompt_template = Template(self.system_prompt.strip())
            # template variables are allowed in the system prompt
            system_content = system_prompt_template.render(content="", **example)
            messages.append({"role": "system", "content": system_content})

        # user message
        prompt_str: str = example[self.prompt_key]
        if self.format_prompt:
            format_prompt = Template(self.format_prompt.strip())
            # the whole example is passed to the template (instruction, history, ...)
            # `content` is kept for compatibility
            prompt_str = format_prompt.render(content=prompt_str, **example)

        if self.image_key in example:
            # https://huggingface.co/docs/transformers/en/tasks/image_text_to_text
            content_list = []
            for i, content in enumerate(prompt_str.split("<image>")):
                if i != 0:
                    content_list.append({"type": "image"})

                if content:
                    content_list.append({"type": "text", "text": content})

            messages.append({"role": "user", "content": content_list})
        elif self.video_key in example:
            content_list = []
            for i, content in enumerate(prompt_str.split("<video>")):
                if i != 0:
                    content_list.append({"type": "video"})

                if content:
                    content_list.append({"type": "text", "text": content})

            messages.append({"role": "user", "content": content_list})
        else:
            messages.append({"role": "user", "content": prompt_str})

        return messages

    def _filter_overlong_prompts(self, example: dict[str, Any]) -> bool:
        messages = self._build_messages(example)
        if self.image_key in example:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            images = example[self.image_key]
            # a single image (not a list) is accepted
            if not isinstance(images, (list, tuple)):
                images = [images]
            if self.image_dir is not None and len(images) != 0 and isinstance(images[0], str):  # image paths
                images = [os.path.join(self.image_dir, image) for image in images]

            processed_images = [] if len(images) != 0 else None  # text-only data
            for image in images:
                processed_images.append(process_image(image, self.min_pixels, self.max_pixels))

            model_inputs = self.processor(processed_images, [prompt], add_special_tokens=False, return_tensors="pt")
            return model_inputs["input_ids"].size(-1) <= self.max_prompt_length
        elif self.video_key in example:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            videos = example[self.video_key]
            if self.image_dir is not None and len(videos) != 0 and isinstance(videos[0], str):  # video paths
                videos = [os.path.join(self.image_dir, video) for video in videos]

            processed_videos = [] if len(videos) != 0 else None  # text-only data
            for video in videos:
                processed_videos.append(process_video(video, self.min_pixels, self.max_pixels, self.video_fps))

            model_inputs = self.processor(
                videos=processed_videos, text=[prompt], add_special_tokens=False, return_tensors="pt"
            )
            return model_inputs["input_ids"].size(-1) <= self.max_prompt_length
        else:
            input_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            return len(input_ids) <= self.max_prompt_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        example: dict = self.dataset[index]
        messages = self._build_messages(example)
        example.pop(self.prompt_key, None)

        image_size = None  # original image size, used to rescale GUI-R1 coordinates
        if self.image_key in example:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            images = example.pop(self.image_key)
            # a single image (not a list) is accepted
            if not isinstance(images, (list, tuple)):
                images = [images]
            if self.image_dir is not None and len(images) != 0 and isinstance(images[0], str):  # image paths
                images = [os.path.join(self.image_dir, image) for image in images]

            # original image size first (GUI-R1 coordinate conversion)
            # gt_bbox is in original pixels and is normalised to 0-1000 with the original size
            if len(images) > 0:
                first_image = images[0]
                if isinstance(first_image, str):
                    with Image.open(first_image) as img:
                        image_size = (img.width, img.height)
                elif isinstance(first_image, dict) and "bytes" in first_image:
                    with Image.open(BytesIO(first_image["bytes"])) as img:
                        image_size = (img.width, img.height)
                elif isinstance(first_image, bytes):
                    with Image.open(BytesIO(first_image)) as img:
                        image_size = (img.width, img.height)
                elif hasattr(first_image, "width") and hasattr(first_image, "height"):
                    image_size = (first_image.width, first_image.height)

            processed_images = [] if len(images) != 0 else None  # text-only data
            for image in images:
                processed_images.append(process_image(image, self.min_pixels, self.max_pixels))

            model_inputs = self.processor(processed_images, [prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            example["multi_modal_data"] = {"images": images}
        elif self.video_key in example:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            videos = example.pop(self.video_key)
            if self.image_dir is not None and len(videos) != 0 and isinstance(videos[0], str):  # video paths
                videos = [os.path.join(self.image_dir, video) for video in videos]

            processed_videos = [] if len(videos) != 0 else None  # text-only data
            video_fps_list = []
            for video in videos:
                processed_video, video_fps = process_video(
                    video, self.min_pixels, self.max_pixels, self.video_fps, return_fps=True
                )
                processed_videos.append(processed_video)
                video_fps_list.append(video_fps)

            model_inputs = self.processor(
                videos=processed_videos, text=[prompt], add_special_tokens=False, return_tensors="pt"
            )
            if "second_per_grid_ts" in self.processor.model_input_names:
                model_inputs["second_per_grid_ts"] = [2.0 / video_sample_fps for video_sample_fps in video_fps_list]

            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            example["multi_modal_data"] = {"videos": videos}
        else:
            prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            model_inputs = self.tokenizer([prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]

        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            # qwen-vl mrope
            if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                from ..models.transformers.qwen3_vl import get_rope_index
            else:
                from ..models.transformers.qwen2_vl import get_rope_index

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=model_inputs.get("image_grid_thw", None),
                video_grid_thw=model_inputs.get("video_grid_thw", None),
                second_per_grid_ts=model_inputs.get("second_per_grid_ts", None),
                attention_mask=attention_mask,
            )  # (3, seq_length)
            text_position_ids = torch.arange(len(input_ids)).unsqueeze(0)  # (1, seq_length)
            position_ids = torch.cat((text_position_ids, vision_position_ids), dim=0)  # (4, seq_length)
        else:
            position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)  # (seq_length,)

        input_ids, attention_mask, position_ids = VF.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        raw_prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        example["input_ids"] = input_ids
        example["attention_mask"] = attention_mask
        example["position_ids"] = position_ids
        example["raw_prompt_ids"] = raw_prompt_ids

        # GUI-R1 format:
        # gt_bbox and gt_action take precedence
        # combined into ground_truth; otherwise answer_key is used
        if "gt_bbox" in example and "gt_action" in example:
            # GUI-R1: rescale gt_bbox by the image size
            gt_bbox = list(example.pop("gt_bbox", [0, 0]))  # copy as a mutable list
            gt_action = example.pop("gt_action", "click")
            gt_input_text = example.pop("gt_input_text", "")

            # whether the original coordinates are normalised (0-1)
            # all (2 or 4) values must lie in [0, 1]
            # GUI-R1 coordinates are usually pixels
            is_normalized_coord = len(gt_bbox) >= 2 and all(0 <= coord <= 1 for coord in gt_bbox)

            # bbox_valid: non-negative coordinates are valid
            bbox_valid = len(gt_bbox) >= 2 and all(coord >= 0 for coord in gt_bbox)

            # rescale gt_bbox by the processed image size (normalised 0-1 input)
            if image_size is not None and len(gt_bbox) >= 2 and is_normalized_coord:
                scale_x, scale_y = image_size
                gt_bbox[0] *= scale_x
                gt_bbox[1] *= scale_y
                if len(gt_bbox) > 2:
                    gt_bbox[2] *= scale_x
                if len(gt_bbox) > 3:
                    gt_bbox[3] *= scale_y

            # ground_truth JSON, including the image size for coordinate conversion in the reward
            gt = {
                "action": gt_action,
                "gt_bbox": gt_bbox,
                "input_text": gt_input_text,
                # image size for the 0-1000 conversion in the reward function
                "image_size": list(image_size) if image_size else None,
                # coordinate validity ([-0.1, -0.1] and similar sentinels are invalid)
                "bbox_valid": bbox_valid,
                # normalised-coordinate flag (used for the swipe-direction convention)
                # False for GUI-R1, True for MemGUI-3K
                "is_normalized": False,  # GUI-R1 coordinates are not normalised
            }
            example["ground_truth"] = json.dumps(gt)
        elif self.answer_key in example:
            # field named by answer_key
            example["ground_truth"] = example.pop(self.answer_key)
        else:
            # fall back to an empty string
            example["ground_truth"] = ""

        return example


def _extract_raw_response(conversations: list) -> str:
    """
    Extract the reference assistant response from a MemGUI-3K conversation.

    Used in training logs (thinking, tool_call, conclusion).
    """
    for conv in conversations:
        if conv.get("role") == "assistant":
            content = conv.get("content", [])
            if isinstance(content, str):
                return content
            elif isinstance(content, list):
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                return "".join(text_parts)
    return ""


def _parse_ground_truth_from_conversations(conversations: list) -> dict:
    """
    Parse the ground truth from a MemGUI-3K conversation.

    The assistant turn carries <tool_call> and <folding> tags; they are parsed into the
    ground_truth structure expected by the reward function.

    Returned structure:
    {
        "action": "click",      # action type
        "gt_bbox": [x, y],      # coordinate (0-1000)
        "input_text": "",       # typed text or swipe direction
        "is_normalized": True,  # coordinates are normalised
        "bbox_valid": True,     # coordinate validity
        "folding": {...},       # folding directive (optional)
        "memory_id": "",        # memory id (optional)
        "content": "",          # memory content (optional)
        "description": ""       # memory description (optional)
    }
    """
    import re as _re

    # empty ground truth
    gt = {
        "action": "",
        "gt_bbox": [],
        "input_text": "",
        "is_normalized": True,  # MemGUI-3K coordinates are 0-1000 normalised
        "bbox_valid": True,
        "folding": None,        # folding directive
        "memory_id": "",        # memory id
        "content": "",          # memory content
        "description": "",      # memory description
    }

    # text of the assistant turn
    assistant_text = ""
    for conv in conversations:
        if conv.get("role") == "assistant":
            content = conv.get("content", [])
            if isinstance(content, str):
                assistant_text = content
            elif isinstance(content, list):
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                assistant_text = "".join(text_parts)
            break

    if not assistant_text:
        return gt

    # JSON inside <tool_call>
    tool_call_match = _re.search(r"<tool_call>\s*({.*?})\s*</tool_call>", assistant_text, _re.DOTALL)
    if tool_call_match:
        try:
            tool_call = json.loads(tool_call_match.group(1))
            args = tool_call.get("arguments", tool_call)

            # action
            action = args.get("action", "")
            gt["action"] = action

            # coordinate
            coord = args.get("coordinate", [])
            if coord and len(coord) >= 2:
                gt["gt_bbox"] = [coord[0], coord[1]]

            # swipe: derive the direction and map to scroll
            if action == "swipe":
                coord2 = args.get("coordinate2", [])
                if coord and coord2 and len(coord) >= 2 and len(coord2) >= 2:
                    dx = coord2[0] - coord[0]
                    dy = coord2[1] - coord[1]
                    if abs(dx) > abs(dy):
                        direction = "right" if dx > 0 else "left"
                    else:
                        direction = "down" if dy > 0 else "up"
                    gt["input_text"] = direction
                    gt["action"] = "scroll"  # swipe -> scroll
                elif args.get("direction"):
                    gt["input_text"] = args.get("direction", "")
                    gt["action"] = "scroll"

            # typed text
            if action == "type":
                gt["input_text"] = args.get("text", "")

            # answer text
            if action == "answer":
                gt["input_text"] = args.get("text", "")

            # system_button
            if action == "system_button":
                button = args.get("button", "").lower()
                button_mapping = {"back": "press_back", "home": "press_home", "menu": "press_recent", "enter": "enter"}
                gt["action"] = button_mapping.get(button, action)

            # terminate -> complete, keeping the status
            if action == "terminate":
                gt["action"] = "complete"
                gt["input_text"] = args.get("status", "success")  # status stored in input_text

            # wait: duration
            if action == "wait":
                time_val = args.get("time", 0)
                gt["input_text"] = str(time_val) if time_val else ""

            # long_press: duration in an extra field
            if action == "long_press":
                time_val = args.get("time", 0)
                if time_val:
                    gt["time"] = time_val

            # open_app
            if action == "open":
                gt["action"] = "open_app"
                gt["input_text"] = args.get("app_name", "")

            # memory operations
            if action in ["memory_add", "memory_update", "memory_delete"]:
                gt["memory_id"] = args.get("memory_id", "")
                gt["content"] = args.get("content", "")
                gt["description"] = args.get("description", "")

        except (json.JSONDecodeError, TypeError, KeyError):
            pass  # continue with folding

    # <folding> tag (independent of tool_call)
    folding_match = _re.search(r"<folding>\s*(.*?)\s*</folding>", assistant_text, _re.DOTALL | _re.IGNORECASE)
    if folding_match:
        folding_str = folding_match.group(1).strip()
        try:
            if folding_str.startswith("{"):
                gt["folding"] = json.loads(folding_str)
            else:
                # not JSON: treat the text as the summary
                gt["folding"] = {"range": [0, 0], "summary": folding_str}
        except json.JSONDecodeError:
            gt["folding"] = {"range": [0, 0], "summary": folding_str}

    return gt


def compute_difficulty(attempt_stats: dict) -> str:
    """
    Task difficulty from attempt_stats.

    Definition:
    - easy: every attempt succeeded
    - medium: some attempts succeeded
    - hard: every attempt failed

    Args:
        attempt_stats: dict with total_attempts and pass@attempts

    Returns:
        "easy", "medium", "hard" or "unknown"
    """
    total = attempt_stats.get("total_attempts", 0)
    passed = attempt_stats.get("pass@attempts", 0)

    if total == 0:
        return "unknown"
    elif passed == total:
        return "easy"
    elif passed == 0:
        return "hard"
    else:
        return "medium"


def load_mobilese_data(
    data_path: Union[str, List[str], Tuple[str, ...]],
    data_loading_mode: int = 3,
    train_val_split: float = 0.9,
    split_seed: int = 42,
    is_train: bool = True,
    positive_only: bool = True,
    train_task_count: int = -1,
    val_task_count: int = -1,
    load_all: bool = False,
) -> Tuple[List[dict], List[str]]:
    """
    Load MemGUI-3K-format data and split train/val by task_id.

    Args:
        data_path: one path, comma-separated paths or a list of JSON files
        data_loading_mode: difficulty filter (1: medium, 2: medium+easy, 3: all)
        train_val_split: train fraction (used when both task counts are -1)
        split_seed: seed of the split
        is_train: return the training part
        positive_only: keep only positive samples
        train_task_count: number of training trajectories; -1 uses train_val_split
        val_task_count: number of validation trajectories; -1 uses train_val_split
        load_all: skip the split and load everything (separate-file mode)

    Returns:
        (samples, task_ids used by the split)
    """
    # raw data
    data_paths = _split_data_paths(data_path)
    all_samples = []
    for current_path in data_paths:
        with open(current_path, "r", encoding="utf-8") as f:
            current_samples = json.load(f)
        if not isinstance(current_samples, list):
            raise ValueError(f"MobileSE data file should contain a JSON list: {current_path}")
        print(f"[MobileSE Debug] loaded {current_path}: {len(current_samples)} samples")
        all_samples.extend(current_samples)

    print(f"[MobileSE Debug] raw samples: {len(all_samples)}")

    # derive the fields needed downstream
    difficulty_counts = {"easy": 0, "medium": 0, "hard": 0, "unknown": 0}
    for sample in all_samples:
        metadata = sample.get("metadata", {})

        # task_id: top-level field, else metadata
        if "task_id" not in sample:
            sample["task_id"] = metadata.get("task_id", f"unknown_{id(sample)}")

        # step_number: top-level field, else metadata
        if "step_number" not in sample:
            sample["step_number"] = metadata.get("step_number", 0)

        # is_positive: top-level field, else metadata
        # a sample is positive iff impact == "positive"
        if "is_positive" not in sample:
            impact = metadata.get("impact", "")
            sample["is_positive"] = impact == "positive"

        # ground_truth: top-level field, else parsed from the assistant turn
        if "ground_truth" not in sample:
            ground_truth = _parse_ground_truth_from_conversations(sample.get("conversations", []))
            sample["ground_truth"] = ground_truth

        # reference assistant response (for logging)
        if "raw_response" not in sample:
            sample["raw_response"] = _extract_raw_response(sample.get("conversations", []))

        # difficulty from attempt_stats
        attempt_stats = metadata.get("attempt_stats", {})
        if not attempt_stats:
            attempt_stats = sample.get("attempt_stats", {})
        sample["difficulty"] = compute_difficulty(attempt_stats)
        difficulty_counts[sample["difficulty"]] = difficulty_counts.get(sample["difficulty"], 0) + 1

    print(f"[MobileSE Debug] difficulty distribution: {difficulty_counts}")
    positive_count = sum(1 for s in all_samples if s.get("is_positive", False))
    print(f"[MobileSE Debug] positive samples: {positive_count}/{len(all_samples)}")

    # difficulty filter
    before_difficulty_filter = len(all_samples)
    if data_loading_mode == 1:
        all_samples = [s for s in all_samples if s.get("difficulty") == "medium"]
        print(f"[MobileSE Debug] difficulty filter (mode=1, medium only): {before_difficulty_filter} -> {len(all_samples)}")
    elif data_loading_mode == 2:
        all_samples = [s for s in all_samples if s.get("difficulty") in ["medium", "easy"]]
        print(f"[MobileSE Debug] difficulty filter (mode=2, medium+easy): {before_difficulty_filter} -> {len(all_samples)}")
    # mode 3: keep everything

    # positive filter
    before_positive_filter = len(all_samples)
    if positive_only:
        all_samples = [s for s in all_samples if s.get("is_positive", False)]
        print(f"[MobileSE Debug] positive filter: {before_positive_filter} -> {len(all_samples)}")

    # all task_ids
    task_ids = list(set(s["task_id"] for s in all_samples))
    print(f"[MobileSE Debug] task_ids after filtering: {len(task_ids)}")

    # load_all: skip the split
    if load_all:
        print("[MobileSE Debug] load_all=True: loading everything (no train/val split)")
        selected_samples = all_samples
        selected_task_ids = task_ids
        print(f"[MobileSE Debug] final samples: {len(selected_samples)}")

        if len(selected_samples) == 0:
            print("[MobileSE Error] no samples left!")
            print("[MobileSE Error] please check:")
            print("  - the data files were loaded correctly")
            print(f"  - data_loading_mode={data_loading_mode} is not too strict")
            print(f"  - positive_only={positive_only} is intended")

        return selected_samples, selected_task_ids

    # seeded split (identical across runs)
    rng = random.Random(split_seed)
    rng.shuffle(task_ids)

    # split by count when given, otherwise by fraction
    if train_task_count > 0 or val_task_count > 0:
        # split by count
        actual_train_count = train_task_count if train_task_count > 0 else len(task_ids) - val_task_count
        actual_val_count = val_task_count if val_task_count > 0 else len(task_ids) - train_task_count
        
        # requested counts must fit the available task_ids
        total_needed = actual_train_count + actual_val_count
        if total_needed > len(task_ids):
            print(f"[MobileSE Warning] requested {total_needed} task_ids but only {len(task_ids)} are available")
            print(f"[MobileSE Warning] scaled proportionally: train={actual_train_count}, val={actual_val_count}")
            # scale by the requested ratio
            scale = len(task_ids) / total_needed
            actual_train_count = int(actual_train_count * scale)
            actual_val_count = len(task_ids) - actual_train_count
        
        train_task_ids = set(task_ids[:actual_train_count])
        val_task_ids = set(task_ids[actual_train_count:actual_train_count + actual_val_count])
        print(f"[MobileSE Debug] split by count: train={len(train_task_ids)} trajectories, val={len(val_task_ids)} trajectories")
    else:
        # split by fraction
        split_idx = int(len(task_ids) * train_val_split)
        train_task_ids = set(task_ids[:split_idx])
        val_task_ids = set(task_ids[split_idx:])
        print(f"[MobileSE Debug] split by fraction ({train_val_split:.0%}): train={len(train_task_ids)} trajectories, val={len(val_task_ids)} trajectories")

    # select the samples of this split
    if is_train:
        selected_samples = [s for s in all_samples if s["task_id"] in train_task_ids]
        selected_task_ids = list(train_task_ids)
    else:
        selected_samples = [s for s in all_samples if s["task_id"] in val_task_ids]
        selected_task_ids = list(val_task_ids)

    print(f"[MobileSE Debug] {'train' if is_train else 'val'} samples: {len(selected_samples)}")

    if len(selected_samples) == 0:
        print(f"[MobileSE Error] the {'train' if is_train else 'val'} split is empty!")
        print("[MobileSE Error] please check:")
        print("  - the data files were loaded correctly")
        print(f"  - data_loading_mode={data_loading_mode} is not too strict")
        print(f"  - positive_only={positive_only} is intended")
        print("  - the data contain matching samples")

    return selected_samples, selected_task_ids


class MobileSERLHFDataset(Dataset):
    """
    RLHF dataset for the MemGUI-3K (MobileSE) conversation format.

    Supports:
    - conversation records (`conversations` field)
    - base64-encoded or file-referenced images
    - train/val split by task_id
    - difficulty filtering (data_loading_mode)
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        is_train: bool = True,
        data_loading_mode: int = 3,
        train_val_split: float = 0.9,
        split_seed: int = 42,
        positive_only: bool = True,
        train_task_count: int = -1,
        val_task_count: int = -1,
        load_all: bool = False,
        max_prompt_length: int = 1024,
        truncation: str = "error",
        format_prompt: Optional[str] = None,
        system_prompt: Optional[str] = None,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        filter_overlong_prompts: bool = True,
    ):
        """
        Args:
            data_path: one path, comma-separated paths or a list of JSON files
            tokenizer: tokenizer
            processor: multimodal processor
            is_train: return the training part
            data_loading_mode: difficulty filter (1: medium, 2: medium+easy, 3: all)
            train_val_split: train fraction (used when both task counts are -1)
            split_seed: seed of the split
            positive_only: keep only positive samples
            train_task_count: number of training trajectories; -1 uses train_val_split
            val_task_count: number of validation trajectories; -1 uses train_val_split
            load_all: skip the split and load everything (separate-file mode)
            max_prompt_length: maximum prompt length
            truncation: truncation strategy
            format_prompt: user-prompt template file
            system_prompt: system-prompt file
            min_pixels: minimum image pixels
            max_pixels: maximum image pixels
            filter_overlong_prompts: drop prompts longer than max_prompt_length
        """
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.is_train = is_train

        # load and split
        samples, task_ids = load_mobilese_data(
            data_path=data_path,
            data_loading_mode=data_loading_mode,
            train_val_split=train_val_split,
            split_seed=split_seed,
            is_train=is_train,
            positive_only=positive_only,
            train_task_count=train_task_count,
            val_task_count=val_task_count,
            load_all=load_all,
        )
        self.samples = samples
        self.task_ids = task_ids

        print(f"[MobileSE] loading data: is_train={is_train}, mode={data_loading_mode}")
        print(f"[MobileSE] samples: {len(self.samples)}, task_ids: {len(self.task_ids)}")

        # system prompt
        self.system_prompt = None
        if system_prompt:
            with open(system_prompt, encoding="utf-8") as f:
                self.system_prompt = f.read()

        # format prompt
        self.format_prompt = None
        if format_prompt:
            with open(format_prompt, encoding="utf-8") as f:
                self.format_prompt = f.read()

        # drop over-long prompts
        if filter_overlong_prompts:
            original_count = len(self.samples)
            self.samples = [s for s in self.samples if self._check_prompt_length(s)]
            print(f"[MobileSE] over-long prompts removed: {original_count} -> {len(self.samples)}")

    def _check_prompt_length(self, sample: dict) -> bool:
        """Whether the prompt fits max_prompt_length."""
        try:
            messages = self._build_messages(sample)
            if self.processor is not None:
                prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                # image
                images = self._get_images(sample)
                if images:
                    processed_images = [process_image(img, self.min_pixels, self.max_pixels) for img in images]
                    model_inputs = self.processor(
                        processed_images, [prompt], add_special_tokens=False, return_tensors="pt"
                    )
                    return model_inputs["input_ids"].size(-1) <= self.max_prompt_length

            input_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            return len(input_ids) <= self.max_prompt_length
        except Exception:
            return False

    def _build_messages(self, sample: dict) -> list[dict]:
        """
        Build the message list from a MemGUI-3K record.

        Format:
        - `conversations` already holds the full context (folded history, memory, recent step)
        - the system and user turns are used as they are
        - no template re-rendering is needed
        """
        messages = []
        conversations = sample.get("conversations", [])

        # use the stored system and user turns
        # the record already carries the complete context
        for conv in conversations:
            role = conv.get("role", "")
            content = conv.get("content", [])

            # skip the assistant turn (the reference response is not part of the prompt)
            if role == "assistant":
                continue

            if role == "system":
                # system turn
                if isinstance(content, str):
                    messages.append({"role": "system", "content": content})
                elif isinstance(content, list):
                    # text content
                    text_parts = []
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            text_parts.append(item.get("text", ""))
                    if text_parts:
                        messages.append({"role": "system", "content": "".join(text_parts)})

            elif role == "user":
                # user turn (kept verbatim: context plus image placeholder)
                if isinstance(content, str):
                    # user message with the image
                    images = self._get_images(sample)
                    if images and "<image>" in content:
                        content_list = []
                        for i, text_part in enumerate(content.split("<image>")):
                            if i != 0:
                                content_list.append({"type": "image"})
                            if text_part:
                                content_list.append({"type": "text", "text": text_part})
                        messages.append({"role": "user", "content": content_list})
                    else:
                        messages.append({"role": "user", "content": content})
                elif isinstance(content, list):
                    # list content: collect the text parts
                    # note: the text already contains the <image> placeholder
                    # so image_url parts do not add another one
                    user_text = ""
                    has_image_in_list = False
                    for item in content:
                        if isinstance(item, dict):
                            item_type = item.get("type", "")
                            if item_type == "text":
                                user_text += item.get("text", "")
                            elif item_type in ["image", "image_url"]:
                                has_image_in_list = True

                    # is the <image> placeholder already present?
                    has_image_placeholder = "<image>" in user_text

                    # user message with the image
                    images = self._get_images(sample)
                    if images and (has_image_placeholder or has_image_in_list):
                        if has_image_placeholder:
                            # placeholder present: use the text as is
                            content_list = []
                            for i, text_part in enumerate(user_text.split("<image>")):
                                if i != 0:
                                    content_list.append({"type": "image"})
                                if text_part:
                                    content_list.append({"type": "text", "text": text_part})
                            messages.append({"role": "user", "content": content_list})
                        else:
                            # no placeholder: append the image
                            content_list = [{"type": "text", "text": user_text}]
                            for _ in images:
                                content_list.append({"type": "image"})
                            messages.append({"role": "user", "content": content_list})
                    else:
                        messages.append({"role": "user", "content": user_text})

        return messages

    def _get_images(self, sample: dict) -> List[ImageObject]:
        """Images of a record."""
        images = []

        # images referenced in the conversation
        for conv in sample.get("conversations", []):
            content = conv.get("content", [])
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        item_type = item.get("type", "")
                        image_data = None

                        # two image encodings are accepted:
                        # 1. {"type": "image", "image": "base64..."}
                        # 2. {"type": "image_url", "image_url": {"url": "data:image;base64,..."}}
                        if item_type == "image":
                            image_data = item.get("image", "")
                        elif item_type == "image_url":
                            image_url = item.get("image_url", {})
                            if isinstance(image_url, dict):
                                image_data = image_url.get("url", "")
                            elif isinstance(image_url, str):
                                image_data = image_url

                        if image_data:
                            try:
                                img = self._decode_base64_image(image_data)
                                if img:
                                    images.append(img)
                            except Exception:
                                pass

        # also check the `images` field
        if not images and "images" in sample:
            for img_data in sample["images"]:
                try:
                    img = self._decode_base64_image(img_data)
                    if img:
                        images.append(img)
                except Exception:
                    pass

        return images

    def _decode_base64_image(self, image_data: str) -> Optional[ImageObject]:
        """Decode an image from a local path, data URL, or raw base64 string."""
        try:
            # Path-backed image format: JSON stores only the local file path to
            # avoid loading huge embedded base64 payloads into memory.  Relative paths
            # (e.g. "MemGUI-3K/images/xxx.png" in the released MemGUI-3K-Verl files) are
            # resolved against $MEMGUI_IMAGE_ROOT when they do not exist as given.
            if os.path.exists(image_data):
                image = Image.open(image_data)
                image.load()
                return image
            image_root = os.environ.get("MEMGUI_IMAGE_ROOT")
            if image_root and os.path.exists(os.path.join(image_root, image_data)):
                image = Image.open(os.path.join(image_root, image_data))
                image.load()
                return image

            # data:image URIs
            if image_data.startswith("data:image"):
                if "," in image_data:
                    image_data = image_data.split(",", 1)[1]

            image_bytes = base64.b64decode(image_data)
            image = Image.open(BytesIO(image_bytes))
            image.load()
            return image
        except Exception:
            return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        messages = self._build_messages(sample)

        # image
        images = self._get_images(sample)
        image_size = None

        if images:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            processed_images = [process_image(img, self.min_pixels, self.max_pixels) for img in images]

            if processed_images:
                image_size = (processed_images[0].width, processed_images[0].height)

            model_inputs = self.processor(processed_images, [prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
        else:
            prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            model_inputs = self.tokenizer([prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]

        # position_ids (Qwen-VL mrope)
        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                from ..models.transformers.qwen3_vl import get_rope_index
            else:
                from ..models.transformers.qwen2_vl import get_rope_index

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=model_inputs.get("image_grid_thw", None),
                video_grid_thw=model_inputs.get("video_grid_thw", None),
                second_per_grid_ts=model_inputs.get("second_per_grid_ts", None),
                attention_mask=attention_mask,
            )
            text_position_ids = torch.arange(len(input_ids)).unsqueeze(0)
            position_ids = torch.cat((text_position_ids, vision_position_ids), dim=0)
        else:
            position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)

        input_ids, attention_mask, position_ids = VF.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        raw_prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        # output
        metadata = sample.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}

        example = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "raw_prompt_ids": raw_prompt_ids,
            "task_id": sample.get("task_id", ""),
            "step_number": sample.get("step_number", 0),
            "trajectory_id": metadata.get("trajectory_id", sample.get("trajectory_id", sample.get("task_id", ""))),
            "action_type": metadata.get("action_type", ""),
        }

        if images:
            example["multi_modal_data"] = {"images": images}

        # ground_truth
        ground_truth = sample.get("ground_truth", {})
        if isinstance(ground_truth, dict):
            # image size
            if image_size:
                ground_truth["image_size"] = list(image_size)
            example["ground_truth"] = json.dumps(ground_truth)
        elif isinstance(ground_truth, str):
            example["ground_truth"] = ground_truth
        else:
            example["ground_truth"] = ""

        # reference assistant response (for logging)
        example["raw_response"] = sample.get("raw_response", "")

        return example
