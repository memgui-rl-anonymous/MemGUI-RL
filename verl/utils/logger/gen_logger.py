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


import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

from ..py_functional import is_package_available


if is_package_available("wandb"):
    import wandb  # type: ignore


if is_package_available("swanlab"):
    import swanlab  # type: ignore


@dataclass
class GenerationLogger(ABC):
    config: dict[str, Any]

    @abstractmethod
    def log(self, samples: List[Tuple[str, str, str, float]], step: int) -> None: ...

    def _max_chars(self) -> int:
        trainer_config = (self.config or {}).get("trainer", {})
        return int(trainer_config.get("generation_log_max_chars", 4000))

    def _clip(self, value: Any) -> str:
        text = "" if value is None else str(value)
        max_chars = self._max_chars()
        if max_chars <= 0 or len(text) <= max_chars:
            return text
        head = max_chars // 2
        tail = max_chars - head
        return f"{text[:head]}\n...[truncated {len(text) - max_chars} chars]...\n{text[-tail:]}"

    def _clip_sample(self, sample: Tuple[str, str, str, float]) -> Tuple[str, str, str, float]:
        inp, out, lab, score = sample
        return self._clip(inp), self._clip(out), self._clip(lab), score


@dataclass
class ConsoleGenerationLogger(GenerationLogger):
    def log(self, samples: List[Tuple[str, str, str, float]], step: int) -> None:
        for inp, out, lab, score in samples:
            inp, out, lab, score = self._clip_sample((inp, out, lab, score))
            print(f"[prompt] {inp}\n[output] {out}\n[ground_truth] {lab}\n[score] {score}\n")


@dataclass
class FileGenerationLogger(GenerationLogger):
    def log(self, samples: List[Tuple[str, str, str, float]], step: int) -> None:
        with open(os.path.join(self.config["trainer"]["save_checkpoint_path"], "generations.log"), "a") as f:
            for inp, out, lab, score in samples:
                inp, out, lab, score = self._clip_sample((inp, out, lab, score))
                f.write(f"[prompt] {inp}\n[output] {out}\n[ground_truth] {lab}\n[score] {score}\n\n")


@dataclass
class WandbGenerationLogger(GenerationLogger):
    def log(self, samples: List[Tuple[str, str, str, float]], step: int) -> None:
        trainer_config = (self.config or {}).get("trainer", {})
        if not trainer_config.get("wandb_log_generations", False):
            return

        columns = ["step", "sample_idx", "input", "output", "label", "score"]
        rows = []
        for sample_idx, sample in enumerate(samples):
            inp, out, lab, score = self._clip_sample(sample)
            rows.append([step, sample_idx, inp, out, lab, score])

        # Log only the current validation step. Do not carry historical rows
        # forward, otherwise wandb creates a growing table artifact each time.
        wandb.log({"val/generations": wandb.Table(columns=columns, data=rows)}, step=step)


@dataclass
class SwanlabGenerationLogger(GenerationLogger):
    def log(self, samples: List[Tuple[str, str, str, float]], step: int) -> None:
        swanlab_text_list = []
        for i, sample in enumerate(samples):
            sample = self._clip_sample(sample)
            row_text = "\n\n---\n\n".join(
                (f"input: {sample[0]}", f"output: {sample[1]}", f"label: {sample[2]}", f"score: {sample[3]}")
            )
            swanlab_text_list.append(swanlab.Text(row_text, caption=f"sample {i + 1}"))

        swanlab.log({"val/generations": swanlab_text_list}, step=step)


GEN_LOGGERS = {
    "console": ConsoleGenerationLogger,
    "file": FileGenerationLogger,
    "wandb": WandbGenerationLogger,
    "swanlab": SwanlabGenerationLogger,
}


class AggregateGenerationsLogger:
    def __init__(self, loggers: List[str], config: Optional[dict[str, Any]] = None):
        self.loggers: List[GenerationLogger] = []

        for logger in loggers:
            if logger == "wandb" and not (config or {}).get("trainer", {}).get("wandb_log_generations", False):
                continue
            if logger in GEN_LOGGERS:
                self.loggers.append(GEN_LOGGERS[logger](config))

    def log(self, samples: List[Tuple[str, str, str, float]], step: int) -> None:
        for logger in self.loggers:
            logger.log(samples, step)
