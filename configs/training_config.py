"""
Configuration dataclasses for MedGemma QLoRA fine-tuning on MIMIC-CXR.
"""
from dataclasses import dataclass, field
from typing import Optional, List

import torch


def _detect_dtype() -> str:
    """Return the best compute dtype string for the current GPU."""
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability()
        if cap[0] >= 8:
            return "bfloat16"
    return "float16"


@dataclass
class ModelConfig:
    model_id: str = "google/medgemma-4b-it"

    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True

    bnb_4bit_compute_dtype: str = field(default_factory=_detect_dtype)


@dataclass
class LoraConfig:
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    bias: str = "none"

    target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
        ]
    )

    task_type: str = "CAUSAL_LM"


@dataclass
class DataConfig:
    mimic_cxr_jpg_dir: str = "data/mimic-cxr-jpg"
    mimic_cxr_dir: str = "data/mimic-cxr"
    processed_dir: str = "data/processed"

    max_train_samples: Optional[int] = None
    max_val_samples: Optional[int] = None

    max_length: int = 512

    prompt_template: str = (
        "Describe the findings in this chest X-ray image "
        "and provide a clinical impression."
    )


@dataclass
class TrainingConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)
    data: DataConfig = field(default_factory=DataConfig)

    output_dir: str = "output"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8

    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"

    gradient_checkpointing: bool = True
    optim: str = "paged_adamw_8bit"

    logging_steps: int = 10
    save_steps: int = 200
    eval_steps: int = 200
    save_total_limit: int = 3

    seed: int = 42
    dataloader_num_workers: int = 2
    dataloader_pin_memory: bool = True

    report_to: str = "none"

    @property
    def use_bf16(self) -> bool:
        return self.model.bnb_4bit_compute_dtype == "bfloat16"

    @property
    def use_fp16(self) -> bool:
        return self.model.bnb_4bit_compute_dtype == "float16"
