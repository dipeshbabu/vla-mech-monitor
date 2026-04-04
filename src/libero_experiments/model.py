"""Model loading and action inference."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

from libero_experiments.utils import DEVICE, OPENVLA_V01_SYSTEM_PROMPT


ACTION_DIM = 7
LANCZOS = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS


def load_model(cfg: Any):
    # Try to use flash_attention_2, but fallback gracefully if unavailable or incompatible
    attn_implementation = "flash_attention_2"
    try:
        # Check if flash-attn is available and compatible
        import flash_attn  # noqa: F401

        # Try to load with flash_attention_2 first
        try:
            model = AutoModelForVision2Seq.from_pretrained(
                cfg.model.checkpoint,
                attn_implementation="flash_attention_2",
                torch_dtype=torch.bfloat16,
                load_in_8bit=cfg.model.load_in_8bit,
                load_in_4bit=cfg.model.load_in_4bit,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
            print("Successfully loaded model with flash_attention_2")
        except (ValueError, RuntimeError, ImportError) as e:
            # If flash_attention_2 fails, fallback to sdpa (scaled dot product attention)
            print(f"Warning: flash_attention_2 failed ({e}), falling back to sdpa")
            attn_implementation = "sdpa"
            model = AutoModelForVision2Seq.from_pretrained(
                cfg.model.checkpoint,
                attn_implementation="sdpa",
                torch_dtype=torch.bfloat16,
                load_in_8bit=cfg.model.load_in_8bit,
                load_in_4bit=cfg.model.load_in_4bit,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
    except ImportError:
        # flash-attn not installed, use sdpa
        print("Warning: flash-attn not available, using sdpa attention")
        attn_implementation = "sdpa"
        model = AutoModelForVision2Seq.from_pretrained(
            cfg.model.checkpoint,
            attn_implementation="sdpa",
            torch_dtype=torch.bfloat16,
            load_in_8bit=cfg.model.load_in_8bit,
            load_in_4bit=cfg.model.load_in_4bit,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )

    if not cfg.model.load_in_8bit and not cfg.model.load_in_4bit:
        model = model.to(DEVICE)

    dataset_statistics_path = Path(cfg.model.checkpoint) / "dataset_statistics.json"
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    if dataset_statistics_path.exists():
        with dataset_statistics_path.open("r", encoding="utf-8") as f:
            norm_stats = json.load(f)
        model.norm_stats = norm_stats
    else:
        print(
            "WARNING: No local dataset_statistics.json file found for current checkpoint.\n"
            "You can ignore this if you are loading the base VLA checkpoint. "
            "Otherwise, you may run into errors when calling predict_action due to a missing unnorm key."
        )

    print(f"Loaded model: {type(model)}")
    return model


def get_processor(cfg: Any):
    return AutoProcessor.from_pretrained(cfg.model.checkpoint, trust_remote_code=True)


def crop_and_resize(image: Image.Image, crop_scale: float, size: tuple[int, int] = (224, 224)) -> Image.Image:
    width, height = image.size
    scale = float(np.clip(np.sqrt(crop_scale), 0.0, 1.0))
    crop_width = max(1, int(round(width * scale)))
    crop_height = max(1, int(round(height * scale)))
    left = max(0, (width - crop_width) // 2)
    top = max(0, (height - crop_height) // 2)
    return image.crop((left, top, left + crop_width, top + crop_height)).resize(size, LANCZOS)


def _build_prompt(task_label: str, checkpoint: str) -> str:
    if "openvla-v01" in checkpoint:
        return f"{OPENVLA_V01_SYSTEM_PROMPT} USER: What action should the robot take to {task_label.lower()}? ASSISTANT:"
    return f"In: What action should the robot take to {task_label.lower()}?\nOut:"


def get_action(model, processor, cfg: Any, obs: dict, task_label: str, unnorm_key: str) -> np.ndarray:
    image = Image.fromarray(obs["full_image"]).convert("RGB")

    if cfg.model.center_crop:
        image = crop_and_resize(image, crop_scale=0.9)

    prompt = _build_prompt(task_label, cfg.model.checkpoint)
    inputs = processor(prompt, image).to(DEVICE, dtype=torch.bfloat16)
    action = model.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
    assert action.shape == (ACTION_DIM,)
    return action
