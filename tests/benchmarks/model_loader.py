"""Shared model loading infrastructure for PyQuifer benchmarks.

Provides a single entry point for loading LLMs (real or dummy) across all
benchmark scripts.  Env vars control model selection:

    PYQUIFER_LLM_MODEL   Model name/path (default: "" → DummyLLM)
    PYQUIFER_LLM_DEVICE  Device string   (default: "cpu")

When PYQUIFER_LLM_MODEL is unset the module returns lightweight DummyLLM /
DummyTokenizer instances so every benchmark can run without a GPU or model
download.

Usage:
    from model_loader import load_model, load_pyquifer_model, get_model_config

    config = get_model_config()
    model, tokenizer = load_model(config.model_name, config.device)
    # or with bridge:
    model, tokenizer, bridge = load_pyquifer_model(config.model_name, config.device)
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn

_benchmark_dir = Path(__file__).resolve().parent
_pyquifer_src = _benchmark_dir.parent.parent / "src"
if str(_pyquifer_src) not in sys.path:
    sys.path.insert(0, str(_pyquifer_src))


# ============================================================
# Configuration
# ============================================================

@dataclass
class ModelConfig:
    model_name: str = ""
    device: str = "cpu"
    quantize: str = "auto"  # "auto", "4bit", "8bit", "none"

    @property
    def has_real_model(self) -> bool:
        return bool(self.model_name)


def get_model_config() -> ModelConfig:
    """Read model config from environment variables."""
    return ModelConfig(
        model_name=os.environ.get("PYQUIFER_LLM_MODEL", ""),
        device=os.environ.get("PYQUIFER_LLM_DEVICE", "cpu"),
        quantize=os.environ.get("PYQUIFER_LLM_QUANTIZE", "auto"),
    )


# ============================================================
# Dummy Model (always available, no dependencies)
# ============================================================

class DummyTokenizer:
    """Minimal tokenizer stand-in for pipeline testing."""

    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        self.eos_token_id = 2
        self.pad_token_id = 0
        self.bos_token_id = 1

    def __call__(self, text, return_tensors="pt", **kwargs):
        if isinstance(text, str):
            text = [text]
        ids = [torch.randint(3, self.vocab_size, (min(len(t.split()) + 5, 64),))
               for t in text]
        max_len = max(t.shape[0] for t in ids)
        padded = torch.full((len(ids), max_len), self.pad_token_id, dtype=torch.long)
        mask = torch.zeros(len(ids), max_len, dtype=torch.long)
        for i, t in enumerate(ids):
            padded[i, :t.shape[0]] = t
            mask[i, :t.shape[0]] = 1
        return {"input_ids": padded, "attention_mask": mask}

    def decode(self, ids, skip_special_tokens=True):
        return f"<dummy output len={len(ids)}>"

    def batch_decode(self, batch_ids, skip_special_tokens=True):
        return [self.decode(ids) for ids in batch_ids]


class DummyLLM(nn.Module):
    """Minimal LLM stand-in: embedding -> linear -> logits."""

    def __init__(self, vocab_size: int = 32000, hidden_dim: int = 256):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.proj = nn.Linear(hidden_dim, vocab_size)
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.config = type("Config", (), {
            "hidden_size": hidden_dim,
            "vocab_size": vocab_size,
            "num_hidden_layers": 1,
        })()

    def forward(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        """Return logits for next token."""
        h = self.embed(input_ids).mean(dim=1)
        return self.proj(h)

    def get_hidden(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return hidden state (for PyQuifer bridge input)."""
        return self.embed(input_ids).mean(dim=1)

    def generate(self, input_ids, max_new_tokens=32, **kwargs):
        """Simple autoregressive generation with dummy model."""
        for _ in range(max_new_tokens):
            logits = self.forward(input_ids)
            next_token = logits.argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
        return input_ids


# ============================================================
# Real Model Loading
# ============================================================

def _should_quantize(quantize: str, device: str) -> Optional[str]:
    """Decide whether to quantize based on config and hardware."""
    if quantize == "none":
        return None
    if quantize in ("4bit", "8bit"):
        return quantize
    # "auto": quantize to 4bit if CUDA available
    if quantize == "auto" and "cuda" in device:
        try:
            import bitsandbytes  # noqa: F401
            return "4bit"
        except ImportError:
            pass
    return None


def load_model(
    model_name: Optional[str] = None,
    device: Optional[str] = None,
    quantize: str = "auto",
) -> Tuple:
    """Load HF model + tokenizer.  Returns (model, tokenizer).

    If model_name is empty/None, returns (DummyLLM(), DummyTokenizer()).
    """
    if not model_name:
        return DummyLLM(), DummyTokenizer()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = device or "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    load_kwargs = {"trust_remote_code": True}

    quant_mode = _should_quantize(quantize, device)
    if quant_mode == "4bit":
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        load_kwargs["device_map"] = "auto"
    elif quant_mode == "8bit":
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        load_kwargs["device_map"] = "auto"
    else:
        if "cuda" in device:
            load_kwargs["torch_dtype"] = torch.bfloat16
            load_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer


def load_pyquifer_model(
    model_name: Optional[str] = None,
    device: Optional[str] = None,
    quantize: str = "auto",
    bridge_size: str = "default",
) -> Tuple:
    """Load model + tokenizer + PyQuiferBridge.  Returns (model, tokenizer, bridge).

    bridge_size: "default" or "small"
    """
    model, tokenizer = load_model(model_name, device, quantize)

    from pyquifer.bridge import PyQuiferBridge
    if bridge_size == "small":
        bridge = PyQuiferBridge.small()
    else:
        bridge = PyQuiferBridge.default()

    bridge_device = device or "cpu"
    if hasattr(model, "device"):
        bridge_device = str(model.device)
    bridge = bridge.to(bridge_device)

    return model, tokenizer, bridge
