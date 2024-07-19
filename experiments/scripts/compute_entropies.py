import json
import logging
from pathlib import Path
from typing import Optional

import torch
import typer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    QuantoConfig,
    GPTQConfig
)

from src.data import get_dataloader
from src.matrix_entropy import compute_entropies_for_each_sentence

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(
    model_name: str = typer.Option(...),
    dataset: str = typer.Option(...),
    quantization: str = typer.Option("none"),
    output_path: Optional[Path] = typer.Option(None),
    num_samples: int = typer.Option(25000),
    split: str = typer.Option("train"),
):
    assert split in ["train", "validation", "test"]
    if output_path is None:
        output_path = Path(f"data/entropies/{model_name}-{quantization}-{dataset}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = model_name.replace("_", "/")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    quantization_config = get_quantization_config(quantization, tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        output_hidden_states=True,
        device_map="auto",
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
    )
    dataloader = get_dataloader(
        tokenizer, dataset, split=split, num_samples=num_samples
    )
    entropies = compute_entropies_for_each_sentence(
        model, dataloader, alpha=1, device=device
    )

    with open(output_path, "w") as f:
        json.dump(entropies, f)
    logger.info(f"Entropies saved to {output_path}")


def get_quantization_config(quantization: str, tokenizer = None):
    framework, bits = quantization.split("-")
    if framework == "quanto":
        assert bits in ["float8", "int8", "int4", "int2"]
        return QuantoConfig(weights=bits)
    elif framework == "bnb":
        assert bits in ["int8", "int4"]
        if bits == "int8":
            return BitsAndBytesConfig(load_in_8bit=True)
        elif bits == "int4":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
    elif framework == "gptq":
        assert bits in ["2", "3", "4", "8"]
        assert tokenizer is not None
        return GPTQConfig(bits=int(bits), dataset="wikitext2", tokenizer=tokenizer)
    else:
        return None


if __name__ == "__main__":
    typer.run(main)
