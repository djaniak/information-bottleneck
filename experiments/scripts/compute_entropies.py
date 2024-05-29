import json
import logging
from pathlib import Path
from typing import Optional

import torch
import typer
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data import get_dataloader
from src.matrix_entropy import compute_entropies_for_each_sentence

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(
    model_name: str = typer.Option(...),
    dataset: str = typer.Option(...),
    compression: str = typer.Option(...),
    output_path: Optional[Path] = typer.Option(None),
):
    if output_path is None:
        output_path = Path(f"data/entropies/{model_name}-{compression}-{dataset}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = model_name.replace("_", "/")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    compression_kwargs = {
        "load_in_8bit": compression == "quantization_8bit",
        "load_in_4bit": compression == "quantization_4bit",
    }
    model = AutoModelForCausalLM.from_pretrained(
        model_name, output_hidden_states=True, device_map="auto", **compression_kwargs
    )
    dataloaders = {
        "train": get_dataloader(tokenizer, dataset, split="train"),
        "validation": get_dataloader(tokenizer, dataset, split="validation"),
    }

    entropies = {
        split: compute_entropies_for_each_sentence(model, dataloader, alpha=1, device=device)
        for split, dataloader in dataloaders.items()
    }

    with open(output_path, "w") as f:
        json.dump(entropies, f)
    logger.info(f"Entropies saved to {output_path}")


if __name__ == "__main__":
    typer.run(main)
