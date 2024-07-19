import json
import logging
from pathlib import Path
from typing import Optional
from tqdm import tqdm
import torch
import typer
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM

from src.data import get_dataloader
from src.matrix_entropy import compute_entropies_for_each_sentence

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(
    dataset: str = typer.Option(...),
    model_checkpoint_dir: Optional[Path] = typer.Option(None),
    output_dir: Optional[Path] = typer.Option(None),
    num_samples: int = typer.Option(25000),
):
    if output_dir is None:
        output_dir = Path(f"data/entropies_sft/{dataset}/{model_checkpoint_dir.name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # iterate over checkpoints (there are checkpoints every 500 steps)
    for checkpoint_dir in tqdm(model_checkpoint_dir.iterdir()):
        for split in ["train", "test"]:
            tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
            model = AutoPeftModelForCausalLM.from_pretrained(checkpoint_dir, torch_dtype=torch.float16,output_hidden_states=True, device_map="auto")
            dataloader = get_dataloader(
                tokenizer, dataset, split=split, num_samples=num_samples
            )
            entropies = compute_entropies_for_each_sentence(
                model, dataloader, alpha=1, device=device
            )
            checkpoint_output_dir = output_dir / f"{checkpoint_dir.name}-{split}.json"
            with open(checkpoint_output_dir, "w") as f:
                json.dump(entropies, f)
            logger.info(f"Entropies saved to {checkpoint_output_dir}")


if __name__ == "__main__":
    typer.run(main)
