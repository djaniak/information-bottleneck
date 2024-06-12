import os
from pathlib import Path
from typing import Optional

import torch
import typer


def main(
    model_name: str = typer.Option(...),
    task: str = typer.Option(...),
    quantization: str = typer.Option(...),
    output_path: Optional[Path] = typer.Option(None),
    batch_size: int = typer.Option(256),
):
    if output_path is None:
        output_path = Path(f"data/lm_eval/{model_name}/{quantization}/{task}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = model_name.replace("_", "/")

    model_args = f"pretrained={model_name}"
    if quantization == "int8":
        model_args += ",load_in_8bit=True"
    elif quantization == "int4":
        model_args += ",load_in_4bit=True"
    else:
        pass

    os.system(
        f"lm_eval --model hf --model_args {model_args} --tasks {task} --device {device} --output_path {output_path} --batch_size {batch_size}"
    )


if __name__ == "__main__":
    typer.run(main)
