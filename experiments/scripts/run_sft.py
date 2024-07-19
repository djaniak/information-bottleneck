from pathlib import Path
from typing import Optional

import torch
import typer

from datasets import load_dataset
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    QuantoConfig,
)
from peft import LoraConfig, prepare_model_for_kbit_training


def main(
    model_name: str = typer.Option(...),
    dataset_name: str = typer.Option(...),
    quantization: str = typer.Option(...),
    output_dir: Optional[Path] = typer.Option(None),
):
    if output_dir is None:
        output_dir = Path(f"data/sft/{dataset_name}/{model_name}-{quantization}-lora")
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = model_name.replace("_", "/")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=get_quantization_config(quantization),
    )
    model = prepare_model_for_kbit_training(model)

    dataset, trainer_kwargs, sft_config_kwargs = get_dataset(dataset_name, tokenizer)

    # Define SFT configuration
    sft_config = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,
        optim="paged_adamw_32bit",
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        logging_steps=50,
        num_train_epochs=3,
        max_steps=1500,
        save_strategy="steps",
        save_steps=250,
        fp16=True,
        **sft_config_kwargs,
    )

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )   
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=sft_config,
        max_seq_length=512,
        peft_config=peft_config,
        **trainer_kwargs,
    )

    trainer.train()


def get_quantization_config(quantization: str):
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
    else:
        return None



def get_dataset(dataset_name, tokenizer=None):
    trainer_kwargs = {}
    sft_config_kwargs = {}
    print(dataset_name)

    if dataset_name == "imdb":
        dataset = load_dataset("imdb", split="train")
        sft_config_kwargs = {"dataset_text_field": "text"}

    elif dataset_name == "CodeAlpaca-20k":
        assert tokenizer is not None

        dataset = load_dataset("lucasmccabe-lmi/CodeAlpaca-20k", split="train")

        # Define a function to format the prompts
        def formatting_prompts_func(example):
            output_texts = []
            for i in range(len(example["instruction"])):
                text = f"### Question: {example['instruction'][i]}\n ### Answer: {example['output'][i]}"
                output_texts.append(text)
            return output_texts

        # Define response template and data collator
        response_template = " ### Answer:"
        collator = DataCollatorForCompletionOnlyLM(
            response_template, tokenizer=tokenizer
        )

        trainer_kwargs = {
            "data_collator": collator,
            "formatting_func": formatting_prompts_func,
        }

    else:
        raise ValueError(f"Unknown dataset {dataset_name}")

    return dataset, trainer_kwargs, sft_config_kwargs


if __name__ == "__main__":
    typer.run(main)
