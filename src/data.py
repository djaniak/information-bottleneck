import math

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader


def get_dataloader(
    tokenizer,
    dataset_name,
    split="train",
    context_length_ratio=1,
    min_length=5,
    max_length=None,
    num_samples=20000,
):
    def tokenize_function(examples):
        return tokenizer(examples[target_column], truncation=True, max_length=2048)

    def adjust_context_length(examples):
        if context_length_ratio == 1:
            return examples
        else:
            input_length = len(examples["input_ids"])
            context_length = max(2, int(input_length * context_length_ratio))
            examples["attention_mask"] = examples["attention_mask"][:context_length]
            examples["input_ids"] = examples["input_ids"][:context_length]

            return examples

    def is_not_wikipedia_heading(example):
        return not (
            example[target_column].strip().startswith("=")
            and example[target_column].strip().endswith("=")
        )

    assert split in ["train", "validation", "test"]
    assert context_length_ratio <= 1

    if dataset_name == "wikitext":
        target_column = "text"
        remove_columns = ["text"]
        dataset = load_dataset("wikitext", "wikitext-103-v1")[split]
    elif dataset_name == "dolly15k":
        target_column = "context"
        remove_columns = ["instruction", "response", "category", "context"]
        dataset = load_dataset("databricks/databricks-dolly-15k")[split]
    elif dataset_name == "imdb":
        target_column = "text"
        remove_columns = ["text", "label"]
        dataset = load_dataset("imdb")[split]
    elif dataset_name == "openwebtext":
        assert split == "train"
        target_column = "text"
        remove_columns = ["text"]
        dataset = load_dataset("stas/openwebtext-10k")[split]
    elif dataset_name == "hh-rlhf":
        target_column = "chosen"
        remove_columns = ["chosen", "rejected"]
        dataset = load_dataset("Anthropic/hh-rlhf")[split]
    else:
        raise ValueError("Dataset not recognized!")
    
    if num_samples is not None:
        num_samples = min(num_samples, len(dataset))
        dataset = dataset.select(range(num_samples))

    tokenized_dataset = dataset.map(tokenize_function, batched=True).shuffle(seed=42)
    tokenized_dataset.set_format("torch")

    # filter out the frequent blank/small examples in the dataset
    tokenized_dataset = tokenized_dataset.filter(
        lambda x: len(x["input_ids"]) >= min_length
    )

    # filter out headings in wikipedia dataset
    if dataset_name == "wikitext":
        tokenized_dataset = tokenized_dataset.filter(is_not_wikipedia_heading)
    tokenized_dataset = tokenized_dataset.remove_columns(remove_columns)

    if max_length is not None:
        tokenized_dataset = tokenized_dataset.filter(
            lambda x: len(x["input_ids"]) <= max_length
        )

    tokenized_dataset = tokenized_dataset.map(adjust_context_length, batched=False)

    # something is weird with batch_size=x argument here, removing it for now
    dataloader = DataLoader(tokenized_dataset, shuffle=False, drop_last=True)
    return dataloader


# from https://github.com/waltonfuture/Matrix-Entropy
def normalize(R):
    with torch.no_grad():
        mean = R.mean(dim=0)
        R = R - mean
        norms = torch.norm(R, p=2, dim=1, keepdim=True)
        R = R / norms
    return R
