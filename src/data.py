import math
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch


ALLOWED_DATASETS = ['wikitext', 'dolly15k']

def get_dataloader(tokenizer, dataset_name, split='train', context_length_ratio=1, min_length=5, max_length=None, num_samples=20000):
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=2048)

    def adjust_context_length(examples):
        if context_length_ratio == 1:
            return examples
        else:
            input_length = len(examples['input_ids'])
            context_length = max(2, int(input_length * context_length_ratio))
            examples['attention_mask'] = examples['attention_mask'][:context_length]
            examples['input_ids'] = examples['input_ids'][:context_length]

            return examples

    def is_not_wikipedia_heading(example):
        return not (example["text"].strip().startswith("=") and example["text"].strip().endswith("="))

    assert dataset_name in ALLOWED_DATASETS
    assert split in ['train', 'validation']
    assert context_length_ratio <= 1

    if dataset_name == 'wikitext':
        dataset = load_dataset("wikitext", 'wikitext-103-v1')[split]
        num_samples = min(num_samples, len(dataset))
        dataset = dataset.select(range(num_samples))
    elif dataset_name == 'dolly15k':
        dataset = load_dataset('databricks/databricks-dolly-15k')[split]
        num_samples = min(num_samples, len(dataset))
        dataset = dataset.select(range(num_samples))

    tokenized_dataset = dataset.map(tokenize_function, batched=True).shuffle(seed=42)
    tokenized_dataset.set_format("torch")

    tokenized_dataset = tokenized_dataset.filter(lambda x: len(x['input_ids']) >= min_length) # filter out the frequent blank/small examples in the dataset
    tokenized_dataset = tokenized_dataset.filter(is_not_wikipedia_heading) # filter out headings
    tokenized_dataset = tokenized_dataset.remove_columns(["text"])

    if max_length is not None:
        tokenized_dataset = tokenized_dataset.filter(lambda x: len(x['input_ids']) <= max_length)

    tokenized_dataset = tokenized_dataset.map(adjust_context_length, batched=False)

    dataloader = DataLoader(tokenized_dataset, shuffle=False, drop_last=True) # something is weird with batch_size=x argument here, removing it for now
    return dataloader


# from https://github.com/waltonfuture/Matrix-Entropy
def normalize(R):
    with torch.no_grad():
        mean = R.mean(dim=0)
        R = R - mean
        norms = torch.norm(R, p=2, dim=1, keepdim=True)
        R = R/norms
    return R
