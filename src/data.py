from datasets import load_dataset
from torch.utils.data import DataLoader


def get_dataloader(tokenizer, dataset_name, split='train'):
    assert split in ['train', 'validation']

    if dataset_name == 'wikitext':
        dataset = load_dataset("wikitext", 'wikitext-103-v1')[split]
        num_samples = min(10000, len(dataset))
        dataset = dataset.select(range(num_samples))
    else:
        raise ValueError("Invalid dataset name")

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=2048)

    tokenized_dataset = dataset.map(tokenize_function, batched=True).shuffle(seed=42)
    tokenized_dataset.set_format("torch")
    #example_texts = tokenized_dataset["text"] # use this to see the raw text, otherwise its removed

    tokenized_dataset = tokenized_dataset.remove_columns(["text"])
    # filter out the frequent blank/small examples in the dataset
    tokenized_dataset = tokenized_dataset.filter(lambda x: len(x['input_ids']) > 5) 

    # something is weird with batch_size=x argument here, removing it for now
    dataloader = DataLoader(tokenized_dataset, shuffle=False, drop_last=True)
    return dataloader
