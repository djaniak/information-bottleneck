import json
import logging
from pathlib import Path
from typing import Optional


from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

import numpy as np
import torch
import torch.nn as nn
import typer


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



# os.makedirs(quantized_model_dir, exist_ok=True)
def get_wikitext2(nsamples, seed, seqlen, model):
    from datasets import load_dataset

    traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
    trainenc = tokenizer("\n\n".join(traindata["text"]), return_tensors="pt")
    testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")

    import random

    random.seed(seed)
    np.random.seed(0)
    torch.random.manual_seed(0)

    traindataset = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        attention_mask = torch.ones_like(inp)
        traindataset.append({"input_ids": inp, "attention_mask": attention_mask})
    return traindataset, testenc


@torch.no_grad()
def opt_eval(model, testenc, dev, seqlen=2048):
    print("Evaluating ...")

    testenc = testenc.input_ids
    nsamples = testenc.numel() // seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.decoder.layers

    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
    if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((nsamples, seqlen, model.config.hidden_size), dtype=dtype, device=dev)
    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * seqlen) : ((i + 1) * seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.cpu()
    if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]

    for i in range(len(layers)):
        # print(i)
        layer = layers[i].to(dev)

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.decoder.final_layer_norm is not None:
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(dev)
    if model.model.decoder.project_out is not None:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.decoder.final_layer_norm is not None:
            hidden_states = model.model.decoder.final_layer_norm(hidden_states)
        if model.model.decoder.project_out is not None:
            hidden_states = model.model.decoder.project_out(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, (i * seqlen) : ((i + 1) * seqlen)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))
    print(ppl.item())

    model.config.use_cache = use_cache

    return ppl.item()


def model_size_in_bytes(model):
    """
    Calculate the size of the model in bytes.
    
    Args:
    model: The PyTorch model.
    
    Returns:
    int: Size of the model in bytes.
    """
    total_size = 0
    for param in model.parameters():
        total_size += param.numel() * param.element_size()
    return total_size


def main(
    pretrained_model_dir: str = typer.Option(...),
    bits: int = typer.Option(...),
    output_path: Optional[Path] = typer.Option(None),
):
    if output_path is None:
        output_path = Path(f"data/perplexities/gptq-wikitext2/{pretrained_model_dir}-{bits}bit.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    quantized_model_dir = f"{pretrained_model_dir}-gptq-{bits}bit-128g-wikitext2"
    pretrained_model_dir = pretrained_model_dir.replace("_", "/")

    traindataset, testenc = get_wikitext2(128, 0, 2048, pretrained_model_dir)
    
    if bits < 16:
        quantize_config = BaseQuantizeConfig(
            bits=bits, 
            group_size=128,  # it is recommended to set the value to 128
            desc_act=False,  # desc_act and group size only works on triton
        )
        # load un-quantized model, the model will always be force loaded into cpu
        model = AutoGPTQForCausalLM.from_pretrained(pretrained_model_dir, quantize_config)

        # quantize model, the examples should be list of dict whose keys can only be "input_ids" and "attention_mask"
        # with value under torch.LongTensor type.
        model.quantize(traindataset, use_triton=False)

        model.save_quantized(quantized_model_dir)
        model.save_quantized(quantized_model_dir, use_safetensors=True)

        model = AutoGPTQForCausalLM.from_quantized(quantized_model_dir, device_map="auto", use_triton=False)
        model_size = model_size_in_bytes(model.model)
        perplexity = opt_eval(model.model, testenc, "cuda:0")
    
    else:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(pretrained_model_dir, device_map="auto", torch_dtype=torch.float16)
        model_size = model_size_in_bytes(model)
        perplexity = opt_eval(model, testenc, "cuda:0")


    with open(output_path, "w") as f:
        json.dump({"model": pretrained_model_dir, "size": model_size, "ppl": perplexity, "quantization": "gptq", "bits": bits, "dataset": "wikitext2"}, f)


if __name__ == "__main__":
    typer.run(main)
