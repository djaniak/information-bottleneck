from typing import Any
import matplotlib.pyplot as plt 
import numpy as np
from collections import defaultdict


def get_metric(data: dict[str, Any], metric: str, hidden_state: str | int | None):
    if metric == "perplexity":
        return data["perplexity"]
    else:
        assert hidden_state is not None
        if isinstance(hidden_state, int):
            _hidden_state = f"hidden_state_{hidden_state}"
        elif isinstance(hidden_state, str):
            _hidden_state = hidden_state
        elif hidden_state == "last":
            _hidden_state = list(data["unnormalized_entropy"].keys())[-1]
        else:
            raise ValueError(f"Unknown hidden state: {hidden_state}")

        entropy = data["unnormalized_entropy"][_hidden_state]
        if metric == "unnormalized_entropy":
            return entropy
        elif metric == "logN_normalized_entropy":
            return entropy / np.log(data["num_words"])
        elif metric == "logD_normalized_entropy":
            return entropy / np.log(data["embedding_dim"])
        elif metric == "logNlogD_normalized_entropy":
            return entropy / (np.log(data["num_words"])*np.log(data["embedding_dim"]))
        else:
            raise ValueError(f"Unknown metric: {metric}")


def plot_all_entropies(entropies: dict[str, Any], model_sizes: list[str], hidden_state: int | str = "last", framework: str = "quanto", allow_2bit: bool = False):
    if framework in ["quanto", "bnb"]:
        colors = {
            "int2": "red",
            "int4": "orange",
            "int8": "green",
            "float16": "blue"
        }
    elif framework == "gptq":
        colors = {
            "2": "red",
            "3": "orange",
            "4": "yellow",
            "8": "green",
            "float16": "blue"
        }

    fig, axs = plt.subplots(len(model_sizes), 4, figsize=(12, 12))

    for i, model in enumerate(model_sizes):
        _entropies = entropies[model]
        model_size = model.split("-")[-1]
        for quantization, __entropies in _entropies.items():
            quantization_framework, quantization_type = quantization.split("-")
            for j, metric in enumerate(["unnormalized_entropy", "logN_normalized_entropy", "logD_normalized_entropy", "logNlogD_normalized_entropy"]):
                if quantization_framework != "none" and quantization_framework != framework:
                    continue
                if quantization_type in ["int2", "2"] and not allow_2bit:
                    continue
                
                axs[i, j].hist(get_metric(__entropies, metric, hidden_state), bins=50, density=True, label=quantization_type, alpha=0.5, color=colors[quantization_type])
                axs[i, j].set(title=f"{metric} - {model_size}", xlabel=metric, ylabel="Density")
                axs[i, j].title.set_fontsize(10)  # Set smaller text size for title

                handles, labels = axs[i, j].get_legend_handles_labels()
                labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
                axs[i, j].legend(handles, labels)

    plt.tight_layout()
    plt.show()


def plot_mean_entropies(entropies: dict[str, Any], hidden_state: int | str = "last", framework: str = "quanto", allow_2bit: bool = False):
    metrics = ["unnormalized_entropy", "logN_normalized_entropy", "logD_normalized_entropy", "logNlogD_normalized_entropy"]
    model_sizes = [k.split("-")[-1] for k in entropies.keys()]
    mean_metrics = defaultdict(list)

    for metric in metrics:
        mean_metrics[metric] = defaultdict(list)
        for model, _entropies in entropies.items():
            model_size = model.split("-")[-1]
            for quantization, __entropies in _entropies.items():
                mean_metrics[metric][quantization].append(np.mean(get_metric(__entropies, metric, hidden_state)))

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    flatten_axs = axs.flatten()
    for i, metric in enumerate(metrics):
        for quantization, mean_entropies in mean_metrics[metric].items():
            quantization_framework, quantization_type = quantization.split("-")
            if quantization_framework != "none" and quantization_framework != framework:
                continue
            if quantization_type == "int2" and not allow_2bit:
                continue
            flatten_axs[i].plot(model_sizes, mean_entropies, label=quantization_type)
            flatten_axs[i].set(xlabel="# of model params", ylabel=metric, title=f"Mean {metric}")
            flatten_axs[i].title.set_fontsize(10)  # Set smaller text size for title

            handles, labels = flatten_axs[i].get_legend_handles_labels()
            labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
            flatten_axs[i].legend(handles, labels)

    plt.tight_layout()
    plt.show()


def plot_mean_perplexity(entropies: dict[str, Any], model_sizes: list[str], framework: str = "quanto"):
    model_sizes = [k.split("-")[-1] for k in entropies.keys()]
    mean_perplexity = defaultdict(list)

    for model in model_sizes:
        _entropies = entropies[model]
        for quantization, __entropies in _entropies.items():
            quantization_framework, quantization_type = quantization.split("-")
            if quantization_framework != "none" and (quantization_framework != framework or quantization_type == "int2"):
                continue
            mean_perplexity[quantization].append(np.mean(get_metric(__entropies, "perplexity", hidden_state=None)))

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    for quantization, mean_entropies in mean_perplexity.items():
        ax.plot(model_sizes, mean_entropies, label=quantization)
        ax.set(xlabel="# of model params", ylabel="perplexity", title="Mean perplexity", yscale="log")

        handles, labels = ax.get_legend_handles_labels()
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
        ax.legend(handles, labels)

    plt.tight_layout()
    plt.show()
