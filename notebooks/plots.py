from typing import Any
import matplotlib.pyplot as plt 
import numpy as np
from collections import defaultdict


def get_metric(data: dict[str, Any], metric: str):
    if metric == "perplexity":
        return data["perplexity"]
    elif metric == "unnormalized_entropy":
        return data["unnormalized_entropy"]
    elif metric == "logN_normalized_entropy":
        return data["unnormalized_entropy"] / np.log(data["lengths"])
    elif metric == "logD_normalized_entropy":
        return data["unnormalized_entropy"] / np.log(data["dimensions"])
    elif metric == "logNlogD_normalized_entropy":
        return data["unnormalized_entropy"] / (np.log(data["lengths"])*np.log(data["dimensions"]))
    else:
        raise ValueError(f"Unknown metric: {metric}")


def plot_all_entropies(entropies: dict[str, Any], model_sizes: list[str|int], framework: str = "quanto", allow_2bit: bool = False):
    if framework == "quanto":
        colors = {
            "int2": "red",
            "int4": "orange",
            "int8": "green",
            "none": "blue"
        }
    elif framework == "bnb":
        colors = {
            "quantization_4bit": "orange",
            "quantization_8bit": "green",
            "none": "blue"
        }

    fig, axs = plt.subplots(len(model_sizes), 4, figsize=(12, 12))

    for i, _size in enumerate(model_sizes):
        _entropies = {k: entropies[k] for k in entropies.keys() if _size in k}
        for _, (_model, _entropy_data) in enumerate(_entropies.items()):
            for j, metric in enumerate(["unnormalized_entropy", "logN_normalized_entropy", "logD_normalized_entropy", "logNlogD_normalized_entropy"]):
                
                quantization = _model.split("-")[-1]
                if quantization == "int2" and not allow_2bit:
                    continue
                
                axs[i, j].hist(get_metric(_entropy_data, metric), bins=50, density=True, label=quantization, alpha=0.5, color=colors[quantization])
                axs[i, j].set(title=f"{metric} - {_size}", xlabel=metric, ylabel="Density")
                axs[i, j].title.set_fontsize(10)  # Set smaller text size for title

                handles, labels = axs[i, j].get_legend_handles_labels()
                labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
                axs[i, j].legend(handles, labels)

    plt.tight_layout()
    plt.show()


def plot_mean_entropies(entropies: dict[str, Any], model_sizes: list[str|int], allow_2bit: bool = False):
    metrics = ["unnormalized_entropy", "logN_normalized_entropy", "logD_normalized_entropy", "logNlogD_normalized_entropy"]
    mean_metrics = defaultdict(list)

    for metric in metrics:
        mean_metrics[metric] = defaultdict(list)

        for i, model_size in enumerate(model_sizes):
            model_metrics = {k: entropies[k] for k in entropies.keys() if model_size in k}
            for model_name, model_metric_v in model_metrics.items():
                quantization = model_name.split("-")[-1]
                mean_metrics[metric][quantization].append(np.mean(get_metric(model_metric_v, metric)))

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    flatten_axs = axs.flatten()
    for i, metric in enumerate(metrics):
        for quantization, mean_entropies in mean_metrics[metric].items():
            if quantization == "int2" and metric == "perplexity":
                continue
            flatten_axs[i].plot(model_sizes, mean_entropies, label=quantization)
            flatten_axs[i].set(xlabel="# of model params", ylabel=metric, title=f"Mean {metric}")
            flatten_axs[i].title.set_fontsize(10)  # Set smaller text size for title

            handles, labels = flatten_axs[i].get_legend_handles_labels()
            labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
            flatten_axs[i].legend(handles, labels)

    plt.tight_layout()
    plt.show()


def plot_mean_perplexity(entropies: dict[str, Any], model_sizes: list[str|int]):
    metric = "perplexity"
    mean_perplexity = defaultdict(list)

    for i, model_size in enumerate(model_sizes):
        model_metrics = {k: entropies[k] for k in entropies.keys() if model_size in k}
        for model_name, model_metric_v in model_metrics.items():
            quantization = model_name.split("-")[-1]
            if quantization == "int2":
                continue
            mean_perplexity[quantization].append(np.mean(get_metric(model_metric_v, metric)))

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    for quantization, mean_entropies in mean_perplexity.items():
        ax.plot(model_sizes, mean_entropies, label=quantization)
        ax.set(xlabel="# of model params", ylabel=metric, title=f"Mean {metric}")

        handles, labels = ax.get_legend_handles_labels()
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
        ax.legend(handles, labels)

    plt.tight_layout()
    plt.show()
