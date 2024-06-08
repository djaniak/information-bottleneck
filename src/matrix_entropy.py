import math
from typing import List

import repitl.matrix_itl as itl
import torch
import tqdm
from torch.utils.data import DataLoader


def normalize(R):
    """
    Normalize the input matrix by subtracting the mean and dividing by the L2 norm.
    From https://github.com/waltonfuture/Matrix-Entropy

    Args:
        R (torch.Tensor): Input matrix to be normalized.

    Returns:
        torch.Tensor: Normalized matrix.

    """
    with torch.no_grad():
        mean = R.mean(dim=0)
        R = R - mean
        norms = torch.norm(R, p=2, dim=1, keepdim=True)
        R = R / norms
    return R


def compute_entropies_for_each_sentence(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    alpha: float = 1,
) -> List[float]:
    output_dict = {
        "unnormalized_entropy": [],
        "perplexity": [],
        "lengths": [],
        "dimensions": [],
    }

    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            batch["labels"] = batch["input_ids"].clone()

            outputs = model(**batch)
            hidden_states = outputs.hidden_states
            loss = outputs.loss
            N, D = hidden_states[0].shape[1:]

            last_hidden_state = normalize(outputs.hidden_states[-1].squeeze())
            # (batch_size, num_words, embedding_dim)

            # be efficient here, XX^T and X^TX have the same eigenvalues and thus the same entropy
            if N > D:
                cov = last_hidden_state.T @ last_hidden_state
            else:
                cov = last_hidden_state @ last_hidden_state.T
            cov /= torch.trace(cov)

            entropy = itl.matrixAlphaEntropy(cov.float(), alpha=alpha)
            output_dict["unnormalized_entropy"].append(entropy.item())
            output_dict["perplexity"].append(math.exp(loss.item()))
            output_dict["lengths"].append(N)
            output_dict["dimensions"].append(D)

    return output_dict
