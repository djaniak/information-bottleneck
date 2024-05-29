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
        R = R/norms
    return R


def compute_entropies_for_each_sentence(model: torch.nn.Module, dataloader: DataLoader, device: torch.device, alpha: float = 1) -> List[float]:
    """
    Computes the entropies for each sentence in the given dataloader using the provided model.

    Args:
        model (torch.nn.Module): The model used for computing the entropies.
        dataloader (DataLoader): The dataloader containing the sentences.
        device (torch.device): The device to perform the computations on.
        alpha (float, optional): The alpha parameter for the entropy calculation. Defaults to 1.

    Returns:
        List[float]: A list of entropies for each sentence in the dataloader.
    """

    entropy_list = []
    
    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            hidden_states = outputs.hidden_states
            N, D = hidden_states[0].shape[1:]

            last_hidden_state = normalize(hidden_states[-1].squeeze()).float()

            # be efficient here, XX^T and X^TX have the same eigenvalues and thus the same entropy
            if N > D:
                cov = last_hidden_state.T @ last_hidden_state
            else:
                cov = (last_hidden_state @ last_hidden_state.T)
            cov /= torch.trace(cov)

            entropy = itl.matrixAlphaEntropy(cov, alpha=alpha)

            # the matrix LLM paper [5] does this in equation 5, not sure it is strictly necessary
            entropy /= math.log(D) 
            entropy_list.append(entropy.item())

    return entropy_list
