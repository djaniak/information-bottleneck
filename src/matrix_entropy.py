import math
from collections import defaultdict
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
) -> list[float]:
    
    output_dict = {
        "unnormalized_entropy": defaultdict(list),
        "perplexity": [],
        "num_words": [],
        "embedding_dim": [],
    }

    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            batch["labels"] = batch["input_ids"].clone()

            outputs = model(**batch)
            hidden_states = outputs.hidden_states
            loss = outputs.loss
            # (batch_size, num_words, embedding_dim)
            N, D = hidden_states[0].shape[1:]
            for i, hidden_state in enumerate(hidden_states):
                hidden_state = normalize(hidden_state.squeeze())
                # be efficient here, XX^T and X^TX have the same eigenvalues and thus the same entropy
                if N > D:
                    cov = hidden_state.T @ hidden_state
                else:
                    cov = hidden_state @ hidden_state.T
                cov /= torch.trace(cov)
                try:
                    entropy = itl.matrixAlphaEntropy(cov.float(), alpha=alpha)
                except Exception as e:
                    print(f"Error: {e}")
                    entropy = torch.tensor(math.nan)
                output_dict["unnormalized_entropy"][f"hidden_state_{i}"].append(entropy.item())
            output_dict["num_words"].append(N)
            output_dict["embedding_dim"].append(D)

    return output_dict
