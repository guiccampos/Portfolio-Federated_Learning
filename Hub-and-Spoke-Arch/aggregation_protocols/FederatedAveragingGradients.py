from utils.fl_types import ClientTrainResult
import torch
import numpy as np
import numpy.typing as npt

def FedAvgGradients(client_results: list[ClientTrainResult]) -> list[npt.NDArray[np.generic]]:
    """Federated Averaging of model updates (gradients) weighted by dataset sizes."""
    device = "cpu"

    weights = {r["client_id"]: float(r["training_dataset_size"]) for r in client_results}

    weight_normalizing_value = sum(weights.values()) or 1.0

    weighted_updates = []
    for result in client_results:
        client_id = result["client_id"]
        w = weights[client_id] / weight_normalizing_value
        update = [torch.from_numpy(layer).to(device) * w for layer in result["model_update"]]
        weighted_updates.append(update)

    # Compute average weights of each layer
    agg_layers = [
        torch.stack(layer_updates,dim=0).sum(dim=0)
        for layer_updates in zip(*weighted_updates) # Tranposes the list-of-lists to allow by layer grouping
    ]

    # back to numpy NDArrays
    return [w.detach().cpu().numpy() for w in agg_layers]