from utils.fl_types import ClientTrainResult, ClientTestResult
from torch import Tensor, nn
from typing import List, Dict, Tuple, Any
from collections import defaultdict
from time import perf_counter
import math
import torch
import numpy as np
import numpy.typing as npt

def flatten(gradient: list[npt.NDArray[np.generic]]) -> npt.NDArray[np.generic]:
    return np.concatenate([np.asarray(a, dtype=np.float64, order="C").ravel() for a in gradient])

def FedAvgGradients(client_results: list[ClientTrainResult],
                    importance_coefficients: dict[int, float],
                    round: int,
                    device: torch.device | str
                    ) -> list[npt.NDArray[np.generic]]:

    """Federated Averaging of model updates (gradients) weighted by dataset sizes."""
    if(round == 0):
        weights = {result["client_id"]: float(result["local_training_dataset_size"]) for result in client_results}
    else:
        weights = {client_id: float(w) for client_id, w in importance_coefficients.items()}

    weight_normalizing_value = sum(weights.values()) or 1.0

    weighted_updates = []
    for result in client_results:
        client_id: int = result["client_id"]
        w: float = weights[client_id] / weight_normalizing_value
        update = []
        for layer in result["model_update"]:
            arr = np.asarray(layer)          # handles np.int64, lists, etc.
            update.append(torch.from_numpy(arr).to(device) * w)
        weighted_updates.append(update)

    # Compute average weights of each layer
    agg_layers = [
        torch.stack(layer_updates,dim=0).sum(dim=0)
        for layer_updates in zip(*weighted_updates) # Tranposes the list-of-lists to allow by layer grouping
    ]

    # back to numpy NDArrays
    return [w.detach().cpu().numpy() for w in agg_layers]

def normalize_gradients(client_results: list[ClientTrainResult],
                        gamma: float,
                        eps: float,
                        ) -> None:
    
    for result in client_results:
        gradient = result["model_update"]

        sum_of_squares = 0.0
        grad64: list[np.ndarray] = []
        for layer in gradient:
            layer_array = np.asarray(layer, dtype=np.float64, order="C")
            v = layer_array.ravel()
            sum_of_squares += np.dot(v,v)
            grad64.append(layer_array)
        l2_norm = float(np.sqrt(sum_of_squares))

        if l2_norm > 0.0:
            scale = gamma / max(l2_norm, eps)
        else:
            scale = 0.0

        normalized_gradient: list[npt.NDArray[np.generic]] = []
        for orig_layer, a64 in zip(gradient, grad64):
            orig_dtype = np.asarray(orig_layer).dtype
            normalized_gradient.append((a64 * scale).astype(orig_dtype, copy=False))

        result["model_update"] = normalized_gradient
        result["flattened_normalized_gradient"] = flatten(normalized_gradient)

def compute_cosine_similarity(client_results: list[ClientTrainResult],
                              aggregated_normalized_gradients: list[npt.NDArray[np.generic]],
                              old_importance_coefficients: dict[int, float],
                              alpha: float,
                              eps: float,
                              gamma: float
                              ) -> tuple[
                                dict[int, dict[str, float]],     # client_contribution
                                dict[int, float],                # importance_coefficients
                                npt.NDArray[np.float64],         # flattened_aggregated_gradient
                                float,                           # flattened_aggregated_gradient_norm
                                float,                           # cosine_similarity_time
                                float,                           # ic_normalization_time
                                dict[int,float]                  # u_i,t norm
                            ]:

    importance_coefficients: dict[int, float] = {}
    client_contribution = defaultdict(dict)

    client_gradient_norm: dict[int, float] = {}

    I_t: dict[int, float] = {}
    B_t: dict[int, float] = {}
    # Flatten u_(N,t)
    flattened_aggregated_gradient = flatten(aggregated_normalized_gradients)
    flattened_aggregated_gradient_norm = max(float(np.linalg.norm(flattened_aggregated_gradient)), 1e-7)

    cosine_similarity_start_time = perf_counter()
    for result in client_results:
        client_id = result["client_id"]
        client_num_samples = result["local_training_dataset_size"]

        u_it_flat = np.asarray(result["flattened_normalized_gradient"], dtype=np.float64, order="C")
        u_it_norm = max(np.linalg.norm(u_it_flat), eps)

        client_gradient_norm[client_id]=u_it_norm

        # Compute Cosine Similarity
        dot_product = np.dot(u_it_flat,flattened_aggregated_gradient)
        cosine_similarity = dot_product / (u_it_norm * flattened_aggregated_gradient_norm)

        if cosine_similarity <= 0.0:
            cosine_similarity = 1e-10

        I_t[client_id] = 1/(dot_product + eps)
        B_t[client_id] = I_t[client_id] * gamma * gamma
        # Compute Importance Coefficient
        importance_coefficient = alpha * old_importance_coefficients.get(client_id, 0) + (1.0 - alpha) * cosine_similarity
    
        if importance_coefficient <= 0.0:
            importance_coefficient = 1e-3

        client_contribution[client_id]["cosine_similarity"] = cosine_similarity
        client_contribution[client_id]["importance_coefficient"] = importance_coefficient
        client_contribution[client_id]["local_training_dataset_size"] = client_num_samples
        importance_coefficients[client_id] = importance_coefficient
    cosine_similarity_end_time = perf_counter()
    cosine_similarity_computation_time = cosine_similarity_end_time - cosine_similarity_start_time

    # Normalizing Importance Coefficient
    importance_coefficient_normalization_start_time = perf_counter()
    normalizing_value = sum(importance_coefficients.values())
    for client_id in importance_coefficients:
        client_contribution[client_id]["importance_coefficient"] = (client_contribution[client_id]["importance_coefficient"]/normalizing_value)
        importance_coefficients[client_id] = importance_coefficients[client_id]/normalizing_value
    importance_coefficient_normalization_end_time = perf_counter()
    importance_coefficient_normalization_time = importance_coefficient_normalization_end_time - importance_coefficient_normalization_start_time
    return client_contribution, importance_coefficients, flattened_aggregated_gradient, flattened_aggregated_gradient_norm, cosine_similarity_computation_time, importance_coefficient_normalization_time, I_t, B_t

def compute_gradient_reward(normalized_client_results: list[ClientTrainResult],
                            aggregated_normalized_gradients: list[npt.NDArray[np.generic]],
                            flattened_aggregated_normalized_gradient: npt.NDArray[np.generic],
                            importance_coefficients: dict[int, float],
                            beta: float,
                            eps: float, 
                            mode: str = "magnitude"
                            ) -> tuple[
                                dict[int, list[npt.NDArray[np.generic]]],   # v_by_client
                                dict[int, float],                           # tanh_vals
                                float,                                      # tanh_normalization_value
                                dict[int, float],                           # q_by_client
                                dict[int, int],                             # mask_order_by_client
                                float,                                      # tahn_vals_computation_time
                                float                                       # compute_gradient_reward_time
                            ]:

    v_by_client: dict[int, list[npt.NDArray[np.generic]]] = {}                                    # Reward Gradient
    q_by_client: dict[int, float] = {}                  # Reward Gradient Quality
    mask_order_by_client: dict[int, float] = {} 

    tanh_vals_computation_start_time = perf_counter()
    tanh_vals: dict[int, float] = {r["client_id"]: np.tanh(beta * importance_coefficients[r["client_id"]]) for r in normalized_client_results}
    max_tanh = max(tanh_vals.values()) if tanh_vals else 1.0
    tanh_normalization_value = max(max_tanh, eps)
    tanh_vals_computation_end_time = perf_counter()
    tahn_vals_computation_time = tanh_vals_computation_end_time - tanh_vals_computation_start_time

    compute_gradient_reward_start_time = perf_counter()
    # Total number of scalar parameters across all layers
    D = int(flattened_aggregated_normalized_gradient.size)
    for result in normalized_client_results:
        client_id = result["client_id"]

        ratio: float = tanh_vals[client_id] / (tanh_normalization_value + eps)

        if mode == "magnitude":
            mask_order = int(math.floor(D * ratio))
            q_by_client[client_id] = ratio
            mask_order_by_client[client_id] = mask_order
            masked_layers = (
                mask_grad_update_by_order_global_magnitude(
                    aggregated_gradient=aggregated_normalized_gradients,
                    aggregated_gradient_size=D,
                    flattened_aggregated_normalized_gradient=flattened_aggregated_normalized_gradient,
                    mask_order=mask_order,
                )
            )

        elif mode == "layer":
            q_by_client[client_id] = float(ratio)

            masked_layers,_, _, _, _ = (
                mask_grad_update_per_layer_by_percentile(
                    gradient=aggregated_normalized_gradients,
                    mask_percentile=ratio,
                )
            )

        v_by_client[client_id] = masked_layers
    compute_gradient_reward_end_time = perf_counter()
    compute_gradient_reward_time = compute_gradient_reward_end_time - compute_gradient_reward_start_time
    return v_by_client, tanh_vals, tanh_normalization_value, q_by_client, mask_order_by_client, tahn_vals_computation_time, compute_gradient_reward_time

def mask_grad_update_by_order_global_magnitude(
    aggregated_gradient: list[npt.NDArray[np.generic]],
    aggregated_gradient_size: int,
    flattened_aggregated_normalized_gradient: npt.NDArray[np.generic],
    mask_order: int,
) -> list[npt.NDArray[np.generic]]:
    # Defensive clamp
    if mask_order < 0:
        mask_order = 0

    D = aggregated_gradient_size

    if mask_order == 0:
        return mask_grad_update_by_magnitude(aggregated_gradient, float("inf"))

    if D == 0:
        # Empty gradient list / all empty arrays
        return [np.array(g, copy=True) for g in aggregated_gradient]

    if mask_order > D:
        mask_order = D  # torch.topk would error if k > D

    # threshold = k-th largest magnitude (same as torch.topk(...).values[-1])
    abs_flat = np.abs(flattened_aggregated_normalized_gradient)
    threshold = float(np.partition(abs_flat, -mask_order)[-mask_order])

    return mask_grad_update_by_magnitude(aggregated_gradient, threshold)

def mask_grad_update_by_magnitude(
    grad_update: list[npt.NDArray[np.generic]],
    mask_constant: float,
) -> list[npt.NDArray[np.generic]]:

    masked_layers: list[npt.NDArray[np.generic]] = [np.array(g, copy=True) for g in grad_update]

    for i in range(len(masked_layers)):
        arr = masked_layers[i]
        arr[np.abs(arr) < mask_constant] = 0
        masked_layers[i] = arr

    return masked_layers

def mask_grad_update_per_layer_by_percentile(gradient: list[npt.NDArray[np.generic]], 
                                 mask_percentile: float
                                 ) -> list[npt.NDArray[np.generic]]:
    
    mask_percentile: float = max(0.0, float(mask_percentile))  # Clamps to 0 if the percentile is negative
    masked_layers: list[npt.NDArray[np.generic]] = []

    for layer in gradient:
        arr = np.asarray(layer)
        flat = arr.ravel()
        L = flat.size

        if L == 0:
            masked_layers.append(arr.copy())
            continue

        k = int(math.ceil(L * mask_percentile))

        if k <= 0:
            masked_layers.append(np.zeros_like(arr))
            continue

        if k >= L:
            # keep everything
            masked_layers.append(arr.copy())
            continue

        abs_flat = np.abs(flat)

        # exact top-k indices (unordered)
        idx_topk = np.argpartition(abs_flat, L - k)[L - k:]

        # build masked output
        masked_flat = np.zeros_like(flat)
        masked_flat[idx_topk] = flat[idx_topk]

        masked_layers.append(masked_flat.reshape(arr.shape))

    return masked_layers
