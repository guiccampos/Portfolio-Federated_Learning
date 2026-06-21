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

def compute_gradient_reward_max_norm(client_results: list[ClientTrainResult],
                            aggregated_gradient: list[npt.NDArray[np.generic]],
                            # flattened_aggregated_normalized_gradient: npt.NDArray[np.generic],
                            importance_coefficients: dict[int, float],
                            beta: float,
                            eps: float, 
                            mode: str = "topk"
                            ) -> tuple[
                                dict[int, list[npt.NDArray[np.generic]]],   # v_by_client
                                dict[int, float],                           # tanh_vals
                                float,                                      # tanh_normalization_value
                                dict[int, float],                           # q_by_client
                                dict[int, int],                             # mask_order_by_client
                                dict[int, float],                           # v_non_zero_frac_by_client
                                dict[int, float],                           # v_sparsity_by_client
                                float,                                      # tahn_vals_computation_time
                                float                                       # compute_gradient_reward_time
                            ]:

    v_by_client: dict[int, list[npt.NDArray[np.generic]]] = {}                                    # Reward Gradient
    q_by_client: dict[int, float] = {}                  # Reward Gradient Quality
    mask_order_by_client: dict[int, float] = {} 
    v_non_zero_frac_by_client: dict[int, float] = {}    # Reward Gradient Non-Zero Fraction
    v_sparsity_by_client: dict[int, float] = {}         # Reward Gradient Sparsity

    tanh_vals_computation_start_time = perf_counter()
    tanh_vals: dict[int, float] = {r["client_id"]: np.tanh(beta * importance_coefficients[r["client_id"]]) for r in client_results}
    tanh_vals_computation_end_time = perf_counter()
    tahn_vals_computation_time = tanh_vals_computation_end_time - tanh_vals_computation_start_time

    max_tanh = max(tanh_vals.values()) if tanh_vals else 1.0
    tanh_normalization_value = max(max_tanh, eps)

    # Total number of scalar parameters across all layers
    flattened_aggregated_gradient = flatten(aggregated_gradient)
    D = int(flattened_aggregated_gradient.size)

    compute_gradient_reward_start_time = perf_counter()
    for result in client_results:
        client_id = result["client_id"]

        ratio: float = tanh_vals[client_id] / (tanh_normalization_value + eps)

        if mode == "topk":
            mask_order = int(math.floor(D * ratio))
            q_by_client[client_id] = ratio
            mask_order_by_client[client_id] = mask_order

            masked_layers, nonzero_fraction, sparsity, kept_count, total_count = (
                mask_grad_update_by_order_global_magnitude_for_analysis(
                    aggregated_gradient=aggregated_gradient,
                    aggregated_gradient_size=D,
                    flattened_aggregated_gradient=flattened_aggregated_gradient,
                    mask_order=mask_order,
                )
            )

        elif mode == "layer-topk":
            q_by_client[client_id] = float(ratio)

            masked_layers, nonzero_fraction, sparsity, kept_count, total_count = (
                mask_grad_update_per_layer_by_percentile_for_analysis(
                    gradient=aggregated_gradient,
                    mask_percentile=ratio,
                )
            )

        v_by_client[client_id] = masked_layers
        v_non_zero_frac_by_client[client_id] = nonzero_fraction
        v_sparsity_by_client[client_id] = sparsity     
    compute_gradient_reward_end_time = perf_counter()
    compute_gradient_reward_time = compute_gradient_reward_end_time - compute_gradient_reward_start_time
    return v_by_client, tanh_vals, tanh_normalization_value, q_by_client, mask_order_by_client, v_non_zero_frac_by_client, v_sparsity_by_client, tahn_vals_computation_time, compute_gradient_reward_time

def mask_grad_update_by_order_global_magnitude_for_analysis(
    aggregated_gradient: list[npt.NDArray[np.generic]],
    aggregated_gradient_size: int,
    flattened_aggregated_gradient: npt.NDArray[np.generic],
    mask_order: int,
) -> tuple[list[npt.NDArray[np.generic]],       # v_by_layer
           float,                               # nonzero_fraction
           float,                               # sparsity
           int,                                 # kept_count
           int]:                                # total_count
    # Defensive clamp
    if mask_order < 0:
        mask_order = 0

    D = aggregated_gradient_size

    if mask_order == 0:
        return mask_grad_update_by_magnitude_for_analysis(aggregated_gradient, float("inf"))

    if D == 0:
        # Empty gradient list / all empty arrays
        return [np.array(g, copy=True) for g in aggregated_gradient], 0.0, 1.0, 0, 0

    if mask_order > D:
        mask_order = D  # torch.topk would error if k > D

    # threshold = k-th largest magnitude (same as torch.topk(...).values[-1])
    abs_flat = np.abs(flattened_aggregated_gradient)
    threshold = float(np.partition(abs_flat, -mask_order)[-mask_order])

    return mask_grad_update_by_magnitude_for_analysis(aggregated_gradient, threshold)

def mask_grad_update_by_magnitude_for_analysis(
    grad_update: list[npt.NDArray[np.generic]],
    mask_constant: float,
) -> tuple[list[npt.NDArray[np.generic]], float, float, int, int]:
    
    masked_layers: list[npt.NDArray[np.generic]] = [np.array(g, copy=True) for g in grad_update]

    for i in range(len(masked_layers)):
        arr = masked_layers[i]
        arr[np.abs(arr) < mask_constant] = 0
        masked_layers[i] = arr

    total_count = 0
    kept_count = 0
    for arr in masked_layers:
        total_count += arr.size
        kept_count += int(np.count_nonzero(arr))

    nonzero_fraction = (kept_count / total_count) if total_count > 0 else 0.0
    sparsity = 1.0 - nonzero_fraction
    return masked_layers, nonzero_fraction, sparsity, kept_count, total_count

def mask_grad_update_per_layer_by_percentile_for_analysis(gradient: list[npt.NDArray[np.generic]], 
                                 mask_percentile: float
                                 ) -> tuple[list[npt.NDArray[np.generic]], float, float, int, int]:
    
    mask_percentile: float = max(0.0, float(mask_percentile))  # Clamps to 0 if the percentile is negative
    masked_layers: list[npt.NDArray[np.generic]] = []

    kept_count = 0
    total_count = 0

    for layer in gradient:
        arr = np.asarray(layer)
        flat = arr.ravel()
        L = flat.size
        total_count += L

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
            kept_count += L
            continue

        abs_flat = np.abs(flat)

        # exact top-k indices (unordered)
        idx_topk = np.argpartition(abs_flat, L - k)[L - k:]

        # build masked output
        masked_flat = np.zeros_like(flat)
        masked_flat[idx_topk] = flat[idx_topk]

        masked_layers.append(masked_flat.reshape(arr.shape))
        kept_count += k

    nonzero_fraction = (kept_count / total_count) if total_count > 0 else 0.0
    sparsity = 1.0 - nonzero_fraction

    return masked_layers, nonzero_fraction, sparsity, kept_count, total_count