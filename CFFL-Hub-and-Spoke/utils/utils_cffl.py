from utils.fl_types import ClientTrainResult
from torch.utils.data import DataLoader
from torch import nn
from time import perf_counter
import math
import torch
import numpy as np
import numpy.typing as npt
from typing import Callable


def flatten(gradient: list[npt.NDArray[np.generic]]) -> npt.NDArray[np.generic]:
    return np.concatenate([np.asarray(a, dtype=np.float64, order="C").ravel() for a in gradient])


def FedAvgGradients(client_results: list[ClientTrainResult],
                    partitioning: str,
                    max_class_count: int = 1,
                    ) -> list[npt.NDArray[np.generic]]:
    """Federated averaging of model updates weighted by dataset size."""
    device = "cpu"

    if (partitioning == "CLA"):
        weights = {r["client_id"]: float(r["class_j"]) for r in client_results}
        # weight_normalizing_value = max_class_count
        weight_normalizing_value = sum(weights.values()) or 1.0
    else:
        weights = {r["client_id"]: float(r["local_training_dataset_size"]) for r in client_results}
        weight_normalizing_value = sum(weights.values()) or 1.0

    weighted_updates = []
    for result in client_results:
        client_id = result["client_id"]
        w = weights[client_id] / weight_normalizing_value
        update = [torch.from_numpy(layer).to(device) * w for layer in result["model_update"]]
        weighted_updates.append(update)

    agg_layers = [
        torch.stack(layer_updates, dim=0).sum(dim=0)
        for layer_updates in zip(*weighted_updates)
    ]

    return [w.detach().cpu().numpy() for w in agg_layers]

# def FedAvgGradients(
#     client_results: list[ClientTrainResult],
#     partitioning: str,
#     max_class_count: int = 1,
#     cla_aggregation_mode: str = "paper",   # "paper" or "fedavg"
# ) -> list[npt.NDArray[np.generic]]:
#     """
#     Aggregation rule:
#     - Non-CLA: standard FedAvg by dataset size
#     - CLA + cla_aggregation_mode="paper": class_j / max_class_count
#     - CLA + cla_aggregation_mode="fedavg": standard FedAvg by dataset size
#     """
#     device = "cpu"

#     use_paper_cla = (partitioning == "CLA" and cla_aggregation_mode == "paper")

#     if use_paper_cla:
#         weights = {r["client_id"]: float(r["class_j"]) for r in client_results}
#         weight_normalizing_value = float(max_class_count) if max_class_count > 0 else 1.0
#     else:
#         weights = {
#             r["client_id"]: float(r["local_training_dataset_size"])
#             for r in client_results
#         }
#         weight_normalizing_value = sum(weights.values()) or 1.0

#     weighted_updates = []
#     for result in client_results:
#         client_id = result["client_id"]
#         w = weights[client_id] / weight_normalizing_value
#         update = [torch.from_numpy(layer).to(device) * w for layer in result["model_update"]]
#         weighted_updates.append(update)

#     agg_layers = [
#         torch.stack(layer_updates, dim=0).sum(dim=0)
#         for layer_updates in zip(*weighted_updates)
#     ]

#     return [w.detach().cpu().numpy() for w in agg_layers]


def reconstruct_client_models(
    client_results: list[ClientTrainResult],
    client_models: dict[int, list[npt.NDArray[np.generic]]],
    add_update_to_model: Callable[[list[npt.NDArray[np.generic]], list[npt.NDArray[np.generic]]], list[npt.NDArray[np.generic]]],
) -> dict[int, list[npt.NDArray[np.generic]]]:
    """
    Reconstruct the server-side copy of each client's local model after this round.
    """
    updated_client_models: dict[int, list[npt.NDArray[np.generic]]] = {}

    for result in client_results:
        client_id = result["client_id"]
        previous_model = client_models[client_id]
        delta_w_j = result["model_update"]
        updated_client_models[client_id] = add_update_to_model(previous_model, delta_w_j)

    return updated_client_models


def compute_CFFL_validation(
    client_results: list[ClientTrainResult],
    client_models: dict[int, list[npt.NDArray[np.generic]]],
    load_model: Callable[[nn.Module, list[npt.NDArray[np.generic]]], None],
    net: nn.Module,
    test: Callable[[nn.Module, DataLoader, torch.device], tuple[float, float]],
    validation_dataset: DataLoader,
    device: torch.device,
    punishment_factor: float,
    contribution_c_vals: dict[int, float],
    eps: float = 1e-12,
) -> tuple[
    dict[int, float],   # vloss_values
    dict[int, float],   # vacc_values
    float,              # sum_vacc_val
    dict[int, float],   # contribution_c_vals
    float,              # time_spent_testing_client_models
    float,              # time_spent_computing_contribution_values
    float,              # time_spent_normalizing_contribution_values
]:
    vloss_values: dict[int, float] = {}
    vacc_values: dict[int, float] = {}

    t0 = perf_counter()
    for result in client_results:
        client_id = result["client_id"]
        load_model(net=net, parameters=client_models[client_id])
        vloss_values[client_id], vacc_values[client_id] = test(net, validation_dataset, device)
        # print(f"[DEBUG] client={client_id} type(vloss)={type(vloss_values[client_id])} value={vloss_values[client_id]}")
        # print(f"[DEBUG] client={client_id} type(vacc)={type(vacc_values[client_id])} value={vacc_values[client_id]}")
    t1 = perf_counter()
    time_spent_testing_client_models = t1 - t0

    num_clients = max(len(client_results), 1)
    default_previous_c = 1.0 / num_clients

    c_j_vals: dict[int, float] = {}
    new_c_vals: dict[int, float] = {}

    t0 = perf_counter()
    sum_vacc_val = max(sum(vacc_values.values()), eps)

    for result in client_results:
        client_id = result["client_id"]
        normalized_vacc = vacc_values[client_id] / sum_vacc_val
        # print(f"[DEBUG] client={client_id} type(normalized vacc)={type(normalized_vacc)} value={normalized_vacc}")
        c_j_vals[client_id] = float(np.sinh(punishment_factor * normalized_vacc))

        previous_c = float(contribution_c_vals.get(client_id, default_previous_c))
        new_c_vals[client_id] = 0.5 * previous_c + 0.5 * c_j_vals[client_id]

    t1 = perf_counter()
    time_spent_computing_contribution_values = t1 - t0

    t0 = perf_counter()
    sum_c_vals = sum(new_c_vals.values())

    if sum_c_vals <= eps:
        uniform_c = 1.0 / num_clients
        for result in client_results:
            client_id = result["client_id"]
            contribution_c_vals[client_id] = uniform_c
    else:
        for result in client_results:
            client_id = result["client_id"]
            contribution_c_vals[client_id] = new_c_vals[client_id] / sum_c_vals

    t1 = perf_counter()
    time_spent_normalizing_contribution_values = t1 - t0

    return (
        vloss_values,
        vacc_values,
        sum_vacc_val,
        contribution_c_vals,
        time_spent_testing_client_models,
        time_spent_computing_contribution_values,
        time_spent_normalizing_contribution_values,
    )

def build_discounted_reward(
    allocated_update: list[npt.NDArray[np.generic]],
    local_update: list[npt.NDArray[np.generic]],
    self_ratio: float,
) -> list[npt.NDArray[np.generic]]:
    return [
        allocated_layer - (self_ratio * local_layer)
        for allocated_layer, local_layer in zip(allocated_update, local_update)
    ]

def build_effective_reward_for_reset_client(
    allocated_update: list[npt.NDArray[np.generic]],
    local_update: list[npt.NDArray[np.generic]],
    self_ratio: float,
    emulate_keep_local_gradient: bool,
) -> list[npt.NDArray[np.generic]]:
    """
    Client models are reset after local training in the uploaded model files.
    So the server must send the *actual* reward to be added to the reset model.

    If emulate_keep_local_gradient=True:
        reward = allocated_update + (1 - self_ratio) * local_update
               = local_update + [allocated_update - self_ratio * local_update]

    If emulate_keep_local_gradient=False:
        reward = allocated_update
    """
    if emulate_keep_local_gradient:
        return [
            allocated_layer + ((1.0 - self_ratio) * local_layer)
            for allocated_layer, local_layer in zip(allocated_update, local_update)
        ]
    return [np.array(layer, copy=True) for layer in allocated_update]

def compute_CFFL_reward(
    client_results: list[ClientTrainResult],
    aggregated_gradient: list[npt.NDArray[np.generic]],
    contribution_c_vals: dict[int, float],
    eps: float,
    partitioning: str,
    max_class_count: int = 1,
    mode: str = "norm",
    reward_allocation_mode: str = "paper_topk",
    round_idx: int = 0,
    reward_decay_gamma: float = 1.0,
) -> tuple[
    dict[int, list[npt.NDArray[np.generic]]],  # v_by_client
    dict[int, list[npt.NDArray[np.generic]]],  # discounted_layers_by_client
    dict[int, int],                            # n_j_vals
    float,                                     # max_n_j_val
    dict[int, float],                          # n_ratio_by_client
    dict[int, float],                          # q_by_client
    dict[int, int],                            # K_by_client
    dict[int, float],                          # reward_non_zero_frac_by_client
    dict[int, float],                          # reward_sparsity_by_client
    float,                                     # preparation_time
    float,                                     # compute_gradient_reward_time
]:
    """
    Computes CFFL rewards.

    For the norm-limited version, the reward is:

        reward_i = aggregated_gradient * q_i * reward_decay_gamma ** round_idx

    where q_i is still computed using the CFFL stratification rule:

        q_i = contribution_ratio_i * shard_ratio_i

    for reward_allocation_mode == "paper_topk".

    Since the clients reset their models after local training, v_by_client is the
    direct reward sent to each client. The local gradient is not added back here.
    """

    v_by_client: dict[int, list[npt.NDArray[np.generic]]] = {}
    discounted_layers_by_client: dict[int, list[npt.NDArray[np.generic]]] = {}
    n_ratio_by_client: dict[int, float] = {}
    q_by_client: dict[int, float] = {}
    K_by_client: dict[int, int] = {}
    v_non_zero_frac_by_client: dict[int, float] = {}
    v_sparsity_by_client: dict[int, float] = {}
    n_j_vals: dict[int, int] = {}

    t0 = perf_counter()

    max_c_val = max(max(contribution_c_vals.values(), default=0.0), eps)

    for result in client_results:
        client_id = result["client_id"]

        if partitioning == "CLA":
            n_j_vals[client_id] = int(result["class_j"])
        else:
            n_j_vals[client_id] = int(result["local_training_dataset_size"])

    if partitioning == "CLA":
        max_n_j_val = float(max_class_count)
    else:
        max_n_j_val = float(max(n_j_vals.values()))

    flattened_aggregated_gradient = flatten(aggregated_gradient)
    D = int(flattened_aggregated_gradient.size)

    round_decay_factor = float(reward_decay_gamma) ** int(round_idx)

    t1 = perf_counter()
    preparation_time = t1 - t0

    t0 = perf_counter()

    for result in client_results:
        client_id = result["client_id"]

        shard_ratio = float(n_j_vals[client_id]) / max(max_n_j_val, eps)
        c_ratio = float(contribution_c_vals[client_id]) / max_c_val

        if reward_allocation_mode == "paper_topk":
            q_i = c_ratio * shard_ratio
        elif reward_allocation_mode == "importance_only_topk":
            q_i = c_ratio
        else:
            raise ValueError(
                f"Unsupported reward_allocation_mode: {reward_allocation_mode}"
            )

        q_i = min(max(q_i, 0.0), 1.0)
        effective_q_i = q_i * round_decay_factor

        n_ratio_by_client[client_id] = shard_ratio

        if mode == "topk":
            # Keep original CFFL Top-K behavior.
            K = int(math.floor(D * q_i))
            K = max(0, min(K, D))

            q_by_client[client_id] = q_i
            K_by_client[client_id] = K

            delta_w_jg, nonzero_fraction, sparsity, kept_count, total_count = (
                mask_grad_update_by_order_global_magnitude_for_analysis(
                    aggregated_gradient=aggregated_gradient,
                    aggregated_gradient_size=D,
                    flattened_aggregated_gradient=flattened_aggregated_gradient,
                    mask_order=K,
                )
            )

        elif mode == "norm":
            # New norm-limited CFFL reward:
            # old: aggregated_gradient * q_i
            # new: aggregated_gradient * q_i * gamma^t
            q_by_client[client_id] = effective_q_i
            K_by_client[client_id] = D

            target_norm = effective_q_i * gradient_norm(aggregated_gradient)

            delta_w_jg = scale_gradient_to_norm(
                gradient=aggregated_gradient,
                target_norm=target_norm,
                eps=eps,
            )

            flat_reward = flatten(delta_w_jg)
            total_count = int(flat_reward.size)
            kept_count = int(np.count_nonzero(flat_reward))
            nonzero_fraction = (kept_count / total_count) if total_count > 0 else 0.0
            sparsity = 1.0 - nonzero_fraction

        else:
            raise ValueError(f"Unsupported mode: {mode}")

        # Kept only for logging / compatibility with your current tuple return.
        # Since clients reset after local training, the actual sent reward should be
        # v_by_client[client_id], not discounted_layers_by_client[client_id].
        own_uploaded_update = result["model_update"]
        discounted_reward = build_discounted_reward(
            allocated_update=delta_w_jg,
            local_update=own_uploaded_update,
            self_ratio=shard_ratio,
        )

        discounted_layers_by_client[client_id] = discounted_reward
        v_by_client[client_id] = delta_w_jg
        v_non_zero_frac_by_client[client_id] = nonzero_fraction
        v_sparsity_by_client[client_id] = sparsity

    t1 = perf_counter()
    compute_gradient_reward_time = t1 - t0

    return (
        v_by_client,
        discounted_layers_by_client,
        n_j_vals,
        max_n_j_val,
        n_ratio_by_client,
        q_by_client,
        K_by_client,
        v_non_zero_frac_by_client,
        v_sparsity_by_client,
        preparation_time,
        compute_gradient_reward_time,
    )

# def compute_CFFL_reward(
#     client_results: list[ClientTrainResult],
#     aggregated_gradient: list[npt.NDArray[np.generic]],
#     contribution_c_vals: dict[int, float],
#     eps: float,
#     partitioning: str,
#     max_class_count: int = 1,
#     mode: str = "topk",
#     reward_allocation_mode: str = "paper_topk",
#     cla_discount_mode: str = "paper",   # "paper" or "fedavg"
# ) -> tuple[
#     dict[int, list[npt.NDArray[np.generic]]],  # v_by_client
#     dict[int, list[npt.NDArray[np.generic]]],  # discounted_layers_by_client
#     dict[int, int],                            # n_j_vals (allocation basis for logging)
#     float,                                     # max_n_j_val (allocation basis for logging)
#     dict[int, float],                          # n_ratio_by_client (self/discount ratio)
#     dict[int, float],                          # q_by_client
#     dict[int, int],                            # K_by_client
#     dict[int, float],                          # reward_non_zero_frac_by_client
#     dict[int, float],                          # reward_sparsity_by_client
#     float,                                     # preparation_time
#     float,                                     # compute_gradient_reward_time
# ]:
#     v_by_client: dict[int, list[npt.NDArray[np.generic]]] = {}
#     discounted_layers_by_client: dict[int, list[npt.NDArray[np.generic]]] = {}
#     n_ratio_by_client: dict[int, float] = {}
#     q_by_client: dict[int, float] = {}
#     K_by_client: dict[int, int] = {}
#     v_non_zero_frac_by_client: dict[int, float] = {}
#     v_sparsity_by_client: dict[int, float] = {}
#     n_j_vals: dict[int, int] = {}

#     allocation_vals: dict[int, int] = {}
#     self_vals: dict[int, int] = {}

#     t0 = perf_counter()
#     max_c_val = max(max(contribution_c_vals.values(), default=0.0), eps)

#     for result in client_results:
#         client_id = result["client_id"]

#         # Allocation basis: keep CLA class-based boundaries
#         if partitioning == "CLA":
#             allocation_vals[client_id] = int(result["class_j"])
#         else:
#             allocation_vals[client_id] = int(result["local_training_dataset_size"])

#         # Self/discount basis: optionally align CLA with FedAvg aggregation
#         if partitioning == "CLA" and cla_discount_mode == "fedavg":
#             self_vals[client_id] = int(result["local_training_dataset_size"])
#         elif partitioning == "CLA":
#             self_vals[client_id] = int(result["class_j"])
#         else:
#             self_vals[client_id] = int(result["local_training_dataset_size"])

#     # Keep existing logging fields tied to the allocation boundary
#     n_j_vals = allocation_vals.copy()

#     if partitioning == "CLA":
#         max_n_j_val = float(max_class_count)
#     else:
#         max_n_j_val = float(max(allocation_vals.values()))

#     max_self_val = float(max(self_vals.values())) if self_vals else 1.0

#     flattened_aggregated_gradient = flatten(aggregated_gradient)
#     D = int(flattened_aggregated_gradient.size)
#     t1 = perf_counter()
#     preparation_time = t1 - t0

#     t0 = perf_counter()
#     for result in client_results:
#         client_id = result["client_id"]

#         allocation_ratio = float(allocation_vals[client_id]) / max(max_n_j_val, eps)
#         self_ratio = float(self_vals[client_id]) / max(max_self_val, eps)
#         c_ratio = float(contribution_c_vals[client_id]) / max_c_val

#         if reward_allocation_mode == "paper_topk":
#             q_i = c_ratio * allocation_ratio
#         elif reward_allocation_mode == "importance_only_topk":
#             q_i = c_ratio
#         else:
#             raise ValueError(
#                 f"Unsupported reward_allocation_mode: {reward_allocation_mode}"
#             )

#         q_i = min(max(q_i, 0.0), 1.0)
#         q_by_client[client_id] = q_i

#         if mode == "topk":
#             K = int(math.floor(D * q_i))
#             K = max(0, min(K, D))
#             K_by_client[client_id] = K

#             delta_w_jg, nonzero_fraction, sparsity, kept_count, total_count = (
#                 mask_grad_update_by_order_global_magnitude_for_analysis(
#                     aggregated_gradient=aggregated_gradient,
#                     aggregated_gradient_size=D,
#                     flattened_aggregated_gradient=flattened_aggregated_gradient,
#                     mask_order=K,
#                 )
#             )
#         else:
#             raise ValueError(f"Unsupported mode: {mode}")

#         own_uploaded_update = result["model_update"]
#         discounted_reward = build_discounted_reward(
#             allocated_update=delta_w_jg,
#             local_update=own_uploaded_update,
#             self_ratio=self_ratio,
#         )

#         discounted_layers_by_client[client_id] = discounted_reward
#         n_ratio_by_client[client_id] = self_ratio
#         v_by_client[client_id] = delta_w_jg
#         v_non_zero_frac_by_client[client_id] = nonzero_fraction
#         v_sparsity_by_client[client_id] = sparsity

#     t1 = perf_counter()
#     compute_gradient_reward_time = t1 - t0

#     return (
#         v_by_client,
#         discounted_layers_by_client,
#         n_j_vals,
#         max_n_j_val,
#         n_ratio_by_client,
#         q_by_client,
#         K_by_client,
#         v_non_zero_frac_by_client,
#         v_sparsity_by_client,
#         preparation_time,
#         compute_gradient_reward_time,
#     )

def compute_update_stats(
    update: list[npt.NDArray[np.generic]],
) -> tuple[list[npt.NDArray[np.generic]], float, float, int, int]:
    copied_layers = [np.array(g, copy=True) for g in update]

    total_count = 0
    kept_count = 0
    for arr in copied_layers:
        total_count += arr.size
        kept_count += int(np.count_nonzero(arr))

    nonzero_fraction = (kept_count / total_count) if total_count > 0 else 0.0
    sparsity = 1.0 - nonzero_fraction
    return copied_layers, nonzero_fraction, sparsity, kept_count, total_count


def mask_grad_update_by_order_global_magnitude_for_analysis(
    aggregated_gradient: list[npt.NDArray[np.generic]],
    aggregated_gradient_size: int,
    flattened_aggregated_gradient: npt.NDArray[np.generic],
    mask_order: int,
) -> tuple[list[npt.NDArray[np.generic]], float, float, int, int]:
    if mask_order < 0:
        mask_order = 0

    D = aggregated_gradient_size

    if mask_order == 0:
        return mask_grad_update_by_magnitude_for_analysis(aggregated_gradient, float("inf"))

    if D == 0:
        return [np.array(g, copy=True) for g in aggregated_gradient], 0.0, 1.0, 0, 0

    if mask_order > D:
        mask_order = D

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


def mask_grad_update_per_layer_by_percentile_for_analysis(
    gradient: list[npt.NDArray[np.generic]],
    mask_percentile: float,
) -> tuple[list[npt.NDArray[np.generic]], float, float, int, int]:
    mask_percentile = max(0.0, float(mask_percentile))
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
            masked_layers.append(arr.copy())
            kept_count += L
            continue

        abs_flat = np.abs(flat)
        idx_topk = np.argpartition(abs_flat, L - k)[L - k:]

        masked_flat = np.zeros_like(flat)
        masked_flat[idx_topk] = flat[idx_topk]

        masked_layers.append(masked_flat.reshape(arr.shape))
        kept_count += k

    nonzero_fraction = (kept_count / total_count) if total_count > 0 else 0.0
    sparsity = 1.0 - nonzero_fraction

    return masked_layers, nonzero_fraction, sparsity, kept_count, total_count


def gradient_norm(
    gradient: list[npt.NDArray[np.generic]],
) -> float:
    return float(np.linalg.norm(flatten(gradient)))


def zero_like_gradient(
    gradient: list[npt.NDArray[np.generic]],
) -> list[npt.NDArray[np.generic]]:
    return [np.zeros_like(layer) for layer in gradient]


def scale_gradient_to_norm(
    gradient: list[npt.NDArray[np.generic]],
    target_norm: float,
    eps: float,
) -> list[npt.NDArray[np.generic]]:
    current_norm = gradient_norm(gradient)

    if current_norm <= eps or target_norm <= 0.0:
        return zero_like_gradient(gradient)

    scale = target_norm / max(current_norm, eps)

    scaled_gradient: list[npt.NDArray[np.generic]] = []
    for layer in gradient:
        arr64 = np.asarray(layer, dtype=np.float64, order="C") * scale
        scaled_gradient.append(arr64.astype(np.asarray(layer).dtype, copy=False))

    return scaled_gradient