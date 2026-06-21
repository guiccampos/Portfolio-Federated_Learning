import os
import csv
import numpy as np
import numpy.typing as npt
from utils.fl_types import ClientTestResult
from typing import Any

SERVER_HEADERS = [
    "round",
    "time_spent_normalizing_gradients",
    "time_spent_aggregating_gradients",
    "time_spent_updating_server_model",
    "time_spent_computing_cosine_similarities",
    "time_spent_normalizing_importance_coefficient",
    "time_spent_computing_tanh_vals",
    "time_spent_computing_sparsified_gradients",
    "time_spent_constructing_rewards_payload",
    "time_spent_testing_global_model",
    "time_spent_in_train_instructions_broadcast",
    "time_spent_in_collecting_training_results",
    "time_spent_in_eval_instructions_broadcast",
    "time_spent_in_scattering_rewards",
    "time_spent_in_collecting_testing_results",
    "round_time_spent_in_server_computation",
    "round_time_spent_in_server_blocking_communication",
    "round_duration",
    "total_time_spent_in_server_computation",
    "total_time_spent_in_server_blocking_communication",
    "total_time",
    "round_clients_local_training_dataset_average_accuracy",
    "round_clients_local_testing_dataset_average_accuracy",
    "round_clients_central_testing_dataset_average_accuracy",
    "round_server_central_testing_dataset_loss",
    "round_server_central_testing_dataset_accuracy",
    "flattened_aggregated_gradient_norm"
]

CLIENT_HEADERS = [
    "client_i_training_loss",
    "client_i_training_accuracy",
    "client_i_training_dataset_size",
    "client_i_evaluation_loss",
    "client_i_evaluation_accuracy",
    "client_i_evaluation_dataset_size",
    "client_i_central_evaluation_loss",
    "client_i_central_evaluation_accuracy",
    "client_i_cosine_similarity",
    "client_i_importance_coefficient",
    "client_i_tanh_val",
    "tanh_normalization_value",
    "client_i_q",
    "client_i_mask_order",
    "client_i_v_nonzero_fraction",
    "client_i_v_sparsity",
    "client_i_I_t",
    "client_i_B",
    "local_epochs",
    "learning_rate",
    "gamma",
    "client_i_corrupted_labels",
    "client_i_corruption_fraction"
]

def _ensure_parent_dir(path: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

def _ensure_csv(path: str, headers: list[str]) -> None:
    _ensure_parent_dir(path)
    if not os.path.exists(path):
        with open(path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)

def write_server_metrics(
    path: str,
    round_idx: int,
    time_spent_normalizing_gradients: float,
    time_spent_aggregating_gradients: float,
    time_spent_updating_server_model: float,
    time_spent_computing_cosine_similarities: float,
    time_spent_normalizing_importance_coefficient: float,
    time_spent_computing_tanh_vals: float,
    time_spent_computing_sparsified_gradients: float,
    time_spent_constructing_rewards_payload: float,
    time_spent_testing_global_model: float,
    time_spent_in_train_instructions_broadcast: float,
    time_spent_in_collecting_training_results: float,
    time_spent_in_eval_instructions_broadcast: float,
    time_spent_in_scattering_rewards: float,
    time_spent_in_collecting_testing_results: float,
    round_time_spent_in_server_computation: float,
    round_time_spent_in_server_blocking_communication: float,
    round_duration: float,
    total_time_spent_in_server_computation: float,
    total_time_spent_in_server_blocking_communication: float,
    total_time: float,
    round_clients_local_training_dataset_average_accuracy: float,
    round_clients_local_testing_dataset_average_accuracy: float,
    round_clients_central_testing_dataset_average_accuracy: float,
    round_server_central_testing_dataset_loss: float,
    round_server_central_testing_dataset_accuracy: float,
    flattened_aggregated_gradient_norm: float,
) -> None:
    _ensure_csv(path, SERVER_HEADERS)
    row = [
        round_idx,
        time_spent_normalizing_gradients,
        time_spent_aggregating_gradients,
        time_spent_updating_server_model,
        time_spent_computing_cosine_similarities,
        time_spent_normalizing_importance_coefficient,
        time_spent_computing_tanh_vals,
        time_spent_computing_sparsified_gradients,
        time_spent_constructing_rewards_payload,
        time_spent_testing_global_model,
        time_spent_in_train_instructions_broadcast,
        time_spent_in_collecting_training_results,
        time_spent_in_eval_instructions_broadcast,
        time_spent_in_scattering_rewards,
        time_spent_in_collecting_testing_results,
        round_time_spent_in_server_computation,
        round_time_spent_in_server_blocking_communication,
        round_duration,
        total_time_spent_in_server_computation,
        total_time_spent_in_server_blocking_communication,
        total_time,
        round_clients_local_training_dataset_average_accuracy,
        round_clients_local_testing_dataset_average_accuracy,
        round_clients_central_testing_dataset_average_accuracy,
        round_server_central_testing_dataset_loss,
        round_server_central_testing_dataset_accuracy,
        flattened_aggregated_gradient_norm
    ]
    with open(path, mode="a", newline="") as f:
        csv.writer(f).writerow(row)

def write_client_metrics(
    path: str,
    round_idx: int,
    client_results_test: list[ClientTestResult],
    client_contribution: dict[int, dict[str, Any]],
    local_epochs: int,
    learning_rate: float,
    gamma: float,
    tanh_vals: dict[int, float],
    tanh_normalization_value: float,
    q_by_client: dict[int, float],
    mask_order_by_client: dict[int, int],
    v_non_zero_frac_by_client: dict[int, float],
    v_sparsity_by_client: dict[int, float],
    I_t_by_client: dict[int,float],
    B_t_by_client:  dict[int,float],
) -> None:
    # Build dynamic headers (wide pairwise columns)
    headers = ["round", "client_id"] + CLIENT_HEADERS

    _ensure_csv(path, headers)

    # Fast lookups by client_id
    test_by_id  = {r["client_id"]: r for r in client_results_test if isinstance(r, dict) and "client_id" in r}

    # Union of all client_ids seen in any of the three sources
    all_ids = sorted(set(test_by_id) | set(client_contribution))

    with open(path, mode="a", newline="") as f:
        w = csv.writer(f)
        for cid in all_ids:
            te = test_by_id.get(cid, {})
            cc = client_contribution.get(cid, {})

            tanh_val = tanh_vals.get(cid, 0)
            q_val = q_by_client.get(cid, 0)
            mask_val = mask_order_by_client.get(cid, 0)
            nzfrac_val = v_non_zero_frac_by_client.get(cid, 0)
            sparsity_val = v_sparsity_by_client.get(cid, 0)
            client_i_I_t_val = I_t_by_client.get(cid, 0)
            client_i_B = B_t_by_client.get(cid, 0)
            row = [
                round_idx,
                cid,
                te.get("local_dataset_train_loss"),
                te.get("local_dataset_train_accuracy"),
                te.get("local_training_dataset_size"),
                te.get("local_dataset_test_loss"),
                te.get("local_dataset_test_accuracy"),
                te.get("local_testing_dataset_size"),
                te.get("central_dataset_test_loss"),
                te.get("central_dataset_test_accuracy"),
                cc.get("cosine_similarity"),
                cc.get("importance_coefficient"),
                tanh_val,
                tanh_normalization_value,
                q_val,
                mask_val,
                nzfrac_val,
                sparsity_val,
                client_i_I_t_val,
                client_i_B,
                local_epochs,
                learning_rate,
                gamma,
                te.get("corrupted_labels", None),
                te.get("corruption_fraction", None),
            ]

            w.writerow(row)