import os
import csv
from utils.fl_types import ClientTestResult


SERVER_HEADERS = [
    "round",
    "time_spent_aggregating_gradients",
    "time_spent_updating_server_model",
    "time_spent_reconstructing_client_models",
    "time_spent_testing_client_models",
    "time_spent_computing_contribution_values",
    "time_spent_normalizing_contribution_values",
    "time_spent_in_reward_preparation",
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
]

CLIENT_HEADERS = [
    "client_i_training_dataset_size",
    "client_i_training_loss",
    "client_i_training_accuracy",
    "client_i_evaluation_dataset_size",
    "client_i_evaluation_loss",
    "client_i_evaluation_accuracy",
    "client_i_central_evaluation_loss",
    "client_i_central_evaluation_accuracy",
    "validation_accuracy",
    "client_i_importance_coefficient",
    "n_j",
    "max_n_j",
    "n_j_over_max_n_j",
    "client_i_q",
    "K_i",
    "client_i_v_nonzero_fraction",
    "client_i_v_sparsity",
    "local_epochs",
    "learning_rate",
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
    time_spent_aggregating_gradients: float,
    time_spent_updating_server_model: float,
    time_spent_reconstructing_client_models: float,
    time_spent_testing_client_models: float,
    time_spent_computing_contribution_values: float,
    time_spent_normalizing_contribution_values: float,
    time_spent_in_reward_preparation: float,
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
) -> None:
    _ensure_csv(path, SERVER_HEADERS)

    row = [
        round_idx,
        time_spent_aggregating_gradients,
        time_spent_updating_server_model,
        time_spent_reconstructing_client_models,
        time_spent_testing_client_models,
        time_spent_computing_contribution_values,
        time_spent_normalizing_contribution_values,
        time_spent_in_reward_preparation,
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
    ]

    with open(path, mode="a", newline="") as f:
        csv.writer(f).writerow(row)


def write_client_metrics(
    path: str,
    round_idx: int,
    client_results_test: list[ClientTestResult],
    contribution_c_vals: dict[int, float],
    local_epochs: int,
    learning_rate: float,
    vacc_values: dict[int, float],
    n_j_vals: dict[int, int],
    max_n_j_val: float,
    q_by_client: dict[int, float],
    K_by_client: dict[int, int],
    reward_non_zero_frac_by_client: dict[int, float],
    reward_sparsity_by_client: dict[int, float],
) -> None:
    headers = ["round", "client_id"] + CLIENT_HEADERS
    _ensure_csv(path, headers)

    test_by_id = {
        r["client_id"]: r for r in client_results_test
        if isinstance(r, dict) and "client_id" in r
    }

    all_ids = sorted(
        set(test_by_id)
        | set(contribution_c_vals)
        | set(vacc_values)
        | set(n_j_vals)
        | set(q_by_client)
        | set(K_by_client)
    )

    with open(path, mode="a", newline="") as f:
        writer = csv.writer(f)

        for cid in all_ids:
            te = test_by_id.get(cid, {})
            n_j = n_j_vals.get(cid, 0)
            n_ratio = (float(n_j) / float(max_n_j_val)) if max_n_j_val > 0 else 0.0

            row = [
                round_idx,
                cid,
                te.get("local_training_dataset_size"),
                te.get("local_dataset_train_loss"),
                te.get("local_dataset_train_accuracy"),
                te.get("local_testing_dataset_size"),
                te.get("local_dataset_test_loss"),
                te.get("local_dataset_test_accuracy"),
                te.get("central_dataset_test_loss"),
                te.get("central_dataset_test_accuracy"),
                vacc_values.get(cid, 0.0),
                contribution_c_vals.get(cid, 0.0),
                n_j,
                max_n_j_val,
                n_ratio,
                q_by_client.get(cid, 0.0),
                K_by_client.get(cid, 0),
                reward_non_zero_frac_by_client.get(cid, 0.0),
                reward_sparsity_by_client.get(cid, 0.0),
                local_epochs,
                learning_rate,
                te.get("corrupted_labels", False),
                te.get("corruption_fraction", 0.0),
            ]
            writer.writerow(row)