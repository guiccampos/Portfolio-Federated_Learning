import os
import csv
import numpy as np
import numpy.typing as npt
from utils.fl_types import ClientTestResult
from typing import Any

SERVER_HEADERS = [
    "round",
    "time_spent_aggregating_gradients",
    "time_spent_updating_server_model",
    "time_spent_testing_global_model",
    "time_spent_in_train_instructions_broadcast",
    "time_spent_in_collecting_training_results",
    "time_spent_in_eval_instructions_broadcast",
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
    "client_i_training_loss",
    "client_i_training_accuracy",
    "client_i_training_dataset_size",
    "client_i_evaluation_loss",
    "client_i_evaluation_accuracy",
    "client_i_evaluation_dataset_size",
    "client_i_central_evaluation_loss",
    "client_i_central_evaluation_accuracy",
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
    round: int,
    time_spent_aggregating_gradients: float,
    time_spent_updating_server_model: float,
    time_spent_testing_global_model: float,
    time_spent_in_train_instructions_broadcast: float,
    time_spent_in_collecting_training_results: float,
    time_spent_in_eval_instructions_broadcast: float,
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
        round,
        time_spent_aggregating_gradients,
        time_spent_updating_server_model,
        time_spent_testing_global_model,
        time_spent_in_train_instructions_broadcast,
        time_spent_in_collecting_training_results,
        time_spent_in_eval_instructions_broadcast,
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
) -> None:
    # Build dynamic headers (wide pairwise columns)
    headers = ["round", "client_id"] + CLIENT_HEADERS

    _ensure_csv(path, headers)

    # Fast lookups by client_id
    test_by_id  = {r["client_id"]: r for r in client_results_test if isinstance(r, dict) and "client_id" in r}

    # Union of all client_ids seen in any of the three sources
    all_ids = sorted(set(test_by_id))

    with open(path, mode="a", newline="") as f:
        w = csv.writer(f)
        for cid in all_ids:
            te = test_by_id.get(cid, {})
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
            ]

            w.writerow(row)