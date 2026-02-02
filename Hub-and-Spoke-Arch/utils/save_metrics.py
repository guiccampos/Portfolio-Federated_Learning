import os
import csv
from utils.fl_types import ClientTestResult

SERVER_HEADERS = [
    "round",
    "time_to_aggregate_gradients",
    "time_to_update_server_model",
    "round_total_server_only_computation_time",
    "round_duration",
    "total_server_only_computation_time",
    "total_time",
    "round_clients_local_training_dataset_average_accuracy",
    "round_clients_local_testing_dataset_average_accuracy",
    "round_clients_central_testing_dataset_average_accuracy",
    "round_server_central_testing_dataset_loss",
    "round_server_central_testing_dataset_accuracy",
]

ROUND_HEADERS = ["round", "local_epochs", "learning_rate"]

CLIENT_SUFFIXES = [
    "training_loss",
    "training_accuracy",
    "training_dataset_size",
    "evaluation_loss",
    "evaluation_accuracy",
    "evaluation_dataset_size",
    "central_evaluation_loss",
    "central_evaluation_accuracy",
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
    time_to_aggregate_gradients: float,
    time_to_update_server_model: float,
    round_server_only_computation_time: float,
    round_duration: float,
    total_server_only_computation_time: float,
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
        time_to_aggregate_gradients,
        time_to_update_server_model,
        round_server_only_computation_time,
        round_duration,
        total_server_only_computation_time,
        total_time,
        round_clients_local_training_dataset_average_accuracy,
        round_clients_local_testing_dataset_average_accuracy,
        round_clients_central_testing_dataset_average_accuracy,
        round_server_central_testing_dataset_loss,
        round_server_central_testing_dataset_accuracy,
    ]

    with open(path, mode="a", newline="") as file:
        csv.writer(file).writerow(row)

def write_client_metrics(
    path: str,
    round_idx: int,
    client_testing_results: ClientTestResult,
    local_epochs: int,
    learning_rate: float,
) -> None:
    client_ids = [str(r["client_id"]) for r in client_testing_results]
    header = _ensure_csv_wide(path, client_ids)

    idx = {name: i for i, name in enumerate(header)}
    row = [""] * len(header)

    row[idx["round"]] = round_idx
    row[idx["local_epochs"]] = local_epochs
    row[idx["learning_rate"]] = learning_rate

    for r in client_testing_results:
        cid = str(r["client_id"])
        for sfx in CLIENT_SUFFIXES:
            row[idx[f"client_{cid}_{sfx}"]] = r[sfx]

    with open(path, "a", newline="") as f:
        csv.writer(f).writerow(row)

def _make_header(client_ids):
    header = ROUND_HEADERS[:]
    for cid in sorted(client_ids, key=int):
        for sfx in CLIENT_SUFFIXES:
            header.append(f"client_{cid}_{sfx}")
    return header


def _ensure_csv_wide(path, client_ids):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    if not os.path.exists(path):
        header = _make_header(client_ids)
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(header)
        return header

    with open(path, "r", newline="") as f:
        header = next(csv.reader(f))

    existing_ids = set()
    for col in header:
        if col.startswith("client_"):
            existing_ids.add(col.split("_", 2)[1])

    missing = set(client_ids) - existing_ids
    if not missing:
        return header

    new_header = header[:]
    for cid in sorted(missing, key=int):
        for sfx in CLIENT_SUFFIXES:
            new_header.append(f"client_{cid}_{sfx}")

    with open(path, "r", newline="") as f:
        rows = list(csv.reader(f))

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(new_header)
        new_width = len(new_header)
        for row in rows[1:]:
            row += [""] * (new_width - len(row))
            w.writerow(row)

    return new_header

