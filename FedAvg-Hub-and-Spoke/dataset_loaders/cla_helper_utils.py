import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


DEFAULT_VALIDATION_SPLIT_SEED = 1111


def build_client_owned_labels(
    num_partitions: int,
    num_classes: int = 10,
    label_order: List[int] | None = None,
) -> List[List[int]]:
    """Replicate the paper/code style nested class ownership schedule.

    Examples:
    - 5 clients  -> [1, 3, 5, 7, 10]
    - 10 clients -> [1, 2, 3, ..., 10]
    """
    if label_order is None:
        label_order = list(range(num_classes))

    class_counts_per_client = np.linspace(1, num_classes, num_partitions, dtype=int)
    return [label_order[:class_count] for class_count in class_counts_per_client.tolist()]


def split_train_and_validation_indices(
    num_samples: int,
    validation_fraction: float,
    split_seed: int = DEFAULT_VALIDATION_SPLIT_SEED,
) -> Tuple[List[int], List[int]]:
    """Replicate the code style global train/validation split before client partitioning."""
    all_indices = list(range(num_samples))
    rng = random.Random(split_seed)
    rng.shuffle(all_indices)

    train_fraction = 1.0 - validation_fraction
    train_cutoff = int(num_samples * train_fraction)

    train_indices = all_indices[:train_cutoff]
    validation_indices = all_indices[train_cutoff:]
    return train_indices, validation_indices


def build_label_to_indices(labels: np.ndarray) -> Dict[int, List[int]]:
    label_to_indices: Dict[int, List[int]] = {}
    for class_label in sorted(set(labels.tolist())):
        matching_indices = np.where(labels == class_label)[0].tolist()
        label_to_indices[int(class_label)] = matching_indices
    return label_to_indices


def build_faithful_class_imbalance_partitions(
    labels: np.ndarray,
    client_owned_labels: List[List[int]],
    samples_per_client: int,
    sampling_seed: int,
) -> Tuple[List[List[int]], dict]:
    """Replicate the original CFFL/CGSV class-imbalance code closely.

    Important behaviors preserved:
    - each client gets the same total number of samples
    - owned labels are nested and deterministic
    - per-class sampling uses random.choices(...), i.e. sampling WITH replacement
    - any leftover due to integer division is topped up from the last owned class
    """
    python_rng = random.Random(sampling_seed)
    label_to_indices = build_label_to_indices(labels)

    client_index_lists: List[List[int]] = []
    class_counts_per_client: List[Dict[int, int]] = []

    for client_id, owned_labels in enumerate(client_owned_labels):
        client_indices: List[int] = []
        client_class_counts: Dict[int, int] = {}

        num_owned_classes = len(owned_labels)
        samples_per_owned_class = samples_per_client // num_owned_classes

        for owned_label_position, class_label in enumerate(owned_labels):
            available_indices_for_label = label_to_indices[class_label]
            sampled_indices = python_rng.choices(
                available_indices_for_label,
                k=samples_per_owned_class,
            )
            client_indices.extend(sampled_indices)
            client_class_counts[class_label] = client_class_counts.get(class_label, 0) + len(sampled_indices)

            is_last_owned_label = owned_label_position == (num_owned_classes - 1)
            if is_last_owned_label and len(client_indices) < samples_per_client:
                extra_samples_needed = samples_per_client - len(client_indices)
                extra_indices = available_indices_for_label[:extra_samples_needed]
                client_indices.extend(extra_indices)
                client_class_counts[class_label] = client_class_counts.get(class_label, 0) + len(extra_indices)
                label_to_indices[class_label] = available_indices_for_label[extra_samples_needed:]

        client_index_lists.append(client_indices)
        class_counts_per_client.append(client_class_counts)

    metadata = {
        "samples_per_client": samples_per_client,
        "client_owned_labels": client_owned_labels,
        "class_counts_per_client": class_counts_per_client,
        "available_by_label": {label: int(len(indices)) for label, indices in label_to_indices.items()},
    }
    return client_index_lists, metadata


def build_partition_summary(metadata: dict) -> List[dict]:
    train_metadata = metadata["train"]
    test_metadata = metadata["test"]
    client_owned_labels = metadata["client_owned_labels"]

    summary_rows: List[dict] = []
    for client_id, owned_labels in enumerate(client_owned_labels):
        train_class_counts = train_metadata["class_counts_per_client"][client_id]
        test_class_counts = test_metadata["class_counts_per_client"][client_id]

        summary_rows.append(
            {
                "client_id": client_id,
                "num_classes": len(owned_labels),
                "labels": list(owned_labels),
                "train_total_samples": int(sum(train_class_counts.values())),
                "test_total_samples": int(sum(test_class_counts.values())),
                "train_class_counts": {int(label): int(count) for label, count in train_class_counts.items()},
                "test_class_counts": {int(label): int(count) for label, count in test_class_counts.items()},
            }
        )
    return summary_rows


def save_partition_summary_json(metadata: dict, output_path: str) -> None:
    summary_rows = build_partition_summary(metadata)
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as output_handle:
        json.dump(summary_rows, output_handle, indent=2)


def save_partition_summary_csv(metadata: dict, output_path: str, num_classes: int = 10) -> None:
    summary_rows = build_partition_summary(metadata)
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    csv_columns = [
        "client_id",
        "num_classes",
        "labels",
        "train_total_samples",
        "test_total_samples",
    ]
    csv_columns += [f"train_class_{class_label}" for class_label in range(num_classes)]
    csv_columns += [f"test_class_{class_label}" for class_label in range(num_classes)]

    with output_file.open("w", newline="", encoding="utf-8") as output_handle:
        writer = csv.DictWriter(output_handle, fieldnames=csv_columns)
        writer.writeheader()

        for summary_row in summary_rows:
            csv_row = {
                "client_id": summary_row["client_id"],
                "num_classes": summary_row["num_classes"],
                "labels": " ".join(map(str, summary_row["labels"])),
                "train_total_samples": summary_row["train_total_samples"],
                "test_total_samples": summary_row["test_total_samples"],
            }

            for class_label in range(num_classes):
                csv_row[f"train_class_{class_label}"] = summary_row["train_class_counts"].get(class_label, 0)
                csv_row[f"test_class_{class_label}"] = summary_row["test_class_counts"].get(class_label, 0)

            writer.writerow(csv_row)


def build_validation_summary(validation_labels: np.ndarray) -> dict:
    unique_labels, label_counts = np.unique(validation_labels, return_counts=True)
    return {
        "validation_total_samples": int(len(validation_labels)),
        "validation_class_counts": {
            int(class_label): int(count)
            for class_label, count in zip(unique_labels.tolist(), label_counts.tolist())
        },
    }
