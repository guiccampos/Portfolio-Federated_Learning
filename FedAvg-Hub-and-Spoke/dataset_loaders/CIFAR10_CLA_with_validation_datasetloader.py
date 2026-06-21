from flwr_datasets import FederatedDataset
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
import numpy as np

from dataset_loaders.cla_helper_utils import (
    build_client_owned_labels,
    build_faithful_class_imbalance_partitions,
    build_partition_summary,
    build_validation_summary,
    save_partition_summary_csv as save_partition_summary_csv_from_metadata,
    save_partition_summary_json as save_partition_summary_json_from_metadata,
    split_train_and_validation_indices,
)

federated_dataset = None
client_train_splits = None
client_test_splits = None
validation_split = None
partition_metadata = None


def load_data(
    partition_id: int,
    num_partitions: int,
    seed: int = 12345,
    batch_size: int = 128,
    alpha: float = 0.0,
    train_samples_per_client: int = 600,
    test_samples_per_client: int = 100,
    label_order_override: list[int] | None = None,
    validation_fraction: float = 0.10,
):
    global federated_dataset, client_train_splits, client_test_splits, validation_split, partition_metadata

    if federated_dataset is None:
        federated_dataset = FederatedDataset(
            dataset="uoft-cs/cifar10",
            partitioners={"train": 1, "test": 1},
            shuffle=True,
            seed=seed,
        )

    if client_train_splits is None or client_test_splits is None or validation_split is None:
        original_train_split = federated_dataset.load_split("train")
        full_test_split = federated_dataset.load_split("test")

        train_indices, validation_indices = split_train_and_validation_indices(
            num_samples=len(original_train_split),
            validation_fraction=validation_fraction,
            split_seed=seed
        )
        train_source_split = original_train_split.select(train_indices)
        validation_split = original_train_split.select(validation_indices)

        client_owned_labels = build_client_owned_labels(
            num_partitions=num_partitions,
            num_classes=10,
            label_order=label_order_override,
        )

        train_index_lists, train_metadata = build_faithful_class_imbalance_partitions(
            labels=np.asarray(train_source_split["label"]),
            client_owned_labels=client_owned_labels,
            samples_per_client=train_samples_per_client,
            sampling_seed=seed,
        )
        test_index_lists, test_metadata = build_faithful_class_imbalance_partitions(
            labels=np.asarray(full_test_split["label"]),
            client_owned_labels=client_owned_labels,
            samples_per_client=test_samples_per_client,
            sampling_seed=seed + 1,
        )

        client_train_splits = [train_source_split.select(index_list) for index_list in train_index_lists]
        client_test_splits = [full_test_split.select(index_list) for index_list in test_index_lists]

        partition_metadata = {
            "client_owned_labels": client_owned_labels,
            "train": train_metadata,
            "test": test_metadata,
            "validation": build_validation_summary(np.asarray(validation_split["label"])),
        }

    image_transform = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def apply_image_transform(batch):
        batch["img"] = [image_transform(image) for image in batch["img"]]
        return batch

    train_split_for_client = client_train_splits[partition_id].with_transform(apply_image_transform)
    validation_split_for_server = validation_split.with_transform(apply_image_transform)
    test_split_for_client = client_test_splits[partition_id].with_transform(apply_image_transform)

    train_loader = DataLoader(train_split_for_client, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_split_for_server, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_split_for_client, batch_size=batch_size, shuffle=False)
    return train_loader, validation_loader, test_loader


def get_partition_metadata():
    return partition_metadata


def get_partition_summary():
    return build_partition_summary(partition_metadata)


def save_partition_summary_json(output_path: str):
    save_partition_summary_json_from_metadata(partition_metadata, output_path)


def save_partition_summary_csv(output_path: str, num_classes: int = 10):
    save_partition_summary_csv_from_metadata(partition_metadata, output_path, num_classes=num_classes)
