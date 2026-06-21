from flwr_datasets import FederatedDataset
from flwr_datasets.preprocessor import Divider
from flwr_datasets.partitioner import IidPartitioner, InnerDirichletPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
from typing import Dict, Tuple

# Cache one FederatedDataset per configuration so repeated calls with the same
# settings do not repartition inside the same Python process.
_FDS_CACHE: Dict[Tuple[int, int, float, float], FederatedDataset] = {}

FASHION_MNIST_TRAIN_SIZE = 60_000
FASHION_MNIST_TEST_SIZE = 10_000


def load_data(
    partition_id: int,
    num_partitions: int,
    seed: int = 1111,
    batch_size: int = 128,
    alpha: float = 0.5,
    validation_fraction: float = 0.10,
):
    """Load one Fashion-MNIST client partition with InnerDirichlet on train split and a centralized validation split.

    Reproducibility notes
    ---------------------
    - Keeping the same seed, alpha, num_partitions, and validation_fraction will
      reproduce the same partitioning in the same environment.
    - Changing the seed will deterministically generate a different InnerDirichlet
      realization for that new seed.
    """
    fds = _get_fds(
        num_partitions=num_partitions,
        seed=seed,
        alpha=alpha,
        validation_fraction=validation_fraction,
    )

    partition_train = fds.load_partition(partition_id, split="train")
    partition_valid = fds.load_split("valid")
    partition_test = fds.load_partition(partition_id, split="test")

    pytorch_transforms = Compose([ToTensor(), Normalize((0.2860,), (0.3530,))])

    def apply_transforms(batch):
        batch["image"] = [pytorch_transforms(x) for x in batch["image"]]
        return batch

    partition_train = partition_train.with_transform(apply_transforms)
    partition_valid = partition_valid.with_transform(apply_transforms)
    partition_test = partition_test.with_transform(apply_transforms)

    trainloader = DataLoader(partition_train, batch_size=batch_size, shuffle=True)
    validloader = DataLoader(partition_valid, batch_size=batch_size, shuffle=False)
    testloader = DataLoader(partition_test, batch_size=batch_size, shuffle=False)
    return trainloader, validloader, testloader


def _equal_partition_sizes(total_size: int, num_partitions: int) -> list[int]:
    base = total_size // num_partitions
    remainder = total_size % num_partitions
    return [base + (1 if pid < remainder else 0) for pid in range(num_partitions)]

def _get_fds(
    num_partitions: int,
    seed: int,
    alpha: float,
    validation_fraction: float,
    train_partition_sizes: list[int] | None = None,
) -> FederatedDataset:
    cache_key = (num_partitions, seed, float(alpha), float(validation_fraction))
    if cache_key in _FDS_CACHE:
        return _FDS_CACHE[cache_key]

    valid_n = int(FASHION_MNIST_TRAIN_SIZE * validation_fraction)
    train_n = FASHION_MNIST_TRAIN_SIZE - valid_n

    if train_partition_sizes is None:
        train_partition_sizes = _equal_partition_sizes(
            total_size=train_n, num_partitions=num_partitions
        )

    divider = Divider(
        divide_config={"train": train_n, "valid": valid_n},
        divide_split="train",
        drop_remaining_splits=False,
    )

    train_partitioner = InnerDirichletPartitioner(
        partition_sizes=train_partition_sizes,
        partition_by="label",
        alpha=alpha,
        shuffle=True,
        seed=seed,
    )
    test_partitioner = IidPartitioner(num_partitions=num_partitions)

    fds = FederatedDataset(
        dataset="zalando-datasets/fashion_mnist",
        partitioners={"train": train_partitioner, "test": test_partitioner},
        preprocessor=divider,
        shuffle=True,
        seed=seed,
    )
    _FDS_CACHE[cache_key] = fds
    return fds
