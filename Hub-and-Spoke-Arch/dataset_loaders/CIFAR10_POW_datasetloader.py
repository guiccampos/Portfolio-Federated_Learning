from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import SizePartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
import numpy as np

S_PARAM     = .35         # Zipf/power-law exponent; higher => heavier skew

# CIFAR-10 totals and your minimums
TRAIN_TOTAL = 50_000
TEST_TOTAL  = 10_000
MIN_TRAIN   = 500
MIN_TEST    = 100

fds = None 

def _largest_remainder(total: int, weights: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, dtype=float)
    p = w / w.sum()
    quotas = total * p
    base = np.floor(quotas).astype(int)
    rem = int(total - base.sum())
    if rem > 0:
        frac = quotas - base
        # take the indices with largest fractional remainders
        idx = np.argpartition(-frac, range(rem))[:rem]
        base[idx] += 1
    return base

def _sizes_powerlaw(n_total: int, n_clients: int, min_size: int, s: float) -> list[int]:
    if min_size * n_clients > n_total:
        raise ValueError("min_size * n_clients exceeds split total")
    remaining = n_total - min_size * n_clients
    if remaining == 0:
        return [min_size] * n_clients
    ranks = np.arange(1, n_clients + 1, dtype=float)    # 1..N
    weights = ranks ** (-s)                              # Zipf weights
    extra = _largest_remainder(remaining, weights)
    sizes = (extra + min_size).astype(int)
    return sizes.tolist()

def _make_sizes(n_total: int, n_clients: int, min_size: int) -> list[int]:
    return _sizes_powerlaw(n_total, n_clients, min_size, S_PARAM)

# Defined how the data is going to be split across users
def load_data(partition_id: int, num_partitions: int, seed: int = 12345, batch_size: int = 128):
    """Load partition CIFAR10 data (quantity-skew by size, labels ~IID)."""
    global fds
    if fds is None:
        # Build quantity-skewed sizes with hard minimums and exact totals
        train_sizes = _make_sizes(TRAIN_TOTAL, num_partitions, MIN_TRAIN)
        test_sizes  = _make_sizes(TEST_TOTAL,  num_partitions, MIN_TEST)

        # Partitioners using the exact sizes we computed
        train_partitioner = SizePartitioner(partition_sizes=train_sizes)
        test_partitioner  = SizePartitioner(partition_sizes=test_sizes)

        # This class downloads and partitions data among clients
        fds = FederatedDataset(
            dataset="uoft-cs/cifar10",
            partitioners={"train": train_partitioner, "test": test_partitioner},
            shuffle=True,
            seed=seed
        )

    partition_train = fds.load_partition(partition_id, split="train")
    partition_test  = fds.load_partition(partition_id, split="test")

    pytorch_transforms = Compose(
        [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    def apply_transforms(batch):
        """Apply transforms to the partition from FederatedDataset."""
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch

    partition_train = partition_train.with_transform(apply_transforms)
    partition_test  = partition_test.with_transform(apply_transforms)

    trainloader = DataLoader(partition_train, batch_size=batch_size, shuffle=True)
    testloader  = DataLoader(partition_test,  batch_size=batch_size, shuffle=False)
    return trainloader, testloader
