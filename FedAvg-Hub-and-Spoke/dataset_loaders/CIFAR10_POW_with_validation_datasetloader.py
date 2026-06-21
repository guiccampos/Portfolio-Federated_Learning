from flwr_datasets import FederatedDataset
from flwr_datasets.preprocessor import Divider
from flwr_datasets.partitioner import SizePartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
import numpy as np
import math
from scipy.stats import powerlaw as sp_powerlaw

PAPER_ALPHA = 1.65911332899

# CIFAR-10 totals and your minimums
TRAIN_TOTAL = 50_000
TEST_TOTAL  = 10_000

fds = None 

# Defined how the data is going to be split across users
def load_data(partition_id: int, num_partitions: int, seed: int = 12345, batch_size: int = 128, validation_fraction: float = 0.10):
    """Load partition CIFAR10 data (quantity-skew by size)."""
    global fds
    if fds is None:

        valid_n = int(TRAIN_TOTAL * validation_fraction)
        train_n = TRAIN_TOTAL - valid_n

        divider = Divider(
            divide_config={"train": train_n, "valid": valid_n},
            divide_split="train",
            drop_remaining_splits=False,  # keep "test" split
        )
        
        # Build quantity-skewed sizes with hard minimums and exact totals
        train_sizes = paper_sizes_topup_largest(train_n, num_partitions, alpha=PAPER_ALPHA)
        test_sizes  = paper_sizes_topup_largest(TEST_TOTAL,  num_partitions, alpha=PAPER_ALPHA)

        # Partitioners using the exact sizes we computed
        train_partitioner = SizePartitioner(partition_sizes=train_sizes)
        test_partitioner  = SizePartitioner(partition_sizes=test_sizes)

        # This class downloads and partitions data among clients
        fds = FederatedDataset(
            dataset="uoft-cs/cifar10",
            partitioners={"train": train_partitioner, "test": test_partitioner},
            preprocessor=divider,
            shuffle=True,
            seed=seed
        )

    partition_train = fds.load_partition(partition_id, split="train")
    partition_valid = fds.load_split("valid")
    partition_test  = fds.load_partition(partition_id, split="test")

    pytorch_transforms = Compose(
        [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    def apply_transforms(batch):
        """Apply transforms to the partition from FederatedDataset."""
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch

    partition_train = partition_train.with_transform(apply_transforms)
    partition_valid = partition_valid.with_transform(apply_transforms)
    partition_test = partition_test.with_transform(apply_transforms)
    trainloader = DataLoader(partition_train, batch_size=batch_size, shuffle=True)
    validloader = DataLoader(partition_valid, batch_size=batch_size, shuffle=False)
    testloader  = DataLoader(partition_test,  batch_size=batch_size, shuffle=False)
    return trainloader, validloader, testloader

def paper_sizes_topup_largest(n_total: int, n_clients: int, alpha: float = PAPER_ALPHA) -> list[int]:
    """
    Paper-like sizes, but uses ALL samples by giving any leftover to the largest client.
    """
    N, K = n_total, n_clients
    if K <= 0:
        raise ValueError("n_clients must be > 0")
    if N < K:
        raise ValueError("n_total must be >= n_clients (so each client can get >= 1 sample)")

    # Paper scaling base (floor)
    party_size = int(N / K)
    base = party_size * K

    b = np.linspace(
        sp_powerlaw.ppf(0.01, alpha),
        sp_powerlaw.ppf(0.99, alpha),
        K,
    )

    sizes = [math.ceil(x / b.sum() * base) for x in b]  # same as paper (pre-slicing)
    used = sum(sizes)

    if used > N:
        # Paper would effectively truncate the tail; do the same first
        overflow = used - N
        sizes[-1] -= overflow
        used = N

    if used < N:
        # Leftover that paper would drop → top up the largest client instead
        leftover = N - used
        j = int(np.argmax(sizes))
        sizes[j] += leftover

    # Safety: ensure all > 0
    if any(s <= 0 for s in sizes):
        raise ValueError(f"Invalid partition size encountered: {sizes}")

    # Uses all samples
    assert sum(sizes) == N
    return list(reversed(sizes))