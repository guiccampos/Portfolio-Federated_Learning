from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import SizePartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
import numpy as np
import torch

S_PARAM     = .35        # Zipf/power-law exponent; higher => heavier skew

# CIFAR-10 totals and your minimums
TRAIN_TOTAL = 60_000
TEST_TOTAL  = 10_000
MIN_TRAIN   = 600
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

def _sizes_powerlaw(n_total: int,
                    n_clients: int,
                    min_size: int,
                    s: float) -> list[int]:
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

def _make_sizes(n_total: int, 
                n_clients: int, 
                min_size: int) -> list[int]:
    return _sizes_powerlaw(n_total, n_clients, min_size, S_PARAM)

# Defined how the data is going to be split across users
def load_noisy_data(
    partition_id: int,
    num_partitions: int,
    seed: int = 12345,
    batch_size: int = 128,
    num_corrupted_clients: int = 0,
    noise_std: float = 0.0,
    noise_prob: float = 1.0,
):
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
            dataset="ylecun/mnist",
            partitioners={"train": train_partitioner, "test": test_partitioner},
            shuffle=True,
            seed=seed
        )

    partition_train = fds.load_partition(partition_id, split="train")
    partition_test  = fds.load_partition(partition_id, split="test")

    # Clamp noise_prob to [0, 1] for safety
    if noise_prob < 0.0:
        noise_prob = 0.0
    if noise_prob > 1.0:
        noise_prob = 1.0

    to_tensor = ToTensor()
    normalize = Normalize((0.1307,), (0.3081,))

    # Per-client RNG stream so noise application is reproducible
    client_gen = torch.Generator()
    client_gen.manual_seed(seed + 10_000 * int(partition_id))


    def apply_transforms_train(batch):
        imgs = []
        for img in batch["image"]:
            x = to_tensor(img)  # [0, 1]

            if noise_std > 0.0 and noise_prob > 0.0:
                apply_noise = True
                if noise_prob < 1.0:
                    apply_noise = bool(torch.rand((), generator=client_gen).item() < noise_prob)

                if apply_noise:
                    eps = torch.randn(
                        x.shape,
                        generator=client_gen,
                        dtype=x.dtype,
                        device=x.device
                    ) * float(noise_std)
                    x = torch.clamp(x + eps, 0.0, 1.0)

            x = normalize(x)
            imgs.append(x)

        batch["image"] = imgs
        return batch

    def apply_transforms_test(batch):
        """Apply transforms to the TEST partition from FederatedDataset (kept clean)."""
        batch["image"] = [normalize(to_tensor(img)) for img in batch["image"]]
        return batch

    partition_train = partition_train.with_transform(apply_transforms_train)
    partition_test = partition_test.with_transform(apply_transforms_test)

    trainloader = DataLoader(partition_train, batch_size=batch_size, shuffle=True)
    testloader  = DataLoader(partition_test,  batch_size=batch_size, shuffle=False)
    return trainloader, testloader
