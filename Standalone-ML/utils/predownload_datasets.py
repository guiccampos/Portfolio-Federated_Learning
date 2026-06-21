import os
from pathlib import Path

from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner


def predownload_mnist(
    num_partitions: int = 10,
    seed: int = 12345,
    hf_home: str | None = None,
):
    # Choose cache location
    if hf_home is None:
        user = os.environ.get("USER", "user")
        hf_home = f"/scratch/{user}/hf_cache"

    os.environ["HF_HOME"] = hf_home
    cache_dir = os.path.join(hf_home, "datasets")
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    print(f"[INFO] HF_HOME={hf_home}")
    print(f"[INFO] cache_dir={cache_dir}")

    # Create a FederatedDataset exactly like your runtime pattern
    fds = FederatedDataset(
        dataset="ylecun/mnist",
        partitioners={
            "train": IidPartitioner(num_partitions=num_partitions),
            "test": IidPartitioner(num_partitions=num_partitions),
        },
        shuffle=True,
        seed=seed,
        cache_dir=cache_dir,
    )

    print("[INFO] Downloading/caching train split...")
    train_part = fds.load_partition(0, split="train")
    print(f"[INFO] Train partition 0 size: {len(train_part)}")

    print("[INFO] Downloading/caching test split...")
    test_part = fds.load_partition(0, split="test")
    print(f"[INFO] Test partition 0 size: {len(test_part)}")

    print("[SUCCESS] MNIST is cached and ready for offline Slurm jobs.")

def predownload_fashion_mnist(
    num_partitions: int = 10,
    seed: int = 12345,
    hf_home: str | None = None,
):
    # Choose cache location
    if hf_home is None:
        user = os.environ.get("USER", "user")
        hf_home = f"/scratch/{user}/hf_cache"

    os.environ["HF_HOME"] = hf_home
    cache_dir = os.path.join(hf_home, "datasets")
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    print(f"[INFO] HF_HOME={hf_home}")
    print(f"[INFO] cache_dir={cache_dir}")

    # Create a FederatedDataset exactly like your runtime pattern
    fds = FederatedDataset(
        dataset="zalando-datasets/fashion_mnist",
        partitioners={
            "train": IidPartitioner(num_partitions=num_partitions),
            "test": IidPartitioner(num_partitions=num_partitions),
        },
        shuffle=True,
        seed=seed,
        cache_dir=cache_dir,
    )

    print("[INFO] Downloading/caching train split...")
    train_part = fds.load_partition(0, split="train")
    print(f"[INFO] Train partition 0 size: {len(train_part)}")

    print("[INFO] Downloading/caching test split...")
    test_part = fds.load_partition(0, split="test")
    print(f"[INFO] Test partition 0 size: {len(test_part)}")

    print("[SUCCESS] Fashion-MNIST is cached and ready for offline Slurm jobs.")

def predownload_cifar10(
    num_partitions: int = 10,
    seed: int = 12345,
    hf_home: str | None = None,
):
    # Choose cache location
    if hf_home is None:
        user = os.environ.get("USER", "user")
        hf_home = f"/scratch/{user}/hf_cache"

    os.environ["HF_HOME"] = hf_home
    cache_dir = os.path.join(hf_home, "datasets")
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    print(f"[INFO] HF_HOME={hf_home}")
    print(f"[INFO] cache_dir={cache_dir}")

    # Create a FederatedDataset exactly like your runtime pattern
    fds = FederatedDataset(
        dataset="uoft-cs/cifar10",
        partitioners={
            "train": IidPartitioner(num_partitions=num_partitions),
            "test": IidPartitioner(num_partitions=num_partitions),
        },
        shuffle=True,
        seed=seed,
        cache_dir=cache_dir,
    )

    print("[INFO] Downloading/caching train split...")
    train_part = fds.load_partition(0, split="train")
    print(f"[INFO] Train partition 0 size: {len(train_part)}")

    print("[INFO] Downloading/caching test split...")
    test_part = fds.load_partition(0, split="test")
    print(f"[INFO] Test partition 0 size: {len(test_part)}")

    print("[SUCCESS] CIFAR10 is cached and ready for offline Slurm jobs.")

def predownload_SST2(
    num_partitions: int = 10,
    seed: int = 12345,
    hf_home: str | None = None,
):
    # Choose cache location
    if hf_home is None:
        user = os.environ.get("USER", "user")
        hf_home = f"/scratch/{user}/hf_cache"

    os.environ["HF_HOME"] = hf_home
    cache_dir = os.path.join(hf_home, "datasets")
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    print(f"[INFO] HF_HOME={hf_home}")
    print(f"[INFO] cache_dir={cache_dir}")

    # Create a FederatedDataset exactly like your runtime pattern
    fds = FederatedDataset(
        dataset="stanfordnlp/sst2",
        partitioners={
            "train": IidPartitioner(num_partitions=num_partitions),
            "test": IidPartitioner(num_partitions=num_partitions),
        },
        shuffle=True,
        seed=seed,
        cache_dir=cache_dir,
    )

    print("[INFO] Downloading/caching train split...")
    train_part = fds.load_partition(0, split="train")
    print(f"[INFO] Train partition 0 size: {len(train_part)}")

    print("[INFO] Downloading/caching test split...")
    test_part = fds.load_partition(0, split="test")
    print(f"[INFO] Test partition 0 size: {len(test_part)}")

    print("[SUCCESS] SST2 is cached and ready for offline Slurm jobs.")

def predownload_SVHN(
    num_partitions: int = 10,
    seed: int = 12345,
    hf_home: str | None = None,
):
    # Choose cache location
    if hf_home is None:
        user = os.environ.get("USER", "user")
        hf_home = f"/scratch/{user}/hf_cache"

    os.environ["HF_HOME"] = hf_home
    cache_dir = os.path.join(hf_home, "datasets")
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    print(f"[INFO] HF_HOME={hf_home}")
    print(f"[INFO] cache_dir={cache_dir}")

    # Create a FederatedDataset exactly like your runtime pattern
    fds = FederatedDataset(
            dataset="ufldl-stanford/svhn",
            subset="cropped_digits",
        partitioners={
            "train": IidPartitioner(num_partitions=num_partitions),
            "test": IidPartitioner(num_partitions=num_partitions),
        },
        shuffle=True,
        seed=seed,
        cache_dir=cache_dir,
    )

    train_part = fds.load_partition(0, split="train")
    test_part = fds.load_partition(0, split="test")

if __name__ == "__main__":
    # predownload_mnist(num_partitions=10, seed=12345)
    # predownload_fashion_mnist(num_partitions=10, seed=12345)
    # predownload_cifar10(num_partitions=10, seed=12345)
    # predownload_SST2(num_partitions=10, seed=12345)
    predownload_SVHN()