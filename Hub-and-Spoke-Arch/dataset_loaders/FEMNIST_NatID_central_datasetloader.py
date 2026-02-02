from typing import Any
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import NaturalIdPartitioner
from flwr_datasets.preprocessor import Divider
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor

_fds: FederatedDataset | None = None

def _get_fds(seed: int, pseudo_test_fraction: float) -> FederatedDataset:
    global _fds
    if _fds is None:
        # Split the single HF "train" split into ("train_pool", "pseudo_test")
        divider = Divider(
            divide_config={
                "train_pool": 1.0 - pseudo_test_fraction,
                "pseudo_test": pseudo_test_fraction,
            },
            divide_split="train",  # explicit is fine; can be inferred if single-split
        )

        _fds = FederatedDataset(
            dataset="flwrlabs/femnist",
            preprocessor=divider,  # <-- correct API in new versions
            partitioners={
                "train_pool": NaturalIdPartitioner(partition_by="writer_id"),
                "pseudo_test": NaturalIdPartitioner(partition_by="writer_id"),
            },
            shuffle=True,
            seed=seed,
        )
    return _fds

def _apply_transforms(ds):
    tfm = Compose([ToTensor(), Normalize((0.5,), (0.5,))])

    def apply(batch):
        batch["image"] = [tfm(im) for im in batch["image"]]
        batch["label"] = batch["character"]
        return batch

    return ds.with_transform(apply)

def load_data(
    partition_id: int,
    seed: int = 12345,
    batch_size: int = 128,
    pseudo_test_fraction: float = 0.2,
    **kwargs: Any,
):
    """Client loaders: train_pool partition + pseudo_test partition, both by writer_id."""
    fds = _get_fds(seed=seed, pseudo_test_fraction=pseudo_test_fraction)

    train_part = _apply_transforms(fds.load_partition(partition_id, split="train_pool"))
    test_part = _apply_transforms(fds.load_partition(partition_id, split="pseudo_test"))

    trainloader = DataLoader(train_part, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test_part, batch_size=batch_size, shuffle=False)
    return trainloader, testloader

def load_centralized_pseudo_test(
    seed: int = 12345,
    batch_size: int = 128,
    pseudo_test_fraction: float = 0.2,
):
    """Server/global eval loader: full pseudo_test split (not partitioned)."""
    fds = _get_fds(seed=seed, pseudo_test_fraction=pseudo_test_fraction)

    # Newer Flower Datasets exposes load_split; PyPI docs mention it. :contentReference[oaicite:2]{index=2}
    try:
        pseudo_test = fds.load_split("pseudo_test")
    except AttributeError:
        # Fallback if load_split isn't available in your exact build:
        # load the raw split from the underlying DatasetDict
        pseudo_test = fds.dataset["pseudo_test"]

    pseudo_test = _apply_transforms(pseudo_test)
    return DataLoader(pseudo_test, batch_size=batch_size, shuffle=False)
