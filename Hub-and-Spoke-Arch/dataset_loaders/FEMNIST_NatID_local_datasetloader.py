from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import NaturalIdPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor

fds = None

def load_data(
    partition_id: int,
    seed: int = 12345,
    batch_size: int = 128,
    local_test_fraction: float = 0.2,
    **kwargs,
):
    """Non-centralized: full dataset -> partition by writer_id -> local 80/20 split per writer."""
    global fds
    if fds is None:
        fds = FederatedDataset(
            dataset="flwrlabs/femnist",
            partitioners={"train": NaturalIdPartitioner(partition_by="writer_id")},
            shuffle=True,
            seed=seed,
        )

    # One writer partition
    partition = fds.load_partition(partition_id, split="train")

    # Deterministic per-writer split
    split = partition.train_test_split(test_size=local_test_fraction, seed=seed + int(partition_id))
    partition_train = split["train"]
    partition_test = split["test"]

    tfm = Compose([ToTensor(), Normalize((0.5,), (0.5,))])

    def apply_transforms(batch):
        batch["image"] = [tfm(im) for im in batch["image"]]
        batch["label"] = batch["character"]
        return batch

    partition_train = partition_train.with_transform(apply_transforms)
    partition_test = partition_test.with_transform(apply_transforms)

    trainloader = DataLoader(partition_train, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(partition_test, batch_size=batch_size, shuffle=False)
    return trainloader, testloader
