from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor

fds = None

def load_data(partition_id: int, num_partitions: int, seed: int = 12345, batch_size: int = 128):
    """Load partition CIFAR100 data with IID partitioning."""
    global fds
    if fds is None:
        # IID partitioners for train and test
        train_partitioner = IidPartitioner(num_partitions=num_partitions)
        test_partitioner = IidPartitioner(num_partitions=num_partitions)

        # Download and partition CIFAR100 from HF
        fds = FederatedDataset(
            dataset="uoft-cs/cifar100",  # HF dataset name
            partitioners={"train": train_partitioner, "test": test_partitioner},
            shuffle=True,
            seed=seed,
        )

    # Load this client's partitions
    partition_train = fds.load_partition(partition_id, split="train")
    partition_test = fds.load_partition(partition_id, split="test")
    pytorch_transforms = Compose(
        [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    def apply_transforms(batch):
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch

    partition_train = partition_train.with_transform(apply_transforms)
    partition_test = partition_test.with_transform(apply_transforms)

    trainloader = DataLoader(partition_train, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(partition_test, batch_size=batch_size, shuffle=False)
    return trainloader, testloader
