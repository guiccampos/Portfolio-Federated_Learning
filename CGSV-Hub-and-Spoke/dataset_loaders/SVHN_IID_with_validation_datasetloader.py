from flwr_datasets import FederatedDataset
from flwr_datasets.preprocessor import Divider
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor

fds = None

# Defined how the data is going to be split across users
def load_data(partition_id: int, num_partitions: int, seed: int = 12345, batch_size: int = 128, validation_fraction: float = 0.10):
    """Load partition SVHN data."""
    global fds
    if fds is None:

        divider = Divider(
            divide_config={"train": 1.0 - validation_fraction, "valid": validation_fraction},
            divide_split="train",
            drop_remaining_splits=False,  # keep "test" split
        )

        # IID partitioners for train and test
        train_partitioner = IidPartitioner(num_partitions=num_partitions)
        test_partitioner = IidPartitioner(num_partitions=num_partitions)

        # Download and partition SVHN from HF
        fds = FederatedDataset(                                                         # This Class downloads and partition data among client
            dataset="ufldl-stanford/svhn",
            subset="cropped_digits",
            partitioners={"train": train_partitioner, "test": test_partitioner},        # Python Dict mapping the Dataset Split to a Partitioner
            preprocessor=divider,
            shuffle=True,
            seed=seed
        )

    partition_train = fds.load_partition(partition_id, split="train")
    partition_valid = fds.load_split("valid")
    partition_test = fds.load_partition(partition_id, split="test")
    pytorch_transforms = Compose(
        [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    def apply_transforms(batch):
        """Apply transforms to the partition from FederatedDataset."""
        batch["image"] = [pytorch_transforms(img) for img in batch["image"]]
        return batch

    partition_train = partition_train.with_transform(apply_transforms)
    partition_valid = partition_valid.with_transform(apply_transforms)
    partition_test = partition_test.with_transform(apply_transforms)
    trainloader = DataLoader(partition_train, batch_size=batch_size, shuffle=True)
    validloader = DataLoader(partition_valid, batch_size=batch_size, shuffle=False)
    testloader  = DataLoader(partition_test,  batch_size=batch_size, shuffle=False)
    return trainloader, validloader, testloader
