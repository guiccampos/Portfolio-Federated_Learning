from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor

fds = None

# Defined how the data is going to be split across users
def load_centralized_data(partition_id: int, num_partitions: int, seed: int = 12345, batch_size: int = 128):
    """Load partition SVHN data."""
    global fds
    if fds is None:
        test_partitioner = IidPartitioner(num_partitions=num_partitions)
        
        fds = FederatedDataset(
            dataset="ufldl-stanford/svhn",
            subset="cropped_digits",
            partitioners={"test": test_partitioner},
            shuffle=True,
            seed=seed
        )

    partition_test = fds.load_partition(partition_id, split="test")
    pytorch_transforms = Compose(
        [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    def apply_transforms(batch):
        """Apply transforms to the partition from FederatedDataset."""
        batch["image"] = [pytorch_transforms(img) for img in batch["image"]]
        return batch

    partition_test = partition_test.with_transform(apply_transforms)
    testloader  = DataLoader(partition_test,  batch_size=batch_size, shuffle=False)
    return testloader