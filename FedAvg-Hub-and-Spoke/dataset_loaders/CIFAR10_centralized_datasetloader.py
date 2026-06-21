from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor

fds = None

# Defined how the data is going to be split across users
def load_centralized_data(partition_id: int, num_partitions: int, seed: int = 12345, batch_size: int = 128):
    """Load partition CIFAR10 data."""
    global fds
    if fds is None:
        #print("fds is None, creating FederatedDataset")
        # Performs IID partitioning. Other partitioning formulas can be inserted here
        test_partitioner = IidPartitioner(num_partitions=num_partitions)
        
        # This Class downloads and partition data among client
        # I'm doing an IID partitioning based on the number of partitions and then each partition is associated with a partitioner
        fds = FederatedDataset(                                                         # This Class downloads and partition data among client
            dataset="uoft-cs/cifar10",                                                  # Specify the name of the dataset in the hugging face hub
            partitioners={"test": test_partitioner},        # Python Dict mapping the Dataset Split to a Partitioner
            shuffle=True,
            seed=seed
        )

    partition_test = fds.load_partition(partition_id, split="test")
    pytorch_transforms = Compose(
        [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    def apply_transforms(batch):
        """Apply transforms to the partition from FederatedDataset."""
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch

    partition_test = partition_test.with_transform(apply_transforms)
    testloader  = DataLoader(partition_test,  batch_size=batch_size, shuffle=False)
    return testloader
