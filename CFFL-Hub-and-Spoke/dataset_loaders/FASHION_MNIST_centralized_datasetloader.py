from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor

fds = None

# Defined how the data is going to be split across users
def load_centralized_data(partition_id: int, num_partitions: int, seed: int = 12345, batch_size: int = 128):
    """Load partition FASHION_MNIST data."""
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        # Performs IID partitioning. Other partitioning formulas can be inserted here
        test_partitioner = IidPartitioner(num_partitions=num_partitions)

        # This Class downloads and partition data among client
        # I'm doing an IID partitioning based on the number of partitions and then each partition is associated with a partitioner
        fds = FederatedDataset(                                                         # This Class downalods and partitiong data among client
            dataset="zalando-datasets/fashion_mnist",                                   # Specify the name of the dataset in the hugging face hub
            partitioners={"test": test_partitioner},        # Python Dict mapping the Dataset Split to a Partitioner
            shuffle=True,
            seed=seed
        )
                                                                                        
    partition_test = fds.load_partition(partition_id, split="test")
    pytorch_transforms = Compose([ToTensor(), Normalize((0.2860,), (0.3530,))])

    def apply_transforms(batch):
        """Apply transforms to the partition from FederatedDataset."""
        batch["image"] = [pytorch_transforms(image) for image in batch["image"]]
        return batch

    partition_test = partition_test.with_transform(apply_transforms)
    testloader  = DataLoader(partition_test,  batch_size=batch_size, shuffle=False)
    return testloader