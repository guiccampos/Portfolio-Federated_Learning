from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
from dataset_loaders.SST2_IID_datasetloader import collect_and_combine

fds = None

# Defined how the data is going to be split across users
def load_centralized_data(partition_id: int,
              num_partitions: int,
              global_token_to_tid: dict[str, int],
              max_length: int,
              seed: int = 12345,
              batch_size: int = 128):    
    """Load partition Stanford Sentiment Tree 2 data."""
    global fds
    if fds is None:
        # IID partitioners for train and test
        train_partitioner = IidPartitioner(num_partitions=num_partitions)
        # validation_partitioner = IidPartitioner(num_partitions=num_partitions)
        # test_partitioner = IidPartitioner(num_partitions=num_partitions)

        # Download and partition SST2 from HF

        fds = FederatedDataset(                                                         # This Class downloads and partition data among client
            dataset="stanfordnlp/sst2",                                                 # Specify the name of the dataset in the hugging face hub
            partitioners={"train": train_partitioner,
                        #   "validation": validation_partitioner, 
                        #   "test": test_partitioner
                          },
            shuffle=True,
            seed=seed
        )

    # partition_train = fds.load_partition(partition_id, split="train")
    # partition_validation = fds.load_partition(partition_id, split="validation")
    # partition_test = fds.load_partition(partition_id, split="test")
    partition = fds.load_partition(partition_id, split="train")
    split_partition = partition.train_test_split(test_size=0.2, seed=seed)
    partition_test  = split_partition["test"]

    collate_fn = collect_and_combine(global_token_to_tid=global_token_to_tid, max_length=max_length)

    testloader  = DataLoader(partition_test,  batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return testloader
