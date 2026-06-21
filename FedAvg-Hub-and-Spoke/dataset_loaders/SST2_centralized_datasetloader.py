from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from flwr_datasets.preprocessor import Divider
from torch.utils.data import DataLoader
from datasets import concatenate_datasets
from dataset_loaders.SST2_IID_with_validation_datasetloader import collect_and_combine

fds = None

CENTRAL_TEST_FRACTION = 0.20
SST2_TRAIN_TOTAL = 67349
DESIRED_VALID_SIZE = 5000
HF_VALID_SIZE_EST = 872

def _build_divider(seed: int) -> tuple[Divider, int, int]:
    base = FederatedDataset(
        dataset="stanfordnlp/sst2",
        partitioners={"train": IidPartitioner(num_partitions=1)},
        shuffle=True,
        seed=seed,
    )
    n_total = len(base.load_partition(0, split="train"))

    central_test_size = int(n_total * CENTRAL_TEST_FRACTION)
    central_valid_extra_size = max(0, DESIRED_VALID_SIZE - HF_VALID_SIZE_EST)
    train_pool_size = n_total - central_test_size - central_valid_extra_size

    divider = Divider(
        divide_config={
            "train": train_pool_size,
            "central_valid_extra": central_valid_extra_size,
            "central_test": central_test_size,
        },
        divide_split="train",
    )
    return divider, train_pool_size, central_valid_extra_size

def load_centralized_data(
    partition_id: int,
    num_partitions: int,
    global_token_to_tid: dict[str, int],
    max_length: int,
    seed: int = 12345,
    batch_size: int = 128,
):
    global fds

    if fds is None:
        divider, _, _ = _build_divider(seed)
        central_partitioner = IidPartitioner(num_partitions=1)

        fds = FederatedDataset(
            dataset="stanfordnlp/sst2",
            partitioners={"central_test": central_partitioner},
            preprocessor=divider,
            shuffle=True,
            seed=seed,
        )

    central_test = fds.load_partition(0, split="central_test")
    collate_fn = collect_and_combine(global_token_to_tid=global_token_to_tid, max_length=max_length)
    return DataLoader(central_test, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)