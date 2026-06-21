from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from flwr_datasets.preprocessor import Divider
from torch.utils.data import DataLoader
from collections import Counter
from datasets import concatenate_datasets
from typing import List
import torch
import re

fds = None

CENTRAL_TEST_FRACTION = 0.20
SST2_TRAIN_TOTAL = 67349  # stable for stanfordnlp/sst2
DESIRED_VALID_SIZE = 5000
HF_VALID_SIZE_EST = 872

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"

def _build_divider(seed: int) -> tuple[Divider, int, int]:
    base = FederatedDataset(
        dataset="stanfordnlp/sst2",
        partitioners={"train": IidPartitioner(num_partitions=1)},
        shuffle=True,
        seed=seed,
    )
    n_total = len(base.load_partition(0, split="train"))

    central_test_size = int(n_total * CENTRAL_TEST_FRACTION)

    # top-up amount
    central_valid_extra_size = max(0, DESIRED_VALID_SIZE - HF_VALID_SIZE_EST)

    # remaining train pool for clients
    train_pool_size = n_total - central_test_size - central_valid_extra_size
    if train_pool_size <= 0:
        raise ValueError("Not enough samples after reserving central_test and central_valid_extra")

    divider = Divider(
        divide_config={
            "train": train_pool_size,
            "central_valid_extra": central_valid_extra_size,
            "central_test": central_test_size,
        },
        divide_split="train",
    )
    return divider, train_pool_size, central_valid_extra_size

def load_data(
    partition_id: int,
    num_partitions: int,
    global_token_to_tid: dict[str, int],
    max_length: int,
    seed: int = 12345,
    batch_size: int = 128,
    alpha: float = 0.0,
):
    """
    Returns:
      trainloader: client shard from the 80% train pool
      validloader: HF's built-in validation split (shared)
      testloader: empty (centralized test comes from SST2_centralized_datasetloader)
    """
    global fds
    extra_sz = max(0, DESIRED_VALID_SIZE - HF_VALID_SIZE_EST)
    if fds is None:
        divider, _train_pool_size, _ = _build_divider(seed)
        train_partitioner = IidPartitioner(num_partitions=num_partitions)

        fds = FederatedDataset(
            dataset="stanfordnlp/sst2",
            partitioners={"train": train_partitioner},
            preprocessor=divider,
            shuffle=True,
            seed=seed,
        )

    # Client train data from the 80% pool
    partition_train = fds.load_partition(partition_id, split="train")

    # Shared validation (pre-existing SST2 validation split)
    hf_valid = fds.load_split("validation")
    if extra_sz > 0:
        extra_valid = fds.load_split("central_valid_extra")
        central_valid = concatenate_datasets([hf_valid, extra_valid])
    else:
        central_valid = hf_valid

    collate_fn = collect_and_combine(global_token_to_tid=global_token_to_tid, max_length=max_length)

    trainloader = DataLoader(partition_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    validloader = DataLoader(central_valid,   batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Empty testloader (centralized test is loaded separately by the server)
    empty = partition_train.select([])
    testloader = DataLoader(empty, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return trainloader, validloader, testloader


# ---------------- Tokenization / collate (unchanged) ----------------

def tokenize(sentence: str) -> List[str]:
    return re.findall(r"[a-z]+(?:'[a-z]+)?", sentence.lower())

def build_global_token_tid_map(seed: int, embed_num: int) -> dict[str, int]:
    # NOTE: This uses full HF train split. If you want to exclude central_test tokens,
    # we can build vocab from the 80% train pool instead.
    train_partitioner = IidPartitioner(num_partitions=1)
    global_fds = FederatedDataset(
        dataset="stanfordnlp/sst2",
        partitioners={"train": train_partitioner},
        shuffle=True,
        seed=seed
    )
    partition = global_fds.load_partition(0, split="train")

    global_token_to_tid = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    counter = Counter()
    for sample in partition:
        counter.update(tokenize(sample["sentence"]))

    max_size = embed_num - 2
    idx = 2
    for token, _freq in counter.most_common(max_size):
        global_token_to_tid[token] = idx
        idx += 1

    return global_token_to_tid

def sentence_to_ids(sentence: str, global_token_to_tid: dict, max_length: int) -> List[int]:
    tokens = tokenize(sentence)
    pad_id = global_token_to_tid[PAD_TOKEN]
    unk_id = global_token_to_tid[UNK_TOKEN]

    token_ids: List[int] = []
    for t in tokens[:max_length]:
        token_ids.append(global_token_to_tid.get(t, unk_id))

    while len(token_ids) < max_length:
        token_ids.append(pad_id)

    return token_ids

def collect_and_combine(global_token_to_tid: dict, max_length: int):
    def collate(batch):
        input_ids_list, labels_list = [], []
        for sample in batch:
            input_ids_list.append(sentence_to_ids(sample["sentence"], global_token_to_tid, max_length))
            labels_list.append(int(sample["label"]))
        return (
            torch.tensor(input_ids_list, dtype=torch.long),
            torch.tensor(labels_list, dtype=torch.long),
        )
    return collate