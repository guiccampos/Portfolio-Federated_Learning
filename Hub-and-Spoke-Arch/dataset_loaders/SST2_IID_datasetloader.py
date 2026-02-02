from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader
from collections import Counter
from typing import List
import torch
import re
from datasets import Dataset

fds = None
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"

# Defined how the data is going to be split across users
def load_data(partition_id: int,
              num_partitions: int,
              global_token_to_tid: dict[str, int],
              max_length: int,
              seed: int = 12345,
              batch_size: int = 128):
    """Load partition Stanford Sentiment Tree 2 data."""
    # For this specific dataset, I will not use the default validation nor test partition. Validation is too small for FL setting and test has no labels.
    # Instead, I will split the train set into train and test (80%-20%)
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
    partition_train = split_partition["train"]
    partition_test  = split_partition["test"]

    collate_fn = collect_and_combine(global_token_to_tid=global_token_to_tid, max_length=max_length)

    trainloader = DataLoader(partition_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    testloader  = DataLoader(partition_test,  batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return trainloader, testloader

# Sample trainloader and test loader
# {'idx': 36804, 'sentence': 'skillfully assembled , highly polished and professional adaptation ', 'label': 1}

def tokenize(sentence: str) -> List[str]:
    sentence = re.findall(r"[a-z]+(?:'[a-z]+)?", sentence.lower())
    return sentence

def build_global_token_tid_map(seed:int,
                               embed_num: int) -> dict[str, int]:
    """
    Build one global token-> token-id mapping from the full train split
    """

    train_partitioner = IidPartitioner(num_partitions=1)
    global_fds = FederatedDataset(                                                         # This Class downloads and partition data among client
            dataset="stanfordnlp/sst2",                                                 # Specify the name of the dataset in the hugging face hub
            partitioners={"train": train_partitioner,
                        #   "validation": validation_partitioner, 
                        #   "test": test_partitioner
                          },
            shuffle=True,
            seed=seed
        )
    
    partition = global_fds.load_partition(0, split="train")

    # Token to Token ID mapping
    global_token_to_tid = {
        PAD_TOKEN: 0,
        UNK_TOKEN: 1
    }

    counter = Counter()
    for sample in partition:
        sentence = sample['sentence']       # Extract sentence from the train sample
        tokens = tokenize(sentence)         # Tokenize the sentence
        counter.update(tokens)              # Update token frequency counter

    max_size = embed_num - 2

    id = 2 # Start indexing from 2, because 0 is for padding and 1 is for unknown words
    for token, token_frequency in counter.most_common(max_size):
        global_token_to_tid[token] = id
        id += 1

    return global_token_to_tid

# Return a list [token_id0, token_id1, ..., token_idN]
def sentence_to_ids(sentence: str,
                    global_token_to_tid: dict,
                    max_length: int) -> List:
    
    tokens = tokenize(sentence)

    token_ids: List[int] = []
    pad_id = global_token_to_tid[PAD_TOKEN]
    unk_id = global_token_to_tid[UNK_TOKEN]

    token_idx = 0

    # truncate
    while (token_idx < len(tokens)) and (token_idx < max_length):
        token = tokens[token_idx]
        token_id = global_token_to_tid.get(token, unk_id)
        token_ids.append(token_id)
        token_idx += 1

    # pad
    while len(token_ids) < max_length:
        token_ids.append(pad_id)

    return token_ids

def collect_and_combine (global_token_to_tid: dict,
                         max_length: int):
    def collate(batch):
        input_ids_list = []
        labels_list = []

        for sample in batch:
            sentence = sample["sentence"]
            label = sample["label"]
            token_ids = sentence_to_ids(sentence=sentence, global_token_to_tid=global_token_to_tid, max_length=max_length)
            input_ids_list.append(token_ids)
            labels_list.append(int(label))

        input_ids = torch.tensor(input_ids_list, dtype=torch.long)
        labels = torch.tensor(labels_list, dtype=torch.long)
        return input_ids, labels
    return collate
