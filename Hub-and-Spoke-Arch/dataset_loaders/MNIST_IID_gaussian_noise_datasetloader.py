from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
import torch

fds = None

# Defined how the data is going to be split across users
def load_noisy_data(
    partition_id: int,
    num_partitions: int,
    seed: int = 12345,
    batch_size: int = 128,
    num_corrupted_clients: int = 0,
    noise_std: float = 0.0,
    noise_prob: float = 1.0,
):
    """Load partition MNIST data."""
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        # Performs IID partitioning. Other partitioning formulas can be inserted here
        train_partitioner = IidPartitioner(num_partitions=num_partitions)
        test_partitioner = IidPartitioner(num_partitions=num_partitions)

        # This Class downloads and partition data among client
        # I'm doing an IID partitioning based on the number of partitions and then each partition is associated with a partitioner
        fds = FederatedDataset(                                                         # This Class downalods and partitiong data among client
            dataset="ylecun/mnist",                                                     # Specify the name of the dataset in the hugging face hub
            partitioners={"train": train_partitioner, "test": test_partitioner},        # Python Dict mapping the Dataset Split to a Partitioner
            shuffle=True,
            seed=seed
        )
                                                                                        
    partition_train = fds.load_partition(partition_id, split="train")
    partition_test = fds.load_partition(partition_id, split="test")

    # Clamp noise_prob to [0, 1] for safety
    if noise_prob < 0.0:
        noise_prob = 0.0
    if noise_prob > 1.0:
        noise_prob = 1.0

    to_tensor = ToTensor()
    normalize = Normalize((0.1307,), (0.3081,))

    # Per-client RNG stream so noise application is reproducible
    client_gen = torch.Generator()
    client_gen.manual_seed(seed + 10_000 * int(partition_id))


    def apply_transforms_train(batch):
        imgs = []
        for img in batch["image"]:
            x = to_tensor(img)  # [0, 1]

            if noise_std > 0.0 and noise_prob > 0.0:
                apply_noise = True
                if noise_prob < 1.0:
                    apply_noise = bool(torch.rand((), generator=client_gen).item() < noise_prob)

                if apply_noise:
                    eps = torch.randn(
                        x.shape,
                        generator=client_gen,
                        dtype=x.dtype,
                        device=x.device
                    ) * float(noise_std)
                    x = torch.clamp(x + eps, 0.0, 1.0)

            x = normalize(x)
            imgs.append(x)

        batch["image"] = imgs
        return batch

    def apply_transforms_test(batch):
        """Apply transforms to the TEST partition from FederatedDataset (kept clean)."""
        batch["image"] = [normalize(to_tensor(img)) for img in batch["image"]]
        return batch

    partition_train = partition_train.with_transform(apply_transforms_train)
    partition_test = partition_test.with_transform(apply_transforms_test)
    
    trainloader = DataLoader(partition_train, batch_size=batch_size, shuffle=True)
    testloader  = DataLoader(partition_test,  batch_size=batch_size, shuffle=False)
    return trainloader, testloader