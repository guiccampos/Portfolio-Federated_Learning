from collections import OrderedDict
from torch import Tensor, nn
from torch.utils.data import DataLoader
import copy
import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.nn.functional as F

# Defined my model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # sentence -> tokens (strings) -> token IDs (ints) -> dense vectors (floats)
        input_channels = 1          # Number of input channels.
        output_channels = 128       # Design choice. Learns 128 different filters for each kernel size. Each filter detects a different kind of useful n-gram
        num_classes = 2             # SST2 has 2 classes: negative, positive
        embed_num = 20_000           # Design choice. Build a vocabulary from training data by keeping the 20,000 most frequent tokens, while everything else becomes <unk>)
        embed_dim = 300             # Each word token is represented by a 300-numbers vector. Think of it as a learned feature vector for each token, no human meaning individually
        kernel_size = [3,4,5]       # K =3 means each convolution filter looks at 3 consecutive words at once, across the full embedding dimension. The n-gram is a trigram.
        # kernel_size = [3,3,3]

        # Embedding is a learnable lookup table
        # input = (N, W), where N = batch size; W = number of tokens per sentence after padding/truncation
        self.embed = nn.Embedding(embed_num, embed_dim, padding_idx=0) # output = (N, W, D). Embedding adds a new dimension D=embed_dim
        self.convs1 = nn.ModuleList([nn.Conv2d(input_channels, output_channels, (K, embed_dim)) for K in kernel_size]) # For each kernel size, run a Conv2d. 


        # After pooling, it gives one vector per kernel size -> concatenated -> size == len(Ks)*Co
        # Then classifify into C classes
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(len(kernel_size) * output_channels, num_classes)
        
    # Forward Pass: The sequence of transformations that are executed during each training epoch
    def forward(self, x):
        # x = (N, W)
        x = self.embed(x) # output (N, W, D)
        x = x.unsqueeze(1) # (N, 1, W, D); pure shape manipulation to make sure that Conv2d receives the required tensor shape
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]    # output after convolution is (N, 128, W - K + 1, 1) Convolution spans K tokens and all embedding dimensions
                                                                    # .squeeze(3) removes the trailing 1, converting in a list like [(N, output_channels, W), ...] * len(kernel_size)

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]      # Each kernel size produces a vector of 128 feature per sentence.
                                                                    # [(N, output_channels), ...] * len(kernel_size)
        x = torch.cat(x, 1)
        x = self.dropout(x) # (N, len(kernel_size) * output_channels)
        logit = self.fc1(x) # (N, C)
        return logit
    
def train(net: nn.Module,
          trainloader: DataLoader,
          epochs: int,
          device: torch.device | str,
          optimizer,
          scheduler=None,) -> tuple[float,
                                    float,
                                    list[npt.NDArray[np.generic]]]:
    
    net.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    
    # Copying the initial model
    initial_model = copy.deepcopy(net)

    net.train()
    total=0
    correct = 0
    running_loss = 0.0
    total_batches = 0
    for _ in range(epochs):
        for batch in trainloader:
            # SST2 collate returns (input_ids, labels)
            input_ids, labels = batch
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = net(input_ids)
            loss = criterion(logits, labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            total_batches += 1
            predictions = logits.argmax(dim=1)            
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
        if scheduler is not None:
            scheduler.step()

    # Copying the final local model
    final_params = model_to_numpy(net)

    net.load_state_dict(initial_model.state_dict())
    
    avg_trainloss: float = running_loss / max(1, total_batches)
    training_accuracy: float = correct / total
    return avg_trainloss, training_accuracy, final_params

def test(net: nn.Module, testloader: DataLoader, device: torch.device | str) -> tuple[float, float]:

    net.to(device)
    net.eval()
    criterion = torch.nn.CrossEntropyLoss()

    total_samples = 0
    correct = 0
    running_loss = 0.0
    total_batches = 0
    
    with torch.no_grad():
        for batch in testloader:
            input_ids, labels = batch
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            logits = net(input_ids)
            loss = criterion(logits, labels)

            running_loss += loss.item()
            total_batches += 1

            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
    accuracy: float = correct / len(testloader.dataset)
    avg_loss: float = running_loss / total_batches  # or / len(testloader)
    return avg_loss, accuracy

def add_update_to_model(model: list[npt.NDArray[np.generic]], update: list[npt.NDArray[np.generic]]) -> list[npt.NDArray[np.generic]]:
    return [p + du for p, du in zip(model, update)]

def model_to_numpy(net: nn.Module) -> list[npt.NDArray[np.generic]]:
    # Returns a list of NumPy arrays, each corresponding to a layer's weights and biases
    # In the context of FL, this output represents the full "local model" at the client side.  
    return [v.detach().cpu().numpy().copy() for _, v in net.state_dict().items()]

def load_model(net: nn.Module, parameters: list[npt.NDArray[np.generic]]) -> None:
    # parameters: the list of NumPy arrays received from the Flower server
    # The parameters are received as a list of NumPy arrays, which is reconstructed into tensors and loaded into the model
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.from_numpy(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)