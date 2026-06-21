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
    conv1: nn.Conv2d
    conv2: nn.Conv2d
    pool: nn.MaxPool2d
    fc1: nn.Linear
    fc2: nn.Linear
    fc3: nn.Linear

    def __init__(self):
        super(Net, self).__init__()
        # Neural network Layers: Defining the layers of the Neural Network
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=0)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=0)
        self.fc1 = nn.Linear(in_features=64 * 4 * 4, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=10)
        
    # Forward Pass: The sequence of transformations that are executed during each training epoch
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
def train(net: nn.Module,
          trainloader: DataLoader,
          epochs: int,
          device: torch.device | str,
          optimizer,
          scheduler=None,) -> tuple[float,
                                    float,
                                    list[npt.NDArray[np.generic]],
                                    list[npt.NDArray[np.generic]]]:
    
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1).to(device)

    initial_model = copy.deepcopy(net)

    net.train()
    correct = 0
    total = 0
    running_loss = 0.0
    num_batches = 0

    for _ in range(epochs):
        for batch in trainloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1

            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        if scheduler is not None:
            scheduler.step()

    
    # Copying the final local model
    final_params = model_to_numpy(net)

    net.load_state_dict(initial_model.state_dict())

    avg_trainloss: float = running_loss / num_batches
    training_accuracy: float = correct / total
    return avg_trainloss, training_accuracy, final_params

def test(net: nn.Module, testloader: DataLoader, device: torch.device | str) -> tuple[float, float]:

    net.to(device)
    net.eval()
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    
    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    avg_testloss = loss / len(testloader)
    return avg_testloss, accuracy

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