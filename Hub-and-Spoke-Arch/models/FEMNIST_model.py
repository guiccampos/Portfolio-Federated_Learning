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

    def __init__(self):
        super(Net, self).__init__()
        # Neural network Layers: Defining the layers of the Neural Network
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.pool  = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1   = nn.Linear(in_features=7 * 7 * 64, out_features=2048)
        self.fc2   = nn.Linear(in_features=2048, out_features=62)
                
    # Forward Pass: The sequence of transformations that are executed during each training epoch
    def forward(self, x: Tensor) -> Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (N, 1, 28, 28)

        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1) # (N, 7*7*64)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
    
# Defining the training procedure
def train(net: nn.Module, trainloader: DataLoader, epochs: int, device: torch.device | str, lr: float)-> tuple[float, float, list[npt.NDArray[np.generic]]]:

    net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)                                          # Moves the loss function to the device. 
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)                                        # MNIST I used 0.1
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.977)
    
    # Copying the initial model
    initial_model = copy.deepcopy(net)

    net.train()
    correct = 0 
    total = 0
    running_loss = 0.0
    num_batches = 0
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            num_batches += 1
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels.to(device)).sum().item()
            total += labels.size(0)
        scheduler.step()

    # Copying the final local model
    final_model = net

    # Generate the "Model Parameter Update Vector" a.k.a. Gradients
    parameter_update = parameter_update_fn(final_model, initial_model, device)

    net.load_state_dict(initial_model.state_dict())

    avg_trainloss: float = running_loss / num_batches
    training_accuracy: float = correct / total
    return avg_trainloss, training_accuracy, parameter_update

def test(net: nn.Module, testloader: DataLoader, device: torch.device | str) -> tuple[float, float]:

    net.to(device)
    net.eval()
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    
    with torch.no_grad():
        for batch in testloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (outputs.argmax(dim=1) == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    avg_testloss = loss / max(1, len(testloader))
    return avg_testloss, accuracy

def parameter_update_fn(final_model: nn.Module, initial_model: nn.Module, device: torch.device | str) -> list[npt.NDArray[np.generic]]:
    final_model, initial_model = final_model.to(device), initial_model.to(device)
    model_update_tensors: list[Tensor] = [final_param.detach() - initial_param.detach() for final_param, initial_param in zip(final_model.parameters(), initial_model.parameters())]
    model_update: list[npt.NDArray[np.generic]] = [tensor.detach().cpu().numpy() for tensor in model_update_tensors]
    return model_update

def add_update_to_model(model: list[npt.NDArray[np.generic]], update: list[npt.NDArray[np.generic]]) -> list[npt.NDArray[np.generic]]:
    return [p + du for p, du in zip(model, update)]

def local_model(net: nn.Module) -> list[npt.NDArray[np.generic]]:
    # Returns a list of NumPy arrays, each corresponding to a layer's weights and biases
    # In the context of FL, this output represents the full "local model" at the client side.  
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def collect_global_model(net: nn.Module, parameters: list[npt.NDArray[np.generic]]) -> None:
    # parameters: the list of NumPy arrays received from the Flower server
    # The parameters are received as a list of NumPy arrays, which is reconstructed into tensors and loaded into the model
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.from_numpy(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)