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
        bn_eps = 1e-3
        bn_momentum = 0.01

        # Neural network Layers: Defining the layers of the Neural Network
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(num_features=32, eps=bn_eps, momentum=bn_momentum)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(num_features=32, eps=bn_eps, momentum=bn_momentum)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop1 = nn.Dropout(p=0.3)
       
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(num_features=64, eps=bn_eps, momentum=bn_momentum)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn4   = nn.BatchNorm2d(num_features=64, eps=bn_eps, momentum=bn_momentum)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop2 = nn.Dropout(p=0.5)

        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)  # "same"
        self.bn5   = nn.BatchNorm2d(num_features=128, eps=bn_eps, momentum=bn_momentum)
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1) # "same"
        self.bn6   = nn.BatchNorm2d(num_features=128, eps=bn_eps, momentum=bn_momentum)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop3 = nn.Dropout(p=0.5)

        # After 3 pools: 32x32 -> 16x16 -> 8x8 -> 4x4
        self.fc1   = nn.Linear(in_features=128 * 4 * 4, out_features=128)
        self.bn7   = nn.BatchNorm1d(num_features=128, eps=bn_eps, momentum=bn_momentum)
        self.drop4 = nn.Dropout(p=0.5)
        self.fc2   = nn.Linear(in_features=128, out_features=10)  # logits

    # Forward Pass: The sequence of transformations that are executed during each training epoch
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.pool1(x)
        x = self.drop1(x)

        x = self.bn3(F.relu(self.conv3(x)))
        x = self.bn4(F.relu(self.conv4(x)))
        x = self.pool2(x)
        x = self.drop2(x)

        x = self.bn5(F.relu(self.conv5(x)))
        x = self.bn6(F.relu(self.conv6(x)))
        x = self.pool3(x)
        x = self.drop3(x)

        x = torch.flatten(x, 1)
        x = self.bn7(F.relu(self.fc1(x)))
        x = self.drop4(x)
        logits = self.fc2(x)
        return logits

    
# Defining the training procedure
def train(net: nn.Module, trainloader: DataLoader, epochs: int, device: torch.device | str, lr: float)-> tuple[float, float, list[npt.NDArray[np.generic]]]:

    net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9,weight_decay=5e-4)
    # optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-7,)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9934)

    # Copying the initial model
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
            outputs = net(images.to(device))
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            num_batches += 1
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels.to(device)).sum().item()
            total += labels.size(0)
        # scheduler.step()

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
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    avg_testloss = loss / len(testloader)
    return avg_testloss, accuracy

def parameter_update_fn(final_model: nn.Module, initial_model: nn.Module, device: torch.device | str):
    # Ensure both models are on same device (optional; state_dict tensors follow the model device)
    final_model.to(device)
    initial_model.to(device)

    final_sd = final_model.state_dict()
    init_sd  = initial_model.state_dict()

    # IMPORTANT: iterate keys in the same order as model_to_numpy uses (state_dict().items())
    update = []
    for k in final_sd.keys():
        du = (final_sd[k].detach() - init_sd[k].detach()).cpu().numpy()
        update.append(du)
    return update

def add_update_to_model(model, update):
    out = []
    for p, du in zip(model, update):
        # Ensure we always keep arrays (avoid numpy scalar outputs)
        p_arr  = np.asarray(p)
        du_arr = np.asarray(du)

        # Don't update integer buffers (e.g., BatchNorm num_batches_tracked)
        if np.issubdtype(p_arr.dtype, np.integer):
            out.append(p_arr)
        else:
            out.append(np.asarray(p_arr + du_arr))
    return out

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