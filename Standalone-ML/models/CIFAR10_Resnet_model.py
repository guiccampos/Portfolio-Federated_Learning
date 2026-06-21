from collections import OrderedDict
from torch import Tensor, nn
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152, ResNet
from torch.utils.data import DataLoader
import copy
import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.nn.functional as F

# Preparing Resnet18 to CIFAR100
def PrepareResnet(version: str = "resnet18") -> ResNet:
    builders = {
        "resnet18": resnet18,
        "resnet34": resnet34,
        "resnet50": resnet50,
        "resnet101": resnet101,
        "resnet152": resnet152,
    }

    if version not in builders:
        raise ValueError(f"Unsupported ShuffleNetV2 version: {version}")

    model = builders[version](weights=None)

    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model


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

    final_params = model_to_numpy(net)
    _, parameter_update = parameter_update_fn(
        initial_model=initial_model,
        final_model=net,
        include_bn_affine=False,
    )

    net.load_state_dict(initial_model.state_dict())

    avg_trainloss = running_loss / num_batches
    training_accuracy = correct / total

    return avg_trainloss, training_accuracy, final_params, parameter_update

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

def parameter_update_fn(
    initial_model: nn.Module,
    final_model: nn.Module,
    include_bn_affine: bool = False,
) -> tuple[list[str], list[npt.NDArray[np.generic]]]:
    """
    Return a FL-portable model update in full state_dict order.

    The returned update has the SAME length and SAME ordering as model_to_numpy(net),
    so it can be safely:
      - aggregated on the server,
      - added to a full model with add_update_to_model(),
      - loaded back with load_model().

    BatchNorm handling:
      - BN buffers (running_mean, running_var, num_batches_tracked) are always zeroed
      - BN affine params (weight, bias) are zeroed unless include_bn_affine=True

    Non-floating tensors are also zeroed to preserve compatibility with state_dict layout.
    """
    initial_sd = initial_model.state_dict()
    final_sd = final_model.state_dict()
    modules = dict(final_model.named_modules())

    state_keys = list(final_sd.keys())
    updates: list[npt.NDArray[np.generic]] = []

    for key in state_keys:
        final_tensor = final_sd[key].detach().cpu()
        initial_tensor = initial_sd[key].detach().cpu()

        # Keep update list aligned with the full state_dict layout
        # Non-floating tensors (e.g., num_batches_tracked) must not be updated
        if not torch.is_floating_point(final_tensor):
            updates.append(torch.zeros_like(final_tensor).numpy().copy())
            continue

        # Recover module name + field name from a key like:
        # "layer1.0.bn1.weight" -> module_name="layer1.0.bn1", field_name="weight"
        if "." in key:
            module_name, field_name = key.rsplit(".", 1)
            module = modules.get(module_name, None)
        else:
            module_name = ""
            field_name = key
            module = None

        is_bn_module = isinstance(module, nn.modules.batchnorm._BatchNorm)

        if is_bn_module:
            # Keep BN running stats local
            if field_name in {"running_mean", "running_var", "num_batches_tracked"}:
                updates.append(torch.zeros_like(final_tensor).numpy().copy())
                continue

            # Optionally exclude BN affine params from FL update / cosine / reward
            if field_name in {"weight", "bias"} and not include_bn_affine:
                updates.append(torch.zeros_like(final_tensor).numpy().copy())
                continue

        delta = final_tensor - initial_tensor
        updates.append(delta.numpy().copy())

    return state_keys, updates