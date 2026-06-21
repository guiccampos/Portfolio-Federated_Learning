from __future__ import annotations

from typing import TypedDict, TypeAlias, Literal, NotRequired
import numpy as np
import numpy.typing as npt

# A single layer tensor represented as a NumPy array (dtype can vary)

class ClientTrainResult(TypedDict):
    client_id: int
    model_update: list[npt.NDArray[np.generic]]
    local_training_dataset_size: int
    flattened_normalized_gradient: NotRequired[npt.NDArray[np.float64]]

class ClientTestResult(TypedDict):
    client_id: int
    local_training_dataset_size: int
    local_dataset_train_loss: float
    local_dataset_train_accuracy: float
    local_dataset_test_loss: float
    local_dataset_test_accuracy: float
    local_testing_dataset_size: int
    central_dataset_test_loss: float
    central_dataset_test_accuracy: float
    corrupted_labels: bool
    corruption_fraction: float

class TrainingInstructions(TypedDict):
    phase: Literal["training"]
    round: int
    local_epochs: int
    has_model: bool
    model: NotRequired[list[npt.NDArray[np.generic]]]
    importance_coefficients: dict[int, float]

class EvaluationInstructions(TypedDict):
    phase: Literal["evaluation"]
    round: int

class StopInstructions(TypedDict):
    phase: Literal["stop"]