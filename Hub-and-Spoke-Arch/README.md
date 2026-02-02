## Repository Structure

This repository is organized as follows:

```text
├── aggregation_protocols               # Aggregation rules (e.g., FedAvg, FedProx)
├── dataset_loaders                     # Dataset loaders (e.g., MNIST, CIFAR-10, Fashion-MNIST)
├── models                              # Model definitions used per dataset
├── utils                               # Type definitions, logging, and metric saving utilities
├── install_packages_AMD.sh             # Dependencies for running on AMD GPUs
├── install_packages_NVIDIA.sh          # Dependencies for running on NVIDIA GPUs
├── run_simulations.sh                  # Experiment runner (parses CLI args and launches jobs)
└── simulation_launcher.py              # Core FL simulator (Python + mpi4py)
```

## Federated Learning Simulation
The main entry point is `simulation_launcher.py`, which implements a Hub-and-Spoke (centralized) federated learning architecture using **Python** + **mpi4py (MPI)**. In this setup, clients communicate only with a single central server.

## Gradient-Exchange Workflow

Instead of sending full model weights, clients exchange model updates as gradients:
- The server broadcasts the current global model state.
- Each client trains locally for one (or more) epochs.
- Each client computes its gradient/update and sends it to the server.
- The client resets back to the pre-training model state (so the server update is the only applied update).
- The server aggregates received gradients and updates the global model.
- The aggregated update is redistributed for the next round.

This design is particularly convenient for Federated Learning Contribution Estimation, where utilities (e.g., cosine similarity between a client update and the aggregated update) are computed directly from gradients/updates.