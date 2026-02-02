# Repo Description
This repo is organized as follows:

```
├── aggregation_protocols               # FedAvg, FedProx, etc.
├── dataset_loaders                     # MNIST, CIFAR10, Fashion-MNIST, etc.
├── models                              # The models utilized for the respective datasets
├── utils                               # Some custom-class type definitions and metric savers
├── install_packages_AMD.sh             # Python Packages necessary to run simulatinos on AMD GPU
├── install_packages_NVIDIA.sh          # Python Packages necessary to run simulatinos on NVIDIA GPU
├── run_simulations.sh                  # Bash script to run the simulations. Parses all the required inputs
└── simulation_launcher.py              # FL Simulator code
```

# FL Simulation
The `simulation_launcher.py` contains the Python + MPI4PY code that's required to run the FL simulation. In this code, I implemented the classic Hub-and-Spoke architecture, in which all clients communicate only with a single central server. 

In my implementation, clients exchange their model gradients instead of the actual model weights. That is, at every FL round, clients train their models, send the computed gradient to the server, return to their initial state and wait for the aggregated gradient. 

This is gradient exchange and update workflow is one of the possible ways to conduct Federated Learning training and is in alignment with Federated Learning Contribution Estimation. Later, I will release the code in which I return different gradients for each client as a result of its contribution to the model.
