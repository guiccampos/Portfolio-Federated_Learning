## Federated Learning Architectures Portfolio

Welcome! This repository is part of my Federated Learning research portfolio and highlights my work with **Python**, **PyTorch**, and **mpi4py (MPI)** for distributed training and simulation.

While several Federated Learning frameworks provide strong support for the **hub-and-spoke** (central server) setting, my Master’s thesis required evaluating **alternative communication topologies** that are not commonly available out-of-the-box—such as **peer-to-peer** and **hierarchical (M-ary tree)** architectures.

Rather than heavily modifying an existing framework (often a large engineering effort with tight coupling to a specific execution model), I implemented a **custom simulation codebase** focused on:
- rapid prototyping of FL architectures,
- reproducible experiments,
- transparent control of aggregation, communication, and client selection logic.

Some modules are **temporarily unavailable for public release** due to research constraints. This repository will be updated as soon as additional components can be shared.

### Tech Stack
- Python
- PyTorch
- mpi4py / MPI

### Architectures Available
- Hub-and-Spoke 