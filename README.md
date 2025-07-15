# SPERC: A Graph-Aware Reinforcement Learning Approach for Container Management

This repository provides the official Python implementation of the research paper: **"Solving the Hybrid Action Space Problem in Edge Computing: A Graph-Aware Reinforcement Learning Approach for Container Management"**.

The core of this work is the **Synergistic Policy network for Edge Resource Control (SPERC)**, a novel deep reinforcement learning framework designed to solve the joint problem of container placement and resource allocation in edge computing environments.

## Table of Contents
1.  [Framework Overview](#framework-overview)
2.  [Repository Structure](#repository-structure)
3.  [Requirements and Installation](#requirements-and-installation)
4.  [Usage](#usage)
    - [Configuration](#configuration)
    - [Training and Evaluation](#training-and-evaluation)
5.  [Key Implementation Details](#key-implementation-details)
    - [The Edge Environment](#the-edge-environment)
    - [The SPERC Agent](#the-sperc-agent)
    - [Baseline Algorithms](#baseline-algorithms)
6.  [Reproducing Paper Results](#reproducing-paper-results)

## Framework Overview

The deployment of containerized services at the network edge is a critical challenge, involving a trade-off between minimizing storage costs (via image layer sharing) and ensuring performance (via resource allocation). SPERC addresses this by:

*   **Formulating a Joint Optimization Problem:** We model the problem as a Mixed-Integer Nonlinear Program (MINLP) that maximizes net revenue by balancing latency and storage costs.
*   **Dual-Stream State Encoder:** SPERC uses a unique encoder to understand the environment:
    *   A **Graph Convolutional Network (GCN)** interprets the physical network topology and resource status.
    *   A **Factorization Machine (FM)** models the latent relationships between a service's required image layers and the layers already cached on edge nodes.
*   **Hierarchical Hybrid Control Policy:** To handle the hybrid (discrete-continuous) action space, SPERC uses a synergistic policy network:
    *   A **Dueling Deep Q-Network (DQN)** makes the high-level discrete decision of *where* to place a container.
    *   A **Deep Deterministic Policy Gradient (DDPG)** agent makes the low-level continuous decision of *how many* computational resources to allocate.
    *   These two policies are tightly coupled, with the placement decision being informed by the optimal resource allocation.


## Repository Structure

The codebase is organized into modular Python files for clarity and extensibility.

```
.
├── config.py             # Central configuration file for all simulation parameters.
├── environment.py        # Defines the Edge Computing environment simulation.
├── models.py             # Contains all PyTorch neural network architectures (GCN, FM, DDPG, DQN).
├── agents.py             # Implements the logic for all agents (SPERC, Baselines).
├── main.py               # Main execution script to run training and evaluation.
└── README.md             # This file.
```

## Requirements and Installation

This project is implemented in Python 3.8+ and relies on several key libraries.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ahmadpanah/sperc.git
    cd sperc
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install core dependencies:**
    ```bash
    pip install torch torchvision torchaudio
    pip install networkx pandas tqdm
    ```

4.  **Install PyTorch Geometric:**
    Installation of PyTorch Geometric depends on your PyTorch version and CUDA setup. Please follow the [official PyTorch Geometric installation instructions](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

    For example, for PyTorch 2.1 and CPU:
    ```bash
    pip install torch_geometric
    pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
    ```

## Usage

### Configuration

All simulation parameters are centralized in `config.py`. You can modify this file to change the experimental setup, such as the number of nodes, service characteristics, or RL hyperparameters. Key parameters from Table 3 in the paper are clearly marked.

```python
# config.py

# Network Parameters
N_NODES = 20
CPU_CAPACITY_RANGE = [2.0, 8.0]  # GHz

# Service Parameters
N_SERVICES = 100
N_UNIQUE_LAYERS = 500

# RL and Model Parameters
LEARNING_RATE = 1e-4
DISCOUNT_FACTOR = 0.99
STRATEGIC_FACTOR_LAMBDA = 0.5 # 0.0=Storage Focus, 1.0=Latency Focus

# Evaluation Parameters
N_TRAINING_EPISODES = 200
N_EVALUATION_RUNS = 20
```

### Training and Evaluation

The main script `main.py` handles the entire workflow. It will:
1.  Instantiate the edge environment.
2.  Instantiate all agents (SPERC and baselines).
3.  Train the reinforcement learning agents (SPERC, SPERC-Base, D-DRL).
4.  Evaluate all agents over multiple independent runs.
5.  Print a summary table of the results, similar to Table 4 in the paper.

To run the full simulation, simply execute:
```bash
python main.py
```

The output will look like this:

```
--- Training SPERCAgent ---
100%|██████████| 200/200 [05:10<00:00, 1.55s/it]
...

--- Starting Evaluation ---
Evaluating SPERC...
100%|██████████| 20/20 [00:15<00:00, 1.25it/s]
...

--- Overall Performance Comparison ---
Algorithm      Net Revenue  Avg. Latency (s)  Avg. Storage Cost (MB)  Acceptance Ratio (%)
SPERC                124.5              1.21                   115.3                  97.2
SPERC-Base           101.2              1.35                   140.8                  94.5
D-DRL                 95.8              1.41                   155.1                  93.1
Greedy-CPU            75.3              1.05                   285.4                  90.5
Greedy-LS             68.9              2.55                    89.2                  88.3
Random                15.7              2.81                   350.6                  75.4
```

## Key Implementation Details

### The Edge Environment (`environment.py`)

The `EdgeEnv` class simulates the physical world. It handles:
*   Generating a network topology using the **Waxman model**.
*   Assigning CPU, storage, and bandwidth capacities to nodes and links.
*   Generating containerized service requests with specific layer dependencies, workloads, and QoS constraints.
*   Calculating rewards based on the net revenue objective function.
*   Updating the state of the network (e.g., available CPU) after an action is taken.

### The SPERC Agent (`agents.py` and `models.py`)

The `SPERCAgent` is the core of our contribution. Its implementation features:
*   **`_get_state_embedding`**: This method processes a raw state from the environment and passes it through the GCN and FM encoders to produce the rich, disentangled state representation.
*   **`select_action`**: Implements the hierarchical decision process. It queries the DDPG actor for resource allocations for all possible nodes and feeds these into the Dueling DQN to select the best placement node.
*   **`update`**: A full implementation of the synergistic training loop where the DQN (critic) and DDPG (actor) networks are co-optimized.

### Baseline Algorithms (`agents.py`)

To demonstrate SPERC's effectiveness, we compare it against several baselines:
*   **SPERC-Base**: An ablated version of SPERC that uses a simple flattened state vector instead of the dual-stream encoder.
*   **D-DRL**: A standard DDPG agent that handles the hybrid action space naively.
*   **Greedy-CPU**: A heuristic that always places the service on the node with the most available CPU.
*   **Greedy-LS**: A heuristic that prioritizes layer sharing by placing the service on the node that requires the fewest new layer downloads.
*   **Random**: A random policy that serves as a lower-bound performance benchmark.

## Reproducing Paper Results

To reproduce the results from the paper's tables, you will need to modify `config.py` and re-run `main.py`.

*   **Table 4 (Overall Performance):** Use the default `config.py` settings.
*   **Table 5 (Impact of CPU):** Change `CPU_CAPACITY_RANGE` to `[1.0, 3.0]`, `[4.0, 6.0]`, and `[7.0, 9.0]` to simulate 2, 5, and 8 GHz average capacity, respectively.
*   **Table 6 (Impact of λ):** Change `STRATEGIC_FACTOR_LAMBDA` to `0.1`, `0.5`, and `0.9`.
*   **Table 7 (Scalability):** Change `N_SERVICES` to `50`, `150`, and `250`.