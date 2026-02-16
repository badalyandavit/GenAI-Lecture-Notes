### Managing Memory

Training modern neural networks is often constrained by memory rather than raw compute. Two resources must be managed: (i) **device memory** for parameters, activations, and optimizer state during forward/backward passes, and (ii) **disk memory** for checkpoints, datasets, and intermediate artifacts.

### Reducing Memory Requirements

Training is memory intensive because we must store model parameters and, during the forward pass, the pre-activations (or activations) for every element in the batch. These cached values are required for backpropagation. Memory can be reduced either by recomputing activations when needed or by changing how batches are processed.

#### Gradient checkpointing

Gradient checkpointing (Chen et al., 2016a) stores activations only at selected layers during the forward pass. If checkpoints are placed every $N$ layers, then intermediate activations are discarded and recomputed during backpropagation from the nearest checkpoint. This reduces activation memory at the cost of extra forward computation.

#### Micro-batching and gradient accumulation

Micro-batching (Huang et al., 2019) splits a batch into smaller sub-batches processed sequentially. Gradients are accumulated across sub-batches and applied once, preserving the effective batch size while lowering peak activation memory. The trade-off is more forward/backward passes per update and reduced parallel efficiency.

#### Reversible networks

Reversible networks (Gomez et al., 2017) enable reconstruction of previous-layer activations from current-layer activations. This removes the need to cache activations during the forward pass, reducing memory use through architectural constraints (see chapter 16).

### Distributed Training

For large models, memory requirements or training time can exceed the capacity of a single processor. Distributed training executes training across multiple nodes. The main difference across approaches is what is replicated and what is partitioned.

#### Data parallelism

In data parallelism, each node holds a full copy of the model and processes a subset of the batch (Xing et al., 2015; Li et al., 2020b). Gradients must be aggregated and redistributed to keep replicas consistent.

In **synchronous** data parallelism, aggregation happens at each step (or at a fixed cadence). Communication overhead can become a bottleneck as the number of nodes grows.

In **asynchronous** training, workers apply updates without strict step synchronization. For example, Hogwild! (Recht et al., 2011) applies each worker’s gradient update to a shared model as soon as it is available, which reduces waiting time but can introduce stale gradients.

#### Decentralized schemes

Decentralized schemes remove the central aggregator and use fixed communication topologies. For example, Zhang et al. (2016a) describe ring-style updates where nodes exchange information with neighbors to propagate updates through the system.

#### Communication primitives

Distributed training relies on a small set of communication primitives. These differ mainly in whether aggregation is collective or server-mediated, and whether coordination is centralized or decentralized.

##### Collective operations

A common pattern in synchronous data-parallel training is to average gradients using **all-reduce**. If worker $i$ produces gradient $g_i$, the synchronized gradient is

$$  
\bar{g}=\frac{1}{K}\sum_{i=1}^{K} g_i,  
$$

where $K$ is the number of workers. Systems often implement all-reduce using bandwidth-efficient schemes (for example, ring-style implementations), and may overlap communication with backpropagation.

##### Parameter server pattern

In a parameter-server design, workers send gradients to one or more servers, and receive updated parameters. Synchronous variants aggregate per step; asynchronous variants apply updates as they arrive. This reduces barrier synchronization but increases staleness risk.

##### Gossip-style (neighbor averaging)

In decentralized designs, nodes communicate only with neighbors in a graph. A standard abstraction is mixing local parameters with neighbor parameters via a matrix $W$:

$$  
\theta_i^{(t+1)}=\sum_{j\in \mathcal{N}(i)\cup{i}} W_{ij},\theta_j^{(t)},  
$$

where $\mathcal{N}(i)$ are neighbors of node $i$. Convergence depends on the graph topology and the mixing weights.

##### Federated learning (NVIDIA FLARE)

The setting differs from multi-GPU training: clients keep data local and exchange model updates periodically, typically with server coordination, rather than using high-throughput collectives each step.