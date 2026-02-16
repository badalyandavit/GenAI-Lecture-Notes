
### Setup
Assume a one step of a feedforward network between adjacent layers:
$$
h = a[f], \qquad f' = \beta + \Omega h
$$

- $f \in \mathbb{R}^{D_h}$, $h \in \mathbb{R}^{D_h}$, $f' \in \mathbb{R}^{D_{h'}}$
- $\Omega \in \mathbb{R}^{D_{h'} \times D_h}$, $\beta \in \mathbb{R}^{D_{h'}}$
- $a[\cdot]$ is ReLU unless stated otherwise


**Initialization model:**
- $\beta_i$ often initialized to $0$
- $\Omega_{ij}$ i.i.d. Gaussian Distribution with mean $0$ and variance $\sigma_\Omega^2$

**Note**:
- Setting means to $0$ introduces non preference settings to the model.
- If we have "expert knowledge" about the task, or aim to fine-tune a particular model, then we may alter the expectations of the weights.
- Same goes for the variance, lower (higher) variance implies less (more) diversity in model outputs.

---

### Expectation of the next pre-activation

For neuron \(i\) in the next layer:

$$
f'_i = \beta_i + \sum_{j=1}^{D_h} \Omega_{ij} h_j
$$

Using $\beta_i = 0$, $\mathbb{E}[\Omega_{ij}] = 0$:

$$
\mathbb{E}[f'_i]
= \mathbb{E}[\beta_i] + \sum_{j=1}^{D_h} \mathbb{E}[\Omega_{ij} h_j]
= 0 + \sum_{j=1}^{D_h} \mathbb{E}[\Omega_{ij}] \, \mathbb{E}[h_j]
= 0
$$

### Variance of the next pre-activation

Using $\mathbb{E}[f'_i]=0$:

$$
\mathrm{Var}(f'_i) = \mathbb{E}[(f'_i)^2] - \mathbb{E}[(f'_i)]^2
= \mathbb{E}\Big[\Big(\sum_{j=1}^{D_h} \Omega_{ij}h_j\Big)^2\Big] - 0 = \mathbb{E}\Big[\Big(\sum_{j=1}^{D_h} \Omega_{ij}h_j\Big)^2\Big]
$$
Expand the square and use independence of $h_i$ with $h_j$:

$$
\Big(\sum_j \Omega_{ij}h_j\Big)^2
= \sum_j \Omega_{ij}^2 h_j^2 + \sum_{j\neq k}\Omega_{ij}h_j \Omega_{ik}h_k = \sum_j \Omega_{ij}^2 h_j^2 + 0 = \sum_j \Omega_{ij}^2 h_j^2
$$

$$
\mathrm{Var}(f'_i)
= \sum_{j=1}^{D_h}\mathbb{E}[\Omega_{ij}^2]\mathbb{E}[h_j^2]
$$
Using $\mathrm{Var}(\Omega^2) = \sigma_\Omega^2 = \mathbb{E}[\Omega_{ij}^2] - \mathbb{E}[\Omega_{ij}]^2 = \mathbb{E}[\Omega_{ij}^2] - 0 = \mathbb{E}[\Omega_{ij}^2]$

$$\mathrm{Var}(f'_i) = \sigma_{f^{'}}^2 = \sigma_\Omega^2 \sum_{j=1}^{D_h}\mathbb{E}[h_j^2] = \frac{D_h}{2} \sigma_\Omega^2 \sigma_f^2$$
Setting $\frac{\sigma_{f^{'}}^2}{\sigma_f^2} = 1$ yields $\sigma_\Omega^2 = \frac{2}{D_{h}}$
This is also known as the He Initialization.
However, one big flaw is the assumption that the dimension size between layers is equal.
The Xavier initialization (algebraic mean) is used when the dimension sizes are not equal. In particular, the He activation is a subtype of the Xavier activation, as setting $D_h = D_{h^{'}}$ yields to the desired He activation.
$$
\sigma_\Omega^2 = \frac{4}{D_h + D_{h'}}
$$

---

### What if the expected value is not 0?

### General mean propagation 
$$
\mathbb{E}[f'_i]
= \mathbb{E}[\beta_i] + \sum_{j=1}^{D_h}\mathbb{E}[\Omega_{ij}h_j]
\approx \mu_\beta + \sum_{j=1}^{D_h}\mathbb{E}[\Omega_{ij}]\,\mathbb{E}[h_j]
= \mu_\beta + D_h \mu_\Omega \mu_h
$$

## 4) Connection to transfer learning: non-zero mean as “prior”, but be precise about what is non-zero

In transfer learning, you do not sample $\Omega$ from $\mathcal{N}(0,\sigma^2)$. You start from pretrained weights $\Omega_{(pre)}$.

A standard fine-tuning objective with weight decay is:

$$
\min_\Omega \; \mathcal{L}(\Omega) + \lambda \|\Omega\|^2
$$

This corresponds to a Gaussian prior centered at 0.

For transfer learning, it is often more faithful to think in terms of:

$$
\min_\Omega \; \mathcal{L}(\Omega) + \lambda \|\Omega - \Omega_{\text{pre}}\|^2
$$

This is equivalent to a Gaussian prior centered at $\Omega_{\text{pre}}$.
At this stage, fine-tuning becomes "data + prior" learning. This makes the “expected value not being zero” meaningful as “prior knowledge”, but the expectation is over your *posterior belief about good weights*, not over random sampling at init.