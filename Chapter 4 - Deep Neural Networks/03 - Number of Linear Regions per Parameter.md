### Setup

Let:
- scalar input $x \in \mathbb{R}$
- scalar output $y \in \mathbb{R}$
- ReLU activations

---

## 1. Shallow Network

### Architecture
$$
f(x) = \sum_{i=1}^{D} a_i \max(0, w_i x + b_i) + c
$$

Each ReLU introduces a breakpoint at:
$$
x = -\frac{b_i}{w_i}
$$

With $D$ hidden units:
- at most $D$ distinct breakpoints on the real line
- these breakpoints partition $\mathbb{R}$ into $D+1$ intervals

Each hidden unit has a weight, bias, and output weight. Furthermore, the output has it's own bias, totaling $3D+1$ parameters

---

## 2. Deep network: K layers, width D > 2

### Architecture
- K hidden layers
- each layer has D ReLU units
- scalar input/output

---

### Composition argument

Let $R_k$ be the maximum number of linear regions after k layers.

Base case:
$$
R_1 = D+1
$$

Recurrence:
$$
R_{K} \le (D+1) R_{K-1} \le \dots \le (D+1)^{K}
$$

---

## 3. Parameter count (deep network)

- As discussed, the first hidden layer has $3D+1$ parameters
- Each subsequent layer maps $\mathbb{R}^D \to \mathbb{R}^D$, totaling to $D(D+1)$ parameters
- Overall, we have $3D + 1 + (K-1)D(D+1)$

---

## 4. Expressivity per parameter

Shallow:
$$
\text{regions} = O(D)
\quad \text{parameters} = O(D)
$$

Deep:
$$
\text{regions} = O((D+1)^K)
\quad \text{parameters} = O(KD^2)
$$

Thus:
- shallow networks grow linearly
- deep networks grow exponentially in depth

