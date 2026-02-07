### Setup

Let:
- $D_i$ = input dimension
- $K$ = number of hidden layers
- $D$ = number of ReLU units per layer
- Assume $D$ is an integer multiple of $D_i$
- All weights are in general position

We consider a deep ReLU network with:
- input $x \in \mathbb{R}^{D_i}$
- $K$ hidden layers, each of width $D$

As shown for shallow networks, if a network has a total of $D_{\text{tot}}$ ReLU units, then:
$$
N_r \le 2^{D_{\text{tot}}}
$$
---

## Step 2: Region growth via composition (folding argument)

A ReLU layer maps $\mathbb{R}^{D_i}$ into a union of polyhedral regions.
Composing layers multiplies region counts.

Key observation:
- A layer with $D$ ReLUs can fold each input dimension independently
- Each input dimension can be folded at most $(D/D_i + 1)$ times per layer

Thus, for one layer:
$$
\text{folds per dimension} = \frac{D}{D_i} + 1
$$

---

## Step 3: Contribution of the first $K-1$ layers

The first $K-1$ layers repeatedly fold the input space.

Each layer multiplies the number of regions by:
$$
\left(\frac{D}{D_i} + 1\right)^{D_i}
$$

After $K-1$ layers:
$$
\left(\frac{D}{D_i} + 1\right)^{D_i(K-1)}
$$

This accounts for repeated, independent folding along each input dimension.

---

## Step 4: Contribution of the final layer

The final layer acts like a shallow ReLU network with:
- input dimension $D_i$
- width $D$

From hyperplane arrangement theory:
$$
\text{max regions} = \sum_{j=0}^{D_i} \binom{D}{j}
$$

This term corresponds exactly to the last layer.

---

## Step 5: Combine the contributions

Multiplying the effects of:
- repeated folding from the first $K-1$ layers
- final partitioning by the last layer

We obtain:
$$
\boxed{
N_r
=
\left(\frac{D}{D_i} + 1\right)^{D_i(K-1)}
\sum_{j=0}^{D_i} \binom{D}{j}
}
$$
