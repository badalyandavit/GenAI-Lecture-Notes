### Setup
Let:
- $D_i$ = input dimension
- $D$ = number of hidden ReLU units

Each ReLU unit introduces a hyperplane:
$$
w_k^\top x + b_k = 0
$$

These $D$ hyperplanes partition $\mathbb{R}^{D_i}$ into regions.
On each region, the network is an affine (linear + bias) function.

---

### Key theorem (hyperplane arrangements)

The maximum number of regions formed by $D$ hyperplanes in $\mathbb{R}^{D_i}$ is:
$$
R(D,D_i) = \sum_{j=0}^{\min(D_i,D)} \binom{D}{j}
$$
---

### Proof (recurrence argument)

Define $R(D,D_i)$ as the maximum number of regions.

Recurrence:
$$
R(D,D_i) = R(D-1,D_i) + R(D-1,D_i-1)
$$

Reason:
- Add the $D$-th hyperplane.
- It splits exactly the regions it intersects.
- The number of intersections equals the number of regions formed by
  the previous $D-1$ hyperplanes restricted to the new hyperplane,
  which lies in $\mathbb{R}^{D_i-1}$.

Base cases:
$$
R(0,D_i)=1,\quad R(D,0)=1
$$

Solving the recurrence yields:
$$
R(D,D_i)=\sum_{j=0}^{D_i}\binom{D}{j}
$$
In the presence of parallel hyperplanes, too many hyperplanes intersecting at one point, or redundant hyperplanes, the upper bound drops.

---

### Connection to ReLU networks

Each ReLU has two regimes:
$$
\text{active: } w^\top x+b>0,\qquad
\text{inactive: } w^\top x+b\le 0
$$

Each activation pattern corresponds to an intersection of half-spaces.
Each nonempty intersection defines one linear region.

---

### Case 1: $D_i \le D$

$$
R(D,D_i)=\sum_{j=0}^{D_i}\binom{D}{j}
$$

Asymptotically for fixed $D_i$:
$$
R(D,D_i)\sim \frac{D^{D_i}}{D_i!}
$$

This grows polynomially in $D$ of degree $D_i$.

Rule of thumb:
$$
2^{D_i} \;\le\; R(D,D_i) \;\le\; 2^{D} \;\le\; O(D^{D_i})
$$

---

### Case 2: $D_i > D$

$$
R(D,D_i)=\sum_{j=0}^{D}\binom{D}{j}=2^D
$$

Reason:
- You cannot intersect more than $D$ independent hyperplanes.
- All binomial terms with $j>D$ vanish.
