## 1. Depth efficiency

### Informal definition

Depth efficiency means:

> There exist functions that can be represented by deep networks with a reasonable number of parameters, but any shallow network requires exponentially many hidden units to represent or approximate the same function.

This is a *strict separation*, not a constant-factor difference.

### Meaning of “exponentially more”

If a deep network represents a function using:
$$
O(\text{poly}(k)) \text{ parameters}
$$

Then any shallow network requires:
$$
\Omega(2^k) \text{ hidden units}
$$

This shows that reducing depth cannot be compensated without an exponential blow-up in width.

---

## 2. Width efficiency

### Informal definition

Width efficiency asks:

> Are there functions that wide, shallow networks can represent easily, but narrow networks cannot unless depth increases significantly?

The answer is yes, but the cost is much smaller.

---

### Lu et al. (2017)

They show:
- Some wide shallow networks cannot be represented by narrow networks
- However, narrow networks only require **polynomially more depth**

Formally:
$$
\text{reducing width} \Rightarrow \text{polynomial increase in depth}
$$

---

### Vardi et al. (2022)

For ReLU networks, they strengthen the result:
- Making the network very narrow
- Only requires a **linear increase in depth**

Thus:
$$
\text{small width} \Rightarrow \text{manageable depth increase}
$$

---

## 3. Asymmetry between depth and width

Combining both efficiency results:

| Constraint imposed | Required compensation |
|-------------------|----------------------|
| Reduce depth | Exponential increase in width |
| Reduce width | Polynomial or linear increase in depth |

This asymmetry shows that depth is the more critical resource.

---
## References 

Lu, Z., Pu, H., Wang, F., Hu, Z., & Wang, L. (2017).
The expressive power of neural networks: A view from the width.
Advances in Neural Information Processing Systems (NeurIPS 2017), 30.

Vardi, G., Yehudai, A., & Shamir, O. (2022).
The tradeoff between width and depth in neural networks.
Proceedings of the 35th Conference on Learning Theory (COLT 2022).
