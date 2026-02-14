## Adam (Adaptive Moment Estimation)

Gradient descent with a fixed learning rate has a structural limitation: the magnitude of the update is directly proportional to the magnitude of the gradient. Parameters associated with large gradients receive large updates, while parameters associated with small gradients receive small updates.

When the loss surface is highly anisotropic, meaning it is steep in one direction and flat in another, choosing a single global learning rate becomes problematic. A learning rate that works well for steep directions may cause instability, while one that is stable in steep regions may result in extremely slow progress along flat directions.

A natural idea is to normalize the gradient coordinate-wise, so that we move approximately a fixed distance in each direction. Consider first the raw gradient and its pointwise square:

$$  
m_{t+1} = \frac{\partial L[\phi_t]}{\partial \phi}  
$$

$$  
v_{t+1} =  
\left(  
\frac{\partial L[\phi_t]}{\partial \phi}  
\right)^2  
$$

A normalized update would then be

$$  
\phi_{t+1}

\leftarrow \phi_t
-
\alpha  
\frac{m_{t+1}}{\sqrt{v_{t+1}} + \epsilon}  
$$

$\epsilon > 0$ prevents division by zero (generally around $10^{-10}$.

Since $v_{t+1}$ is the squared gradient, $\sqrt{v_{t+1}}$ approximates the magnitude of the gradient itself. As a result, the update direction is preserved, but the magnitude becomes approximately constant per coordinate. This reduces sensitivity to curvature differences across dimensions. However, this scheme alone does not accumulate historical information and may oscillate around the minimum.

---

Adam extends this idea by combining two exponential moving averages:

1. A first moment estimate (mean of gradients).
    
2. A second moment estimate (mean of squared gradients).
    

Instead of using only the current gradient, Adam maintains:

$$  
m_{t+1}
\leftarrow
\beta \cdot m_t  
+  
(1-\beta) \cdot 
\frac{\partial L[\phi_t]}{\partial \phi}  
$$

 $$  
v_{t+1} \leftarrow

\gamma \cdot v_t  
+  
(1-\gamma)  \cdot
\left(  
\frac{\partial L[\phi_t]}{\partial \phi}  
\right)^2  
$$

$\beta, \gamma \in [0,1)$ controls the decay rate of the first and moment respectively

---

### Bias Correction

At initialization, $m_0 = 0$ and $v_0 = 0$.  
Therefore, early estimates are biased toward zero. To correct this, Adam applies bias correction:

$$  
\tilde{m}_{t+1}
\leftarrow
\frac{m_{t+1}}{1 - \beta^{t+1}}  
$$

$$  
\tilde{v}_{t+1}
\leftarrow
\frac{v_{t+1}}{1 - \gamma^{t+1}}  
$$

As $t$ increases, the denominators approach 1, and the correction effect diminishes.

---

### Final Update Rule

The parameter update becomes

$$  
\phi_{t+1}
\leftarrow
\phi_t
-
\alpha  \cdot
\frac{\tilde{m}_{t+1}}{\sqrt{\tilde{v}_{t+1}} + \epsilon}  
$$

---

## AdamW

A subtle issue with Adam arises when weight decay is implemented as L2 regularization inside the gradient. In standard Adam, adding an L2 penalty modifies the gradient itself:

$$  
g_t \leftarrow  
\frac{\partial L[\phi_t]}{\partial \phi}

- \lambda \phi_t  
    $$
    

However, because Adam rescales gradients by $\sqrt{v_t}$, the regularization term is also adaptively scaled. This means weight decay no longer behaves like true multiplicative shrinkage of the parameters.

AdamW resolves this by **decoupling weight decay from the gradient update**.

Instead of mixing L2 regularization into the gradient, AdamW performs:

$$  
\phi_{t+1}  
\leftarrow  
\phi_t
-
\alpha  
\frac{\tilde{m}_{t+1}}{\sqrt{\tilde{v}_{t+1}} + \epsilon}
-
\alpha \lambda \phi_t  
$$

Thus, the weight decay term acts directly on the parameters, independent of the adaptive scaling.

In practice, AdamW provides more stable regularization and better generalization, especially in large-scale deep learning models.