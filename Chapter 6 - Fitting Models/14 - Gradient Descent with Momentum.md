## SGD with Momentum

Standard SGD update:

$$
\phi_{t+1} = \phi_t - \eta \, g_t
$$

$$
g_t =
\frac{1}{|\mathcal{B}_t|}
\sum_{i \in \mathcal{B}_t}
\frac{\partial \mathcal{L}_i(\phi_t)}{\partial \phi}
$$

Stochastic Gradient Descent (SGD) has a key limitation: it does not retain information about past gradients. Each update depends only on the current mini-batch gradient. As a result, if the cumulative gradient over a horizon is the same, SGD behaves identically regardless of the order or consistency of those gradients.

This motivates the introduction of momentum, which computes an exponentially weighted moving average (EMA) of past gradients. By accumulating gradients over time, momentum accelerates convergence in directions where gradients are consistently aligned.

Additionally, due to its EMA structure, momentum reduces oscillations in high-curvature directions and acts as a low-pass filter over noisy stochastic gradients.
### Momentum Update

Momentum introduces an exponentially weighted moving average of past gradients:

$$
m_{t+1} = \alpha m_t + (1-\alpha)
\frac{1}{|\mathcal{B}_t|}
\sum_{i \in \mathcal{B}_t}
\frac{\partial \mathcal{L}_i(\phi_t)}{\partial \phi}
$$
$$
\phi_{t+1} = \phi_t - \eta \, m_{t+1}
$$

### Nesterov Accelerated Momentum

The Nesterov Accelerated Momentum differs from SGD, in the sense that it computes the gradient at the predicted point.

$$
m_{t+1} \leftarrow \beta \, m_t 
+ (1 - \beta) 
\sum_{i \in \mathcal{B}_t}
\frac{\partial \mathcal{L}_i \big[ \phi_t - \alpha \beta \, m_t \big]}{\partial \phi}
$$

$$
\phi_{t+1} \leftarrow \phi_t - \alpha \, m_{t+1}
$$
