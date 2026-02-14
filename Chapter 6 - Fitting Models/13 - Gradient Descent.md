### Gradient Descent (GD)
The advancement of GD was primarily inspired by the difficulty in finding exact solutions for a network. In particular, a mini case of this includes Linear Regression, which has cubic complexity in closed form (assuming the inverse of the matrix exists). As the scale grows, finding exact solutions becomes computationally harder.

The goal is to fit a model with respect to some objective function and input/output data.

The primary objective becomes  
$$  
\hat{\phi} = \arg\min_\phi L[\phi]  
$$

While this objective admits multiple optimization strategies, most practical training methods are iterative. The simplest method is gradient descent:

$$  
\phi_{new} \leftarrow \phi_{old} - \alpha \cdot \frac{dL}{d\phi}  
$$

where $\alpha > 0$ determines the magnitude of change with respect to the (partial) derivative of the parameters to the loss.

Solving linear regression via the closed-form normal equation has complexity  
$\mathcal{O}(N d^2 + d^3)$,  
while gradient descent has complexity  
$\mathcal{O}(T N d)$,  
where $N$ is the number of samples, $d$ is the number of features, and $T$ is the number of iterations.

A key limitation of gradient descent with a fixed step size is that the update magnitude depends entirely on the gradient magnitude. The algorithm moves a long distance when the function is changing rapidly, where it may need to be more cautious, and a short distance when the function is changing slowly, where it may need to explore further. This often leads to inefficient oscillatory behavior in narrow valleys and slow convergence in flat regions.

For this reason, gradient descent is often combined with a line search procedure, where the algorithm samples the objective along the descent direction to determine a more appropriate step size. The purpose of line search is to balance stability and progress by adapting the step length to the local geometry of the loss surface.
### Stochastic Gradient Descent (SGD)

One failure mode of gradient descent occurs when the loss surface contains large flat regions or saddle points. In such regions, gradients become very small, and the algorithm may progress extremely slowly. In non-convex optimization, another issue is the presence of local minima. A local optimum is a point that is better than its immediate neighborhood, while a global optimum is the best possible solution over the entire loss surface.

When the surface is flat, gradient-based methods receive very little signal about which direction to move. In classical optimization problems such as hill climbing, techniques like simulated annealing introduce randomness through a temperature parameter. This allows the algorithm to occasionally make unfavorable moves, which helps it explore the landscape more broadly and potentially escape poor local optima. In the worst case, one can keep track of the best solution encountered and revert to it.

During iterative training, however, we generally cannot determine whether we have reached a local minimum, a saddle point, or the global minimum. This uncertainty motivates methods that introduce stochasticity. Stochastic Gradient Descent differs from full-batch gradient descent in that it computes gradients using only a subset of the data at each step. Because the gradient is estimated from a mini-batch, it contains noise. This noise causes the updates to vary more strongly in the local neighborhood, which in practice can help the algorithm move away from shallow local minima or saddle points. Additionally, computing gradients on subsets reduces computational cost per iteration.

One potential issue introduced by SGD is variance in the gradient estimates, especially if the data distribution across batches is uneven. To mitigate bias, data are typically sampled uniformly, often with replacement, ensuring that each example has equal probability of contributing to the gradient. This makes the stochastic gradient an unbiased estimator of the true gradient while preserving computational efficiency.

Finally, empirical evidence in several domains suggests that SGD often converges to solutions that generalize better than those found by full-batch gradient descent. One common hypothesis is that the stochastic noise in SGD biases the optimization trajectory toward flatter minima. While there exist theoretical results supporting aspects of this behavior, a complete optimization-based characterization explaining this phenomenon in general deep non-convex settings remains an open research question.


