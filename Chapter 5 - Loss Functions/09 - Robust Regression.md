Robust losses aim to reduce sensitivity to outliers or heavy-tailed noise.

### 1) Laplace likelihood leads to MAE and the median

Assume:  
$$y\mid x \sim \mathrm{Laplace}(\mu(x),b)$$

PDF:  
$$p(y\mid x)=\frac{1}{2b}\exp\left(-\frac{|y-\mu(x)|}{b}\right)$$

NLL (dropping constants):  
$$\mathcal{L}(x,y)\propto |y-\mu(x)|$$

So minimizing Laplace NLL is MAE minimization:  
$$\min \sum |y-\mu(x)|$$

Under absolute error, the optimal point prediction is the conditional median:  
$$
\mu(x)=\mathrm{median}(y\mid x)
$$   
While squared error grows in quadratic terms, absolute error grows linearly, so extreme points do not explore the loss the same way. However, due to the kink at 0, MAE is less smooth than MSE. 

### 2) Barron’s adaptive robust loss

Barron’s loss defines a continuous family indexed by a “shape” (robustness) parameter $\alpha$ and a scale $c$.

A common form (for residual $r=y-\hat{y}$) is:

For $\alpha \neq 0,2$:  
$$  
\rho(r;\alpha,c)=\frac{|\alpha-2|}{\alpha}\left(\left(\frac{(r/c)^2}{|\alpha-2|}+1\right)^{\alpha/2}-1\right)  
$$

Special cases (limits):

- $\alpha\to 2$ behaves like L2 (MSE)
    
- $\alpha\to 1$ behaves like Charbonnier / pseudo-Huber style
    
- $\alpha\to 0$ behaves like a log penalty (Cauchy-like)
    
- $\alpha\to -\infty$ approaches a bounded loss
    

$\rho$ can be interpreted as the negative log of a univariate density, which smoothly interpolates between normal-like and heavy-tailed behavior.

### 3) Focal loss as an alternative to Cross-Entropy


For binary classification, let:

- $p\in(0,1)$ be the model’s probability for class 1
    
- $y\in{0,1}$
    
- Define $p_t$ as the probability of the true class:  
    $$p_t=  
    \begin{cases}  
    p & \text{if } y=1\  
    1-p & \text{if } y=0  
    \end{cases}  
    $$
    

Focal loss:  
$$\mathrm{FL}(p_t)= -\alpha_t(1-p_t)^\gamma \log(p_t)$$

Parameters:

- $\gamma\ge 0$ is the focusing parameter. Larger $\gamma$ down-weights well-classified examples more strongly.
    
- $\alpha_t$ is an optional class weight to balance positives vs negatives.
    
Key intuition:

- If an example is easy, $p_t$ is near 1, then $(1-p_t)^\gamma$ is near 0, so it contributes little.
    
- If an example is hard, $p_t$ is small, the modulating factor is near 1, so it gets emphasized.

### 4) Hinge and Exponential Loss
We don't strictly need a likelihood, since any  validated distance among the prediction and target labels can be considered a loss.

### Hinge
Binary labels $y\in{-1,+1}$ and score $f(x)\in\mathbb{R}$:  
$$\mathcal{L}_{hinge}=\max(0,1-yf(x))$$

Acts similar to ReLU, especially useful when we want to penalize a particular type of mislabeling
    
### AdaBoost

$$\mathcal{L}_{exp}=\exp(-y f(x))$$
    
## References (copy-paste friendly)
    
- Koenker, R., & Hallock, K. F. (2001). _Quantile Regression_. **Journal of Economic Perspectives**, 15(4), 143–156.
    
- Bassett, G., & Koenker, R. (1978). _Asymptotic theory of least absolute error regression_. **Journal of the American Statistical Association**, 73(363), 618–622.
    
- Huber, P. J. (1964). _Robust Estimation of a Location Parameter_. **Annals of Mathematical Statistics**, 35(1), 73–101.
    
- Barron, J. T. (2019). _A General and Adaptive Robust Loss Function_. **Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)**.
    
- Bishop, C. M. (2006). _Pattern Recognition and Machine Learning_. Springer. 

- Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017).
  Focal Loss for Dense Object Detection.
  **Proceedings of the IEEE International Conference on Computer Vision (ICCV).**
