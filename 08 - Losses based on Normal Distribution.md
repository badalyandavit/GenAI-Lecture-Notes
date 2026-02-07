### Case 1: Homoscedastic Gaussian Regression

Assume homoscedasticity
$$y \mid x \sim \mathcal{N}(\mu(x),\sigma^2) \quad \text{with constant } \sigma^2$$

Univariate NLL (dropping constants):  
$$\mathcal{L}(x,y)= \frac{(y-\mu(x))^2}{2\sigma^2}$$

Minimizing NLL is equivalent to minimizing MSE:  
$$\min \sum (y-\mu(x))^2$$


Under these settings, the model’s best point prediction under squared error is the conditional mean:  
$$\mu(x)=\mathbb{E}[y\mid x]$$

### Case 2: Heteroscedastic Gaussian Regression

Assume heteroscedasticity:  
$$y \mid x \sim \mathcal{N}(\mu(x),\sigma^2(x))$$
Full univariate NLL:  
$$\mathcal{L}(x,y)= \frac{1}{2}\log \sigma^2(x) + \frac{(y-\mu(x))^2}{2\sigma^2(x)} + \frac{1}{2}\log(2\pi)$$
Due to the $\sigma^2(x)$ term in the denominator, noisy regions contribute to the regression less. Meanwhile, we cannot inflate the variance due to the logarithmic term.

By dropping the constant term, and adding parametrization $s(x) = \log {\sigma^2(x)}$  
$$\mathcal{L}(x,y)=\frac{1}{2}s(x) + \frac{(y-\mu(x))^2}{2\exp(s(x))}$$

### Case 3: Multivariate Normal Outputs

If $\mathbf{y}\in\mathbb{R}^d$:  
$$\mathbf{y}\mid \mathbf{x}\sim \mathcal{N}(\boldsymbol{\mu}(\mathbf{x}),\mathbf{\Sigma}(\mathbf{x}))$$

NLL:  
$$\mathcal{L}(\mathbf{x},\mathbf{y})=\frac{1}{2}\log\det\mathbf{\Sigma}(\mathbf{x})+\frac{1}{2}(\mathbf{y}-\boldsymbol{\mu})^\top \mathbf{\Sigma}^{-1}(\mathbf{x})(\mathbf{y}-\boldsymbol{\mu})+\frac{d}{2}\log(2\pi)$$

Two common covariance choices:

- **Diagonal covariance**: $\mathbf{\Sigma}=\mathrm{diag}(\sigma_1^2,\dots,\sigma_d^2)$  
    Equivalent to predicting independent Gaussians per dimension.
    
- **Full covariance**: more informative, but requires PSD
	Approximate via setting $\mathbf{\Sigma}=\mathbf{L}\mathbf{L}^\top$ where $\mathbf{L}$  is a Cholesky factor (lower-triangular with positive diagonal)

## References

Nix, D. A., & Weigend, A. S. (1994).
Estimating the Mean and Variance of the Target Probability Distribution.
Proceedings of the IEEE International Conference on Neural Networks (ICNN).

Williams, C. K. I., & Rasmussen, C. E. (1996).
Gaussian Processes for Regression.
Advances in Neural Information Processing Systems (NeurIPS).

Bishop, C. M. (2006).
Pattern Recognition and Machine Learning.
Springer.

Dorta, G., Vicente, F., Agapito, L., Campbell, N. D. F., & Simpson, I. (2018).
Training VAEs Under Structured Residuals.
arXiv:1804.01050.
