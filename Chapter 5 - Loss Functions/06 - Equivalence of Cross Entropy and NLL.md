The Kullback–Leibler (KL) divergence measures the discrepancy between two probability distributions $q(y)$ and $p(y \mid \theta)$. We assume that these distributions are either analytically available or can be accessed via sampling. It is defined as
$$
D_{\mathrm{KL}}(q \,\|\, p)
= \int_{-\infty}^{\infty} q(y)\,\log q(y)\,dy
- \int_{-\infty}^{\infty} q(y)\,\log p(y \mid \theta)\,dy.
$$

When we observe empirical data $\{y_i\}_{i=1}^{I}$, the empirical distribution can be written as a weighted sum of point masses,
$$
q(y) = \frac{1}{I}\sum_{i=1}^{I} \delta(y - y_i),
$$
where $\delta(\cdot)$ denotes the Dirac delta function.

Our objective is to minimize the KL divergence with respect to $\theta$,
$$
\hat{\theta}
= \arg\min_{\theta}
\left[
\int_{-\infty}^{\infty} q(y)\,\log q(y)\,dy
- \int_{-\infty}^{\infty} q(y)\,\log p(y \mid \theta)\,dy
\right].
$$

Since $q$ is the fixed empirical distribution and does not depend on $\theta$, the first term is constant with respect to $\theta$ and can be dropped. Thus, the objective simplifies to
$$
\hat{\theta}
= \arg\min_{\theta}
\left[
- \int_{-\infty}^{\infty} q(y)\,\log p(y \mid \theta)\,dy
\right].
$$

Substituting the empirical distribution into the remaining term yields
$$
\hat{\theta}
= \arg\min_{\theta}
\left[
- \int_{-\infty}^{\infty}
\left(
\frac{1}{I}\sum_{i=1}^{I} \delta(y - y_i)
\right)
\log p(y \mid \theta)\,dy
\right].
$$

Using the sifting property of the Dirac delta function, this reduces to
$$
\hat{\theta}
= \arg\min_{\theta}
\left[
- \frac{1}{I}\sum_{i=1}^{I} \log p(y_i \mid \theta)
\right].
$$

Since the factor $\frac{1}{I}$ does not affect the minimizer, we obtain
$$
\hat{\theta}
= \arg\min_{\theta}
\left[
- \sum_{i=1}^{I} \log p(y_i \mid \theta)
\right],
$$
which is the negative log-likelihood objective. In the discrete case, this is equivalent to minimizing the cross-entropy loss.
