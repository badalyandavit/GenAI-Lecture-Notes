
When the target is an angle (direction, orientation), linear Gaussians break because $-\pi$ and $\pi$ are the same angle.

The von Mises distribution is the circular analog of a normal distribution on $[-\pi,\pi)$:  
$$  
p(z\mid \mu,\kappa)=\frac{1}{2\pi I_0(\kappa)}\exp\left(\kappa\cos(z-\mu)\right)  
$$

Parameters:

- $\mu$ is the mean direction (where the peak is)
    
- $\kappa>0$ is concentration (higher means tighter around $\mu$)
    
- $I_0(\kappa)$ is a modified Bessel function
    

Why $1/\sqrt{\kappa}$ is like a standard deviation (roughly)  
Near $\mu$, use $\cos(\delta)\approx 1-\delta^2/2$ with $\delta=z-\mu$:  
$$  
\kappa\cos(\delta)\approx \kappa\left(1-\frac{\delta^2}{2}\right)=\kappa-\frac{\kappa}{2}\delta^2  
$$  
So locally:  
$$p(z)\propto \exp\left(-\frac{\kappa}{2}(z-\mu)^2\right)$$  
which resembles a normal with:  
$$\sigma^2\approx \frac{1}{\kappa}\quad\Rightarrow\quad \sigma\approx \frac{1}{\sqrt{\kappa}}$$

Practical tip:

- Predict $\kappa$ via softplus to keep it positive.
    
- For multimodal angles, you can use a mixture of von Mises distributions.
    
## References

Mardia, K. V., & Jupp, P. E. (2000). *Directional Statistics*. Cambridge University Press.

Fisher, N. I. (1993). *Statistical Analysis of Circular Data*. Cambridge University Press.

Jammalamadaka, S. R., & Sengupta, A. (2001). *Topics in Circular Statistics*. World Scientific.

Gumbel, E. J. (1953). A note on the wrapped normal and von Mises distributions. *Journal of the Royal Statistical Society: Series B (Methodological)*, 15(2), 250–252.

Banerjee, A., Dhillon, I. S., Ghosh, J., & Sra, S. (2005). Clustering on the unit hypersphere using von Mises–Fisher distributions. *Journal of Machine Learning Research*, 6, 1345–1382.

Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.

Pewsey, A., Neuhäuser, M., & Ruxton, G. D. (2013). *Circular Statistics in R*. Oxford University Press.