| Activation | Function $f(x)$                                                  | Gradient $f'(x)$                                                     |
| ---------- | ---------------------------------------------------------------- | -------------------------------------------------------------------- |
| Sigmoid    | $f(x)=\sigma(x)=\frac{1}{1+e^{-x}}$                              | $f'(x)=\sigma(x)(1-\sigma(x))$                                       |
| Tanh       | $f(x)=\tanh(x)$                                                  | $f'(x)=1-\tanh^2(x)$                                                 |
| ReLU       | $f(x)=\max(0,x)$                                                 | $f'(x)=\begin{cases}1 & x>0 \\ 0 & x<0\end{cases}$                   |
| Leaky ReLU | $f(x)=\max(\alpha x,x)$                                          | $f'(x)=\begin{cases}1 & x>0 \\ \alpha & x<0\end{cases}$              |
| ELU        | $f(x)=\begin{cases}x & x>0 \\ \alpha(e^x-1) & x\le 0\end{cases}$ | $f'(x)=\begin{cases}1 & x>0 \\ \alpha e^x & x\le 0\end{cases}$       |
| CReLU      | $f(x)=[\max(0,x),\max(0,-x)]$                                    | $f'(x)=[\mathbf{1}_{x>0},\mathbf{1}_{x<0}]$                          |
| Swish      | $f(x)=x\sigma(\beta x)$                                          | $f'(x)=\sigma(\beta x)+\beta x\sigma(\beta x)(1-\sigma(\beta x))$    |
| Hard Swish | $f(x)=x\frac{\mathrm{ReLU6}(x+3)}{6}$                            | $f'(x)=\frac{\mathrm{ReLU6}(x+3)}{6}+\frac{x}{6}\mathbf{1}_{-3<x<3}$ |
| GELU       | $f(x)=x\Phi(x)$                                                  | $f'(x)=\Phi(x)+x\phi(x)$                                             |

Notes:

$\Phi(x)=\displaystyle\int_{-\infty}^{x}\frac{1}{\sqrt{2\pi}}e^{-t^2/2}\,dt$

$\phi(x)=\dfrac{1}{\sqrt{2\pi}}e^{-x^2/2}$

$\mathrm{ReLU6}(z)=\min(\max(0,z),6)$

$\alpha\approx 0.01,\quad \beta\approx 1$


