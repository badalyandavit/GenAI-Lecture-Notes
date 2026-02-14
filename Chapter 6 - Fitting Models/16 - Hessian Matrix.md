### Hessian Matrix and Curvature

The Hessian matrix is the matrix of second-order partial derivatives of the loss function with respect to the parameters:

$$  
H[\phi] = \left[ \frac{\partial^2 L}{\partial \phi_i \partial \phi_j} \right]_{i,j}  
$$

It captures the local curvature of the loss surface. While the gradient indicates the direction of steepest descent, the Hessian describes how that gradient itself changes in different directions.

If the Hessian is positive definite at a point (all eigenvalues are positive), the function is locally convex there and the point is a local minimum. If all eigenvalues are negative, the point is a local maximum. If the Hessian has both positive and negative eigenvalues, the point is a saddle point.

Interestingly, for (2 x 2) matrices, checking the sign of the trace and determinant is sufficient to determine definiteness.

---

> [!note] Proof (2×2 case)  
> Consider a symmetric (2 \times 2) Hessian:
> 
> $$  
> H =  
> \begin{pmatrix}  
> a & b \  
> b & c  
> \end{pmatrix}  
> $$
> 
> Let its eigenvalues be (\lambda_1, \lambda_2). Then:
> 
> $$  
> \lambda_1 + \lambda_2 = \mathrm{tr}(H),  
> \qquad  
> \lambda_1 \lambda_2 = \det(H).  
> $$
> 
> If
> 
> $$  
> \mathrm{tr}(H) > 0  
> \quad \text{and} \quad  
> \det(H) > 0,  
> $$
> 
> then:
> 
> - (\lambda_1 \lambda_2 > 0) ⇒ eigenvalues have the same sign.
>     
> - (\lambda_1 + \lambda_2 > 0) ⇒ that common sign must be positive.
>     
> 
> Therefore,
> 
> $$  
> \lambda_1 > 0,  
> \quad  
> \lambda_2 > 0,  
> $$
> 
> and (H) is positive definite.
> 
> This argument relies on symmetry (real eigenvalues) and holds only for the (2 \times 2) case.

In convex optimization, if the Hessian is positive definite for all parameter values, the function has a single global minimum and no saddle points. In high-dimensional non-convex problems such as neural networks, the Hessian typically contains both positive and negative eigenvalues, which explains the presence of saddle points and complex curvature structures.
