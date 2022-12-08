# Gaussian Processes

As the main mathematical construct behind Gaussian Processes, we first introduce the Multivariate Gaussian distribution. We will analyze this distribution in some more detail to provide reference results. For a more detailed derivation of the results, refer to [Bishop, 2006](https://link.springer.com/book/9780387310732), Section 2.3.

## Multivariate Gaussian Distribution

The **univariate** (for a scalar random variable) Gaussian distribution has the form

$$\mathcal{N}(x; \underbrace{\mu, \sigma^2}_{\text{parameters}}) = \frac{1}{\sqrt{2 \pi \sigma^2}} \exp\left\{ - \frac{(x-\mu)^2}{2 \sigma^2}\right\}.$$

The two parameters are the mean $\mu$ and variance $\sigma^2$.

The multivariate Gaussian distribution then has the following form


$$\mathcal{N}(x; \mu, \Sigma)= \frac{1}{(2\pi)^{d/2}\sqrt{\det (\Sigma)}} \exp \left(-\frac{1}{2}(x-\mu)^{\top}\Sigma^{-1}(x-\mu)\right),$$

with 

- $x \in \mathbb{R}^d \quad $ - feature / sample / random vector
- $\mu \in \mathbb{R}^d \quad $ - mean vector
- $\Sigma \in \mathbb{R}^{d \times d} \quad$ - covariance matrix

Properties:
- $\Delta^2=(x-\mu)^{\top}\Sigma^{-1}(x-\mu)$ is a quadratic form.
- $\Delta$ is the Mahalanobis distance from $\mu$ to $x$. It collapses to the Euclidean distance for $\Sigma = I$.
- $\Sigma$ is symmetric positive semi-definite and its diagonal elements contain the variance, i.e. covariance with itself.
- $\Sigma$ can be diagonalized with real eigenvalues $\lambda_i$

$$\sum u_i = \lambda_i u_i \quad i=1,...,d$$

and eigenvectors $u_i$ forming a unitary matrix

$$U= \left[\begin{array}{l} u_1^{\top} \\ ... \\ u_d^{\top}\\ \end{array}\right], \quad U^{\top}U=I$$

$$U\Sigma U^{\top} = \lambda, \quad \lambda \text{ diagonal matrix of eigenvalues}$$

If we apply the variable transformation $y=U(x-\mu)$, we can transform the Gaussian PDF to the $y$ coordinates according to the change of variables rule (see preliminaries)

$$p_X(x) = p_Y(y) \underbrace{\left| \frac{\partial y}{\partial x} \right|}_{det|U_{ij}|},$$

which leads to 

$$p(y) = \Pi_{i=1}^d \frac{1}{\sqrt{2 \pi \lambda_i}} \exp \left\{ - \frac{y_i^2}{2\lambda_i} \right\}$$

Note that the diagonalization of $\Delta$ leads to a factorization of the PDF into $d$ 1D PDFs.
### 1st and 2nd Moment

$$E[x] = \mu$$

$$E[xx^{\top}] = \mu \mu^{\top} + \Sigma$$

$$Cov(x) = \Sigma$$

### Conditional Gaussian PDF

Consider the case $x \sim \mathcal{N}(x; \mu, \Sigma)$ being a $d$-dimensional Gaussian random vector. We can partition $x$ into two disjoint subsets $x_a$ and $x_b$. 


$$
x=\left(\begin{array}{c}
x_a \\
x_b
\end{array}\right) .
$$

The corresponding partitions of the mean vector $\mu$ and covariance matrix $\Sigma$ become

$$
\mu=\left(\begin{array}{l}
\mu_a \\
\mu_b
\end{array}\right)
$$

$$
\Sigma=\left(\begin{array}{ll}
\Sigma_{a a} & \Sigma_{a b} \\
\Sigma_{b a} & \Sigma_{b b}
\end{array}\right) .
$$

Note that $\Sigma^T=\Sigma$ implies that $\Sigma_{a a}$ and $\Sigma_{b b}$ are symmetric, while $\Sigma_{b a}=\Sigma_{a b}^{\mathrm{T}}$.

We also define the precision matrix $\Lambda = \Sigma^{-1}$, being the inverse of the covariance matrix, where

$$
\Lambda=\left(\begin{array}{ll}
\Lambda_{a a} & \Lambda_{a b} \\
\Lambda_{b a} & \Lambda_{b b}
\end{array}\right)
$$

and $\Lambda_{a a}$ and $\Lambda_{b b}$ are symmetric, while $\Lambda_{a b}^{\mathrm{T}}=\Lambda_{b a}$. Note, $\Lambda_{a a}$ is not simply the inverse of $\Sigma_{a a}$.

Now, we want to evaluate $\mathcal{N}(x_a| x_b; \mu, \Sigma)$ and use $p(x_a, x_b) = p(x_a|x_b)p(x_b)$. We expand all terms of the pdf given the split, and consider all terms that do not involve $x_a$ as constant and then we compare with the generic form of a Gaussian for $p(x_a| x_b)$. We can decompose the equation into quadratic, linear and constant terms in $x_a$ and find an expression for $p(x_a|x_b)$.

> For all intermediate steps refer to [Bishop, 2006](https://link.springer.com/book/9780387310732).

$$
\begin{aligned}
\mu_{a \mid b} & =\mu_a+\Sigma_{a b} \Sigma_{b b}^{-1}\left(x_b-\mu_b\right) \\
\Sigma_{a \mid b} & =\Sigma_{a a}-\Sigma_{a b} \Sigma_{b b}^{-1} \Sigma_{b a}
\end{aligned}
$$

$$p(x_a| x_b) = \mathcal{N(x; \mu_{a|b}, \Sigma_{a|b})}$$

### Marginal Gaussian PDF

For the marginal PDF we integrate out the dependence on $x_b$ of the joint PDF:

$$p(x_a) = \int p(x_a, x_b) dx_b.$$

We can follow similar steps as above for separating terms that involve $x_a$ and $x_b$. After integrating out the Guassian with a quadratic term depending on $x_b$ we are left with a lengthy term involving $x_a$ only. By comparison with a Gaussian PDF and re-using the block relation between $\Lambda$ and $\Sigma$ as above we obtain for the marginal

$$p(x_a) = \mathcal{N}(x_a; \mu_a, \Sigma_{a a}).$$

### Bayes Theorem for Gaussian Variables

Generative learning addresses the problem of finding a posterior PDF from a likelihood and prior. The basis is Bayes rule for conditional probabilities 

$$p(x|y) = \frac{p(y|x)p(x)}{p(y)}$$

We want to find the posterior $p(x|y)$ and the evidence $p(y)$ under the assumption that the likelihood $p(y|x)$ and the prior are **linear Gaussian models**.

- $p(y|x)$ is Gaussian and has a mean that depends at most linearly on $x$ and a variance that is independent of $x$.
- $p(x)$ is Gaussian.

These requirements correspond to the following structure of $p(x)$ and $p(y|x)$

$$
\begin{aligned}
p(x) & =\mathcal{N}\left(x \mid \mu, \Lambda^{-1}\right) \\
p(y \mid x) & =\mathcal{N}\left(y \mid A x+b, L^{-1}\right).
\end{aligned}
$$

From that we can derive an analytical evidence (marginal) and posterior (conditional) distributions (for more details see [Bishop, 2006](https://link.springer.com/book/9780387310732)):

$$
\begin{aligned}
p(y) & =\mathcal{N}\left(y \mid A \mu+b, L^{-1}+A \Lambda^{-1} A^{\top}\right) \\
p(x \mid y) & =\mathcal{N}\left(x \mid \Sigma\left\{A^{\top} L(y-b)+\Lambda \mu\right\}, \Sigma\right),
\end{aligned}
$$

where

$$
\Sigma=\left(\Lambda+A^{\top} L A\right)^{-1} .
$$

### Maximum Likelihood for Gaussians

In generative learning, we need to infer PDFs from data. Given a dataset $X=(x_1, ..., x_N)$, where $x_i$ are i.i.d. random variables drawn from a multivariate Gaussian, we can estimate $\mu$ and $\Sigma$ from the maximum likelihood (ML) (for more details see [Bishop, 2006](https://link.springer.com/book/9780387310732)):

$$\mu_{ML} = \frac{1}{N} \sum_{n=1}^N x_n$$

$$\Sigma_{ML} = \frac{1}{N} \sum_{n=1}^N (x-\mu)^{\top}(x-\mu)$$


$\mu_{ML}$ and $\Sigma_{ML}$ correspond to the so-called sample or empirical estimates. However, $\Sigma_{ML}$ does not deliver an unbiased estimate of the covariance. The difference decreases with $N \to \infty$, but for a certain $N$ we have $\Sigma_{ML}\neq \Sigma$. The practical reason used in the above derivation is that the $\mu_{ML}$ estimate may occur as $\bar{x}$ within the sampling of $\Sigma$ $\Rightarrow$ miscount by one. An unbiased sample variance can be defined as

$$\widetilde{\Sigma} = \frac{1}{N-1} \sum_{n=1}^N (x_n-\mu_{ML})^{\top}(x_n-\mu_{ML})$$


## Further References

- [Pattern Recognition and Machine Learning](https://link.springer.com/book/9780387310732), Section 2.3; Christopher Bishop; 2006