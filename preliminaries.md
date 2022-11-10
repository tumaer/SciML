# Preliminaries

## Probability Theory

### Basic Building Blocks
- $\Omega$ - sample space: the set of all outcomes of a random experiment.
- $\mathbb{P}(E)$ - probability measure of an event $E \in \Omega$: a function $\mathbb{P}: \Omega \rightarrow \mathbb{R}$ which satisfies the following three properties:
    - $0 \le \mathbb{P}(E) \le 1 \quad \forall E \in \Omega$
    - $\mathbb{P}(\Omega)=1$
    - $\mathbb{P}(\cup_{i=1}^n E_i) = \sum_{i=1}^n \mathbb{P}(E_i) \;$ for disjoint events ${E_1, ..., E_n}$
- $\mathbb{P}(A, B)$ - joint probability: probability that both $A$ and $B$ occur simultaneously.
- $\mathbb{P}(A | B)$ - conditional probability: probability that $A$ occurs, if $B$ has occured.
- Product rule of probabilites:
    - general case: 
    $$\mathbb{P}(A, B) = \mathbb{P}(A | B)\cdot  \mathbb{P}(B) = \mathbb{P}(B | A) \cdot \mathbb{P}(A)$$
    - independent events:
    $$\mathbb{P}(A, B) = \mathbb{P}(A) \cdot \mathbb{P}(B)$$
- Sum rule of probabilites: 
$$\mathbb{P}(A)=\sum_{B}\mathbb{P}(A, B)$$
- Bayes rule: solving the general case of the product rule for $\mathbb{P}(A)$ results in:
$$ \mathbb{P}(B|A) = \frac{\mathbb{P}(A|B) \mathbb{P}(B)}{\mathbb{P}(A)} = \frac{\mathbb{P}(A|B) \mathbb{P}(B)}{\sum_{i=1}^n \mathbb{P}(B|A_i)\mathbb{P}(A_i)}$$
    - $p(B|A)$: posterior
    - $p(A|B)$: likelihood
    - $p(B)$: prior
    - $p(A)$: evidence
    
### Random Variables and Their Properties
- Random variable (r.v.) $X$: a function $X:\Omega \rightarrow \mathbb{R}$. This is the formal way by which we move from abstract events to real-valued numbers. $X$ is essentially a variable that does not have a fixed value, but can have different values with certain probabilities.
- Continuous r.v.:
    - Cumulative distribution function (cdf) $F_X(x)$ - probability that the r.v. $X$ is smaller than some value $x$:
    $$F_X(x) = \mathbb{P}(X\le x)$$
    - Probability density function (pdf) $p_X(x)$:
    $$p_X(x)=\frac{dF_X(x)}{dx}\ge 0 \;\text{ and } \; \int_{-\infty}^{+\infty}p_X(x) dx =1$$

<div style="text-align:center">
    <img src="https://i.imgur.com/uHHQU4r.png" alt="drawing" width="400"/>
</div>

- discrete r.v.:
    - Probability mass function (pmf) - same as the pdf but for a discrete r.v. $X$. Integrals become sums.
- $\mu = E[X]$ - mean value or expected value:
$$E[X] = \int_{-\infty}^{+\infty}x \, p_X(x) \, dx$$
- $\sigma^2 = Var[X]$ - variance:
$$Var[X] = \int_{-\infty}^{+\infty}x^2 \, p_X(x) \, dx = E[(X-\mu)^2]$$
- $Cov[X,Y]=E[(X-\mu_X)(Y-\mu_Y)]$
- Change of variables - if $X \sim p_X$ and $Y=h(X)$, then the distribution of $Y$ becomes:
$$p_Y(y)=\frac{p_X(h^{-1}(y))}{\left|\frac{dh}{dx}\right|}$$ 

### Catalogue of important distributions

- Binomial, $X\in\{0,1,...,n\}$. Describes how often we get $k$ positive outcomes out of $n$ independent experiments. Parameter $\lambda$ is the success probability of each trial.
$$\mathbb{P}(X=k|\lambda)=\binom{n}{k}\lambda^k(1-\lambda)^{n-k}, \quad \text{ with } k\in(1,2,..., n).$$
- Bernoulli - special case of Binomial with $n=1$.
- Normal, $X \in \mathbb{R}$.
$$p(x| \mu, \sigma)=\mathcal{N}(x|\mu, \sigma^2) = \frac{1}{\sqrt{2 \pi \sigma^2}}\exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$
- Multivariate Gaussian $\mathcal{N}(\mathbf{\mu}, \mathbf{\Sigma}), \; \mathbf{X}\in \mathbb{R}^n, \; \mathbf{\mu}\in \mathbb{R}^n, \; \mathbb{\Sigma}:\mathbf{n}\times\mathbf{n}$
$$p_X(x)= \frac{1}{(2\pi)^{n/2}\sqrt{\det (\mathbf{\Sigma})}} \exp \left(-\frac{1}{2}(\mathbf{x}-\mathbf{\mu})^{\top}\mathbf{\Sigma}^{-1}(\mathbf{x}-\mathbf{\mu}\right),$$
where $\mathbf{\mu}\in \mathbb{R}^n$: mean vector and $\mathbf{\Sigma}$: covariance matrix.