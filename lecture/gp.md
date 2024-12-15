# Gaussian Processes

As the main mathematical construct behind Gaussian Processes, we first introduce the Multivariate Gaussian distribution. We analyze this distribution in some more detail to provide reference results. For a more detailed derivation of the results, refer to {cite}`bishop2006`, Section 2.3. In the rest of this lecture, we define the Gaussian Process and how it can be applied to regression and classification.

## Multivariate Gaussian Distribution

The **univariate** (for a scalar random variable) Gaussian distribution has the form

$$\mathcal{N}(x; \underbrace{\mu, \sigma^2}_{\text{parameters}}) = \frac{1}{\sqrt{2 \pi \sigma^2}} \exp\left\{ - \frac{(x-\mu)^2}{2 \sigma^2}\right\}.$$ (univariate_gaussian)

The two parameters are the mean $\mu$ and variance $\sigma^2$.

The **multivariate** Gaussian distribution then has the following form

$$\mathcal{N}(x; \mu, \Sigma)= \frac{1}{(2\pi)^{d/2}\sqrt{\det (\Sigma)}} \exp \left(-\frac{1}{2}(x-\mu)^{\top}\Sigma^{-1}(x-\mu)\right),$$ (multivariate_gaussian_duplicate)

with

- $x \in \mathbb{R}^d \quad $ - feature / sample / random vector
- $\mu \in \mathbb{R}^d \quad $ - mean vector
- $\Sigma \in \mathbb{R}^{d \times d} \quad$ - covariance matrix

Properties:

- $\Delta^2=(x-\mu)^{\top}\Sigma^{-1}(x-\mu)$ is a quadratic form.
- $\Delta$ is the Mahalanobis distance from $\mu$ to $x$. It collapses to the Euclidean distance for $\Sigma = I$.
- $\Sigma$ is a symmetric positive definite matrix (i.e., its eigenvalues are strictly positive), and its diagonal elements contain the variance, i.e., covariance with itself.
- $\Sigma$ can be diagonalized with real eigenvalues $\lambda_i$ and the corresponding eigenvectors $u_i$ can be chosen to form an orthonormal set.

$$\Sigma u_i = \lambda_i u_i \quad i=1,...,d.$$ (sigma_eigendecomposition)

We construct the orthogonal matrix $U$ as follows.

$$
\begin{aligned}
    & U= \left[\begin{array}{l} u_1^{\top} \\ ... \\ u_d^{\top}\\ \end{array}\right], \\
    & U^{\top}U = U U^{\top}=I, \\
\end{aligned}
$$ (sigma_eigenvectors_matrix)

which fulfills

$$U\Sigma U^{\top} = \lambda,$$ (sigma_eigendecomposition_matrix)

where $\lambda$ denotes a diagonal matrix of eigenvalues. If we apply the variable transformation $y=U(x-\mu)$, we can transform the Gaussian PDF of $x$ to the $y$ coordinates according to the change of variables rule (see the beginning of [Gaussian Mixture Models](gmm.md)).

$$p_Y(y)=p_X(h^{-1}(y)) \underbrace{\left|\frac{\text{d}h^{-1}(y)}{\text{d}y}\right|}_{(det|U_{ij}|)^{-1}=1}$$  (change_of_vars_duplicate)

which leads to

$$p(y) = \Pi_{i=1}^d \frac{1}{\sqrt{2 \pi \lambda_i}} \exp \left\{ - \frac{y_i^2}{2\lambda_i} \right\}.$$

Note that the diagonalization of $\Delta$ leads to a factorization of the PDF into $d$ 1D PDFs.
### 1st and 2nd Moment

$$E[x] = \mu$$ (def_first_moment)

$$E[xx^{\top}] = \mu \mu^{\top} + \Sigma$$ (def_second_moment)

$$Cov(x) = \Sigma$$ (def_cov)

### Conditional Gaussian PDF

Consider the case $x \sim \mathcal{N}(x; \mu, \Sigma)$ being a $d$-dimensional Gaussian random vector. We can partition $x$ into two disjoint subsets $x_a$ and $x_b$.

$$
x=\left(\begin{array}{c}
x_a \\
x_b
\end{array}\right) .
$$ (rv_partition)

The corresponding partitions of the mean vector $\mu$ and covariance matrix $\Sigma$ become

$$
\mu=\left(\begin{array}{l}
\mu_a \\
\mu_b
\end{array}\right)
$$ (mean_partition)

$$
\Sigma=\left(\begin{array}{ll}
\Sigma_{a a} & \Sigma_{a b} \\
\Sigma_{b a} & \Sigma_{b b}
\end{array}\right) .
$$ (cov_partition)

Note that $\Sigma^T=\Sigma$ implies that $\Sigma_{a a}$ and $\Sigma_{b b}$ are symmetric, while $\Sigma_{b a}=\Sigma_{a b}^{\mathrm{T}}$.

We also define the precision matrix $\Lambda = \Sigma^{-1}$, being the inverse of the covariance matrix, where

$$
\Lambda=\left(\begin{array}{ll}
\Lambda_{a a} & \Lambda_{a b} \\
\Lambda_{b a} & \Lambda_{b b}
\end{array}\right)
$$ (precision_partition)

and $\Lambda_{a a}$ and $\Lambda_{b b}$ are symmetric, while $\Lambda_{a b}^{\mathrm{T}}=\Lambda_{b a}$. Note, $\Lambda_{a a}$ is not simply the inverse of $\Sigma_{a a}$.

Now, we want to evaluate $\mathcal{N}(x_a| x_b; \mu, \Sigma)$ and use $p(x_a, x_b) = p(x_a|x_b)p(x_b)$. We expand all terms of the pdf given the split considering all terms that do not involve $x_a$ as constant, and then compare with the generic form of a Gaussian for $p(x_a| x_b)$. We can decompose the equation into quadratic, linear, and constant terms in $x_a$ and find an expression for $p(x_a|x_b)$.

> For all intermediate steps refer to {cite}`bishop2006`.

$$
\begin{aligned}
\mu_{a \mid b} & =\mu_a+\Sigma_{a b} \Sigma_{b b}^{-1}\left(x_b-\mu_b\right) \\
\Sigma_{a \mid b} & =\Sigma_{a a}-\Sigma_{a b} \Sigma_{b b}^{-1} \Sigma_{b a}
\end{aligned}
$$ (cond_pdf_stats)

$$p(x_a| x_b) = \mathcal{N(x; \mu_{a|b}, \Sigma_{a|b})}$$ (gaussian_cond_pdf)

### Marginal Gaussian PDF

For the marginal PDF, we integrate out the dependence on $x_b$ of the joint PDF:

$$p(x_a) = \int p(x_a, x_b) dx_b.$$ (def_marginal_pdf)

We can follow similar steps as above for separating terms that involve $x_a$ and $x_b$. After integrating out the Gaussian with a quadratic term depending on $x_b$ we are left with a lengthy term involving $x_a$ only. By comparison with a Gaussian PDF and re-using the block relation between $\Lambda$ and $\Sigma$ as above, we obtain the marginal

$$p(x_a) = \mathcal{N}(x_a; \mu_a, \Sigma_{a a}).$$ (gaussian_marginal_pdf)

### Bayes Theorem for Gaussian Variables

Generative learning addresses the problem of finding a posterior PDF from a likelihood and prior. The basis is Bayes rule for conditional probabilities

$$p(x|y) = \frac{p(y|x)p(x)}{p(y)}.$$ (def_bayes_rule)

We want to find the posterior $p(x|y)$ and the evidence $p(y)$ under the assumption that the likelihood $p(y|x)$ and the prior are **linear Gaussian models**.

- $p(y|x)$ is Gaussian and has a mean that depends at most linearly on $x$ and a variance that is independent of $x$.
- $p(x)$ is Gaussian.

These requirements correspond to the following structure of $p(x)$ and $p(y|x)$

$$
\begin{aligned}
p(x) & =\mathcal{N}\left(x \mid \mu, \Lambda^{-1}\right) \\
p(y \mid x) & =\mathcal{N}\left(y \mid A x+b, L^{-1}\right).
\end{aligned}
$$ (gaussian_prior_and_likelihood)

From that, we can derive an analytical *evidence* (marginal) and *posterior* (conditional) distributions (for more details see {cite}`bishop2006`):

$$
\begin{aligned}
p(y) & =\mathcal{N}\left(y \mid A \mu+b, L^{-1}+A \Lambda^{-1} A^{\top}\right) \\
p(x \mid y) & =\mathcal{N}\left(x \mid \Sigma\left\{A^{\top} L(y-b)+\Lambda \mu\right\}, \Sigma\right),
\end{aligned}
$$ (gaussian_evidence_and_posterior)

where

$$
\Sigma=\left(\Lambda+A^{\top} L A\right)^{-1} .
$$ (gaussian_posterior_sigma)

### Maximum Likelihood for Gaussians

In generative learning, we need to infer PDFs from data. Given a dataset $X=(x_1, ..., x_N)$, where $x_i$ are i.i.d. random variables drawn from a multivariate Gaussian, we can estimate $\mu$ and $\Sigma$ from the maximum likelihood (ML) (for more details see [Bishop, 2006](https://link.springer.com/book/9780387310732)):

$$\mu_{ML} = \frac{1}{N} \sum_{n=1}^N x_n$$ (ml_mean)

$$\Sigma_{ML} = \frac{1}{N} \sum_{n=1}^N (x-\mu)^{\top}(x-\mu)$$ (ml_cov)

$\mu_{ML}$ and $\Sigma_{ML}$ correspond to the so-called sample or empirical estimates. However, $\Sigma_{ML}$ does not deliver an unbiased estimate of the covariance if we use $\mu_{ML}$ as $\mu$. The difference decreases with $N \to \infty$, but for a low $N$, we have $\Sigma_{ML} < \Sigma$. The practical reason for the discrepancy is that we use $\mu_{ML}$ to estimate the mean, but this may not be the true mean. An unbiased sample variance can be defined using [Bessel's correction](https://en.wikipedia.org/wiki/Bessel%27s_correction):

$$\widetilde{\Sigma} = \frac{1}{N-1} \sum_{n=1}^N (x_n-\mu_{ML})^{\top}(x_n-\mu_{ML})$$ (ml_cov_unbiased)

## The Gaussian Process (GP)

Gaussian Processes are a generalization of the generative learning concept based on Bayes rule and Gaussian distributions as models derived from [Gaussian Discriminant Analysis](https://eecs189.org/docs/notes/n18.pdf). We explain Gaussian Processes (GPs) on the example of regression and discuss an adaptation to discrimination in the spirit of Gaussian Discriminant Analysis (GDA), however not being identical to GDA at the same time. Consider linear regression:

$$
h(x) = \omega^{\top} \varphi(x),
$$ (lin_reg_with_phi)

with $\omega \in \mathbb{R}^{m}$ being the weights, $\varphi \in \mathbb{R}^{m}$ the feature map, and $h(x)$ the hypothesis giving the probability of $y$. We now introduce an *isotropic* Gaussian prior on $\omega$, where isotropic is defined as having a diagonal covariance matrix $\Sigma$ with some scalar $\alpha^{-1}$

$$
p(\omega) = \mathcal{N}(\omega; 0, \alpha^{-1} I),
$$ (gp_prior)

where $0$ is the zero-mean nature of the Gaussian prior, with covariance $\alpha^{-1} I$.

> Note that $\alpha$ is a hyperparameter, i.e. responsible for the model properties, it is called hyper $\ldots$ to differentiate from the weight parameters $\omega$.

Now let's consider the data samples $\left( y^{(i)} \in \mathbb{R}, x^{(i)} \in \mathbb{R}^{m} \right)_{i=1, \ldots, n}$. We are interested in leveraging the information contained in the data samples, i.e., in probabilistic lingo, the joint distribution of $y^{(1)}, \ldots, y^{(n)}$. We can then write

$$
y = \Phi \hspace{2pt} \omega,
$$ (ling_reg_with_Phi)

where $y \in \mathbb{R}^{n}$, $\Phi \in \mathbb{R}^{n \times m}$, and $\omega \in \mathbb{R}^{m}$. $\Phi$ is the well-known *design matrix* from the lecture on [Support Vector Machines](svm.md), Eq. {eq}`ridge_reg_design_matrix`. As we know that $\omega$ is Gaussian, the probability density function (pdf) of $y$ is also Gaussian:

$$
\begin{align}
    \mathbb{E}[y] &= \mathbb{E}[\Phi \omega] = \Phi \mathbb{E}[\omega] = 0 \\
    \text{Cov}[y] &= \mathbb{E}[y y^{\top}] - 0^2 = \Phi \mathbb{E}[\omega \omega^{\top}] \Phi^{\top}= \frac{1}{\alpha} \Phi \Phi^{\top} = K \\
    \Longrightarrow p(y) &= \mathcal{N}(y; 0,K)
\end{align}
$$ (gp_y_stats)

where $K$ is the Grammian matrix we encountered in the SVM lecture.

A Gaussian Process is now defined as:

- $y^{(i)} = h(x^{(i)}), \; i=1, \ldots, n$ have a joint Gaussian PDF, i.e. are fully determined by their 2nd-order statistics (mean and covariance).

## GP for Regression

Take into account Gaussian noise on sample data as a modeling assumption:

$$
\begin{align}
    y^{(i)} &= h(x^{(i)}) + \varepsilon^{(i)}, \quad i=1, \ldots, m \\
    p(y^{(i)} | h(x^{(i)})) &= \mathcal{N} \left( y^{(i)}; h(x^{(i)}), \frac{1}{\beta} \right),
\end{align}
$$ (gp_model)

for an isotropic Gaussian with precision parameter $\beta$.

```{figure} ../imgs/gp/gp_regression_data.png
---
width: 400px
align: center
name: gp_regression_data
---
Regression example.
```

Here, $h(x^{(i)}) = h^{(i)}$ becomes a *latent variable*. For the latent variables (these correspond to the noise-free regression data we considered earlier), we then obtain a Gaussian marginal PDF with some Grammatrix $K$.

$$
p(h) = \mathcal{N}(h; 0, K).
$$ (gp_h_prior)

The kernel function underlying $K$ defines a similarity such that if $x_i$ and $x_j$ are similar, then $h(x_i)$ and $h(x_j)$ should also be similar. The requisite posterior PDF then becomes

$$
p(y|x) = \int p(y | h, x) p(h|x) dh,
$$ (gp_y_conditional)

i.e. the posterior pdf is defined by the marginalization of $p(y, h| x)$ over the latent variable $h$. For the computation of $p(y|x)$ we follow previous computations with adjustments of the notation where appropriate. We define the joint random variable $z$ as

$$
z = \left[\begin{matrix}
    h \\
    y
\end{matrix}\right], \quad h \in \mathbb{R}^{n}, y \in \mathbb{R}^{n}.
$$ (gp_joint_z)

From Eq. {eq}`gaussian_evidence_and_posterior` with prior $p(h)$ and likelihood $p(y|h)$ we recall that the evidence follows

$$
\begin{align}
    \mathbb{E}[y] &= A \mu + b \\
    \text{Cov}[y] &= L^{-1} + A  \Lambda A^{\top},
\end{align}
$$ (gaussian_evidence_and_posterior_duplicate)

into which we now substitute

$$
\begin{align}
    \mu &= 0, \text{ by modeling assumption} \\
    b &= 0, \text{ by modeling assumption} \\
    A &= I \\
    L^{-1} &= \frac{1}{\beta} I \\
    \Lambda^{-1} &= \text{Cov}[h] = K
\end{align}
$$ (gp_assumptions)

$$
\Longrightarrow p(y|x) = \mathcal{N}(y; 0, \frac{1}{\beta}I + K).
$$ (gp_y_conditional_substituted)

Note that the kernel $K$ can be presumed and is responsible for the prediction accuracy by the posterior. There exists a whole plethora of possible choices of K, the most common of which are:

```{figure} ../imgs/gp/gp_kernels.png
---
width: 500px
align: center
name: gp_kernels
---
Common GP kernels (Source: {cite}`duvenaud2014`).
```

Looking at a practical example of a possible kernel:

$$
k(x^{(i)}, x^{(j)}) = \theta_{0} \exp \{- \frac{\theta_{1}}{2} || x^{(i)} - x^{(j)}||^{2}\} + \theta_{2} + \theta_{3} x^{(i) \top} x^{(j)}
$$ (gp_kernel_example)

where $\theta_{0}$, $\theta_{1}$, $\theta_{2}$, $\theta_{3}$ are model hyperparameters. With the above $p(y|x)$, we have identified which Gaussian is inferred from the data. Moreover, we have not yet formulated a predictive posterior for *unseen data*. For this purpose, we extend the data set

$$
\underbrace{\left( y^{(n+1)}, x^{(n+1)} \right)}_{\text{unseen data}}, \quad \underbrace{\left( y^{(n)}, x^{(n)} \right), \ldots, \left( y^{(1)}, x^{(1)} \right)}_{\text{seen data}}.
$$ (gp_data_split)

We reformulate the prediction problem as that of determining a conditional PDF. Given

$$
y = \left[
    \begin{matrix}
        y^{(1)}\\
        \vdots \\
        y^{(n)}
    \end{matrix}
\right],
$$ (gp_joint_rv)

we search for the conditional PDF $p(y^{(n+1)}| y)$. As intermediate, we need the joint PDF

$$
p(\tilde{y}), \quad \tilde{y} = \left[
    \begin{matrix}
        y^{(1)}\\
        \vdots \\
        y^{(n+1)}
    \end{matrix}
\right].
$$ (gp_joint_pdf)

We assume that it also follows a Gaussian PDF. However, as $x^{(n+1)}$ is unknown before prediction we have to make the dependence of the covariance matrix of $p(\tilde{y})$ explicit. We first decompose the full covariance matrix into

$$
Cov[\tilde{y}] = \left[
    \begin{matrix}
        \frac{1}{\beta}I + K &k \\
        k^{\top} &c
    \end{matrix}
\right]= \left[
    \begin{matrix}
        \Sigma_{bb} &\Sigma_{ba} \\
        \Sigma_{ab} &\Sigma_{aa} \\
    \end{matrix}
\right],
$$

with the vector

$$
k = \left[
    \begin{matrix}
        k(x^{(1)}, x^{(n+1)}) \\
        \vdots \\
        k(x^{(n)}, x^{(n+1)})
    \end{matrix}
\right].
$$ (gp_kernel_similarity)

I.e. the corresponding entries in the extended $\tilde{K} = \frac{1}{\alpha} \tilde{\Phi} \tilde{\Phi}^{\top}$ i.e. $\tilde{K}_{1, \ldots, n+1}$, where $\tilde{\Phi}_{n+1} = \varphi(x^{n+1})$. The $c$-entry above is then given by

$$
c = \tilde{K}_{n+1, n+1} + \frac{1}{\beta}.
$$ (gp_query_c)

 Using the same approach as before, we can then calculate the mean and covariance of the predictive posterior for unseen data $p(y^{(n+1)}|y)$. Recalling from before (Eq. {eq}`cond_pdf_stats`):

$$
\mu_{a | b} = \mu_a + \Sigma_{ab} \Sigma^{-1}_{bb}(x_{b} - \mu_{b}),
$$ (gp_predictive_mean)

where we now adjust the entries for our present case with

$$
\begin{align}
    \mu_{a|b} &= \mu_{y^{n+1}|y} \in \mathbb{R} \\
    \mu_{a} &= 0 \\
    \Sigma_{ab} &= k^{\top} \\
    \Sigma^{-1}_{bb} &=  (K + \frac{1}{\beta} I)^{-1} \\
    x_{b} &= y \\
    \mu_{b} &= 0.
\end{align}
$$ (gp_predictive_mean_values)

And for the covariance $\Sigma$ we recall

$$
\Sigma_{a|b} = \Sigma_{aa} - \Sigma_{ab} \Sigma^{-1}_{bb} \Sigma_{ba},
$$ (gp_predicitve_cov)

adjusting for the entries in our present case

$$
\begin{align}
    \Sigma_{aa} &= c \\
    \Sigma_{ab} &= k^{\top} \\
    \Sigma_{bb}^{-1} &= (K + \frac{1}{\beta} I)^{-1} \\
    \Sigma_{ba} &= k \\
    \Sigma_{a|b} &= \Sigma_{y^{(n+1)}|y} \in \mathbb{R}.
\end{align}
$$ (gp_predicitve_cov_values)

From which follows

$$
\begin{align}
    \mu_{y^{(n+1)}|y} &= k^{\top}(K + \frac{1}{\beta}I)^{-1} y \\
    \Sigma_{y^{(n+1)}|y} &= c - k^{\top} (K + \frac{1}{\beta}I)^{-1} k,
\end{align}
$$ (gp_predictive_distribution)

which fully defines the Gaussian posterior PDF of Gaussian-process regression. 

**Example: 1D GP with 1 measurement**

Given is a single measurement data point $(x,y)=(1,2)$ with $x,y\in \mathbb{R}$. Find $p(y | x=1.25)$.

Solution: We start with our model assumption being fully described by the choice of noise variance $\frac{1}{\beta}=10^{-6}$ and the kernel given by

$$k(x,x')=\theta_0 \exp\left(-\frac{1}{2\theta_1}(x-x')^2\right), \quad \text{with} \; \theta_0=1, \theta_1=0.2.$$ (gp_example_1d_kernel)

We then substitute our values into equations {eq}`gp_predictive_distribution` to obtain

$$\begin{align}
K &= k(1,1) = 1 \\
k &= k(1,1.25) = 0.458 \\
c &= k(1.25,1.25) + \frac{1}{10^-6} = 1 \\
\mu_{y^{(n+1)}|y} &= k^{\top}(K + \frac{1}{\beta}I)^{-1} y \\
 &=0.458 \cdot (1 + 10^{-6})^{-1} \cdot 2 = 0.916\\
\Sigma_{y^{(n+1)}|y} &= c - k^{\top} (K + \frac{1}{\beta}I)^{-1} k \\
 &=1 - 0.458 \cdot (1 + 10^{-6})^{-1} \cdot 0.458 = 0.790.
\end{align}
$$ (gp_example_1d_soln)

```{figure} ../imgs/gp/gp_example_1d.png
---
width: 500px
align: center
name: gp_example_1d
---
GP example with confidence interval $\pm 2\sigma$.
```

<!-- import numpy as np
import matplotlib.pyplot as plt

# Kernel function (Squared Exponential / RBF kernel)
def kernel(x1, x2, length_scale=0.2, variance=1.0):
    return variance * np.exp(-0.5 * (np.subtract.outer(x1, x2)**2) / length_scale**2)

# Single observed data point
x_train = np.array([1.0])  # Input
y_train = np.array([2.0])  # Output

# Query point
x_test = np.linspace(0, 2, 100)  # Points to predict

# Kernel matrices
K = kernel(x_train, x_train) + 1e-6 * np.eye(len(x_train))  # Add noise for numerical stability
K_s = kernel(x_train, x_test)  # Cross covariance
K_ss = kernel(x_test, x_test)  # Covariance of test points

# Predictive mean and covariance
K_inv = np.linalg.inv(K)
mu_s = K_s.T @ K_inv @ y_train  # Predictive mean
cov_s = K_ss - K_s.T @ K_inv @ K_s  # Predictive covariance
std_s = np.sqrt(np.diag(cov_s))  # Predictive standard deviation

# Plot the results
plt.figure(figsize=(8, 5))
plt.plot(x_test, mu_s, 'r', label="Mean prediction")
plt.fill_between(x_test, mu_s - 2 * std_s, mu_s + 2 * std_s, color='r', alpha=0.3, label="Confidence interval")
plt.scatter(x_train, y_train, color='b', label="Observed data")
plt.title("Gaussian Process Regression with Single Data Point")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid()
plt.show()

print("Mean=",2*kernel(1,1.25), "Cov=", 1-kernel(1,1.25)**2, "Std=", np.sqrt(1-kernel(1,1.25)**2)) -->

---

Below we look at some sketches of GP regression, beginning with the data sample $y_{1}$, and predicting the data $y_{2}$

```{figure} ../imgs/gp/gp_2d_marginalization.jpeg
---
width: 300px
align: center
name: gp_2d_marginalization
---
Marginalizing a 2D Gaussian distribution.
```

```{figure} ../imgs/gp/gp_regression_example1d.png
---
width: 450px
align: center
name: gp_regression_example1d
---
1D Gaussian process.
```

In practice, for a 1-dimensional function, which is being approximated with a GP, we monitor the change in the mean, variance, and synonymously the GP behavior (recall, the GP is defined by the behavior of its mean and covariance) with each successive data point.

```{figure} ../imgs/gp/gp_regression_examples1d_.png
---
width: 500px
align: center
name: gp_regression_examples1d_
---
Probability distribution of a 1D Gaussian process (Source: {cite}`duvenaud2014`).
```

> This immediately shows the utility of GP-regression for engineering applications where we often have few data points but yet need to be able to provide guarantees for our predictions which GPs offer at a reasonable computational cost.

### Further Kernel Configurations

There exist many ways in which we can extend beyond just individual kernels by multiplication and addition of simple "basis" kernels to construct better-informed kernels. Just taking a quick glance at some of these possible combinations, where the following explores this combinatorial space for the following three kernels (see {numref}`gp_kernels` for definitions):

- Squared Exponential (SE) kernel
- Linear (Lin) kernel
- Periodic (Per) kernel

```{figure} ../imgs/gp/gp_adding_kernels.png
---
width: 500px
align: center
name: gp_adding_kernels
---
Adding 1D kernels (Source: {cite}`duvenaud2014`).
```

```{figure} ../imgs/gp/gp_multiplying_kernels.png
---
width: 500px
align: center
name: gp_multiplying_kernels
---
Multiplying 1D kernels (Source: {cite}`duvenaud2014`).
```

```{figure} ../imgs/gp/gp_multiplying_kernels_to_2d.png
---
width: 500px
align: center
name: gp_multiplying_kernels_to_2d
---
Multiplying 1D kernels, 2D visualization (Source: {cite}`duvenaud2014`).
```

In higher dimensions, this then takes the following shape:

```{figure} ../imgs/gp/gp_kernels_2d.png
---
width: 500px
align: center
name: gp_kernels_2d
---
Examples of 2D kernels (Source: {cite}`duvenaud2014`).
```

With all these possible combinations, one almost begs the question if these kernels cannot be constructed automatically. The answer is partially yes and partially no. In a sense, the right kernel construction is almost like feature engineering, and while it can be automated in parts, it remains a craft for the domain scientists to understand the nature of their problem to then construct the right prior distribution.

```{figure} ../imgs/gp/gp_kernel_tree_search.png
---
width: 500px
align: center
name: gp_kernel_tree_search
---
Search tree over kernels (Source: {cite}`duvenaud2014`).
```

### Multivariate Prediction

The extension to multivariate prediction is then straightforward. For the unseen data

$$
y' = \left[
    \begin{matrix}
        y^{(n+1)} \\
        \vdots \\
        y^{(n+l)}
    \end{matrix}
\right] \in \mathbb{R}^{l}, \quad x' = \left[
    \begin{matrix}
        x^{(n+1)}\\
        \vdots \\
        x^{(n+l)}
    \end{matrix}
\right] \in \mathbb{R}^{l \times m},
$$ (gp_multiple_targets)

and for the sample data

$$
y = \left[
    \begin{matrix}
        y^{(1)} \\
        \vdots \\
        y^{(n)}
    \end{matrix}
\right] \in \mathbb{R}^{n}, \quad x = \left[
    \begin{matrix}
        x^{(1)}\\
        \vdots \\
        x^{(n)}
    \end{matrix}
\right] \in \mathbb{R}^{n \times m},
$$ (gp_multiple_inputs)

for ever larger $l$'s the process can then just be repeated. The mean and covariance matrix are then given by

$$
\begin{align}
    \mu_{y'|y} &= k(x', x) \left( k(x, x) + \frac{1}{\beta} I \right)^{-1} y \\
    \Sigma_{y'|y} &= k(x', x') + \frac{1}{\beta}I \\
    &- k(x', x) \left( k(x, x) + \frac{1}{\beta} I \right)^{-1} k(x, x')
\end{align}
$$ (gp_posterior_multiple_targets)

### Notes on Gaussian Process Regression

- $K + \frac{1}{\beta}I$ needs to be symmetric and positive definite, where positive definite implies that the matrix is symmetric. $K$ can be constructed from valid kernel functions involving some hyperparameters $\theta$.
- GP regression involves a $n \times n$ matrix inversion, which requires $\mathcal{O}(n^{3})$ operations for learning.
- GP prediction involves a $n \times n$ matrix multiplication which requires $\mathcal{O}(n^{2})$ operations.
- The operation count can be reduced significantly when lower-dimensional projections of the kernel function on basis functions can be employed, or we are able to exploit sparse matrix computations.

### Learning the Hyperparameters

To infer the kernel hyperparameters from data, we need to:

1. Introduce an appropriate likelihood function $p(y | \theta)$
2. Determine the optimum $\theta$ via maximum likelihood estimation (MLE) $\theta^{\star} = \arg \max \ln p(y | \theta)$, which corresponds to linear regression
3. $\ln p(y|\theta) = -\frac{1}{2} \ln |K + \frac{1}{\beta}I| - \frac{1}{2} y^{\top} \left( K + \frac{1}{\beta}I \right)^{-1} y - \frac{n}{2} \ln 2 \pi$
4. Use iterative gradient descent or Newton's method to find the optimum, where you need to be aware of the fact that $p(y | \theta)$ may be non-convex in $\theta$, and hence have multiple maxima

$$
\begin{align*}
\frac{\partial}{\partial \theta_{i}} \ln p(y | \theta) &= - \frac{1}{2} \text{trace} \left[ \left(K + \frac{1}{\beta}I\right)^{-1} \frac{\partial(K + \frac{1}{\beta}I)}{\partial \theta_{i}} \right] \\
 &+ \frac{1}{2} y^{\top} \left( K + \frac{1}{\beta} I \right)^{-1} \frac{\partial(K + \frac{1}{\beta}I)}{\partial \theta_{i}}\left(K + \frac{1}{\beta}\right)^{-1}y
\end{align*}
$$

## GP for Classification

Without discussing all the details, we will now briefly sketch how Gaussian Processes can be adapted to classification. Consider for simplicity the 2-class problem:

$$
0 < y < 1, \quad h(x) = \text{sigmoid}(\varphi(x)).
$$

The PDF for $y$ conditioned on the feature $\varphi(x)$ then follows a Bernoulli-distribution:

$$
p(y | \varphi) = \text{sigmoid}(\varphi)^{y} \left( 1 - \text{sigmoid}(\varphi) \right)^{1-y}, \quad i=1, \ldots, n.
$$

Finding the predictive PDF for unseen data $p(y^{(n+1)}|y)$, given the training data

$$
y = \left[
    \begin{matrix}
        y^{(1)} \\
        \vdots \\
        y^{(n)}
    \end{matrix}
\right]
$$

we then introduce a GP prior on

$$
\tilde{\varphi} = \left[
    \begin{matrix}
        \varphi^{(1)} \\
        \vdots \\
        \varphi^{(n+1)}
    \end{matrix}
\right]
$$

hence giving us

$$
p(\tilde{\varphi}) = \mathcal{N}( \tilde{\varphi}; 0, K + \nu I),
$$

where $K_{ij} = k(x^{(i)}, x^{(j)})$, i.e., a Grammian matrix generated by the kernel functions from the feature map $\varphi(x)$.

> - Note that we do **NOT** include an explicit noise term in the data covariance as we assume that all sample data have been correctly classified.
> - For numerical reasons, we introduce a noise-like form $\nu I$ that improves the conditioning of $K$
> - For two-class classification it is sufficient to predict $p(y^{(n+1)} = 1 | y)$ as $p(y^{(n+1)} = 0 | y) = 1 - p(y^{(n+1)} = 1 | y)$

Using the PDF $p(y=1|\varphi) = \text{sigmoid}(\varphi(x))$ we obtain the predictive PDF:

$$
p(y^{(n+1)} = 1 | y) = \int p(y^{(n+1)} = 1 | \varphi^{(n+1)})  p(\varphi^{(n+1)}| y) d\varphi^{(n+1)}
$$

The integration of this PDF is analytically intractable, we are hence faced with a number of choices to integrate the PDF:

- Use sampling-based methods, and Monte-Carlo approximation for the integral
- Assume a Gaussian approximation for the posterior and evaluate the resulting convolution with the sigmoid in an approximating fashion

$p(\varphi^{(n+1)}|y)$ is the posterior PDF, which is computed from the conditional PDF $p(y| \varphi)$ with a Gaussian prior $p(\varphi)$.

## Relation of GP to Neural Networks

We have not presented a definition of a neural network yet, but we have already met the most primitive definition of a neural network, the perceptron, for classification. The perceptron can be considered a neural network with just one layer of "neurons". On the other hand, the number of hidden units should be limited for perceptrons to limit overfitting. The essential power of neural networks, i.e., the outputs sharing the hidden units, tends to get lost when the number of hidden units becomes very large. This is the core connection between neural networks and GPs, **a neural network with infinite width recovers a Gaussian process**.

## Further References

**The Gaussian Distribution**

- {cite}`bishop2006`, Section 2.3

**Gaussian Processes**

- {cite}`bishop2006`, Section 6.4
- {cite}`duvenaud2014`, up to page 22
- {cite}`rasmussen2006`, Sections 2.1-2.6
