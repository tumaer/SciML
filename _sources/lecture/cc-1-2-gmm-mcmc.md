# GMM and MCMC

This lesson first recaps on Probability Theory and then introduces Gaussian Mixture Models (GMM) and some popular sampling methods like Markov Chain Monte Carlo (MCMC).

On a high level, GMMs and MCMC are two complementary approaches:
- GMMs estimate the density of a given set of samples
- MCMC generates samples from a given density


```{figure} ../imgs/cc1/density_estimation_vs_sampling.png
---
width: 500px
align: center
name: density_estim_vs_sampling
---
Density estimation vs sampling.
```

But first, we revise Probability Theory.

## Probability Theory

### Basic Building Blocks
- $\Omega$ - *sample space*; the set of all outcomes of a random experiment.
- $\mathbb{P}(E)$ - *probability measure of an event $E \in \Omega$*; a function $\mathbb{P}: \Omega \rightarrow \mathbb{R}$  satisfies the following three properties:
    - $0 \le \mathbb{P}(E) \le 1 \quad \forall E \in \Omega$
    - $\mathbb{P}(\Omega)=1$
    - $\mathbb{P}(\cup_{i=1}^n E_i) = \sum_{i=1}^n \mathbb{P}(E_i) \;$ for disjoint events ${E_1, ..., E_n}$
- $\mathbb{P}(A, B)$ - *joint probability*; probability that both $A$ and $B$ occur simultaneously.
- $\mathbb{P}(A | B)$ - *conditional probability*; probability that $A$ occurs, if $B$ has occured.
- Product rule of probabilities:
    - general case: <br>

    $$\mathbb{P}(A, B) = \mathbb{P}(A | B)\cdot  \mathbb{P}(B) = \mathbb{P}(B | A) \cdot \mathbb{P}(A)$$ (product_rule_general)

    - independent events: <br>

    $$\mathbb{P}(A, B) = \mathbb{P}(A) \cdot \mathbb{P}(B)$$ (product_rule_indep)

- Sum rule of probabilities: 

$$\mathbb{P}(A)=\sum_{B}\mathbb{P}(A, B)$$ (sum_rule)

- Bayes rule: solving the general case of the product rule for $\mathbb{P}(A)$ results in:

    $$ \mathbb{P}(B|A) = \frac{\mathbb{P}(A|B) \mathbb{P}(B)}{\mathbb{P}(A)} = \frac{\mathbb{P}(A|B) \mathbb{P}(B)}{\sum_{i=1}^n \mathbb{P}(A|B_i)\mathbb{P}(B_i)}$$ (bayes_rule)

    - $p(B|A)$ - *posterior*
    - $p(A|B)$ - *likelihood*
    - $p(B)$ - *prior*
    - $p(A)$ - *evidence*
    
### Random Variables and Their Properties
- *Random variable* (r.v.) $X$ is a function $X:\Omega \rightarrow \mathbb{R}$. This is the formal way by which we move from abstract events to real-valued numbers. $X$ is essentially a variable that does not have a fixed value, but can have different values with certain probabilities.
- Continuous r.v.:
    - $F_X(x)$ - *Cumulative distribution function* (CDF); probability that the r.v. $X$ is smaller than some value $x$:
    
    $$F_X(x) = \mathbb{P}(X\le x)$$ (cdf)

    - $p_X(x)$ - *Probability density function* (PDF):

    $$p_X(x)=\frac{dF_X(x)}{dx}\ge 0 \;\text{ and } \; \int_{-\infty}^{+\infty}p_X(x) dx =1$$ (pdf)



```{figure} ../imgs/cc1/pdf_cdf.png
---
width: 400px
align: center
name: pdf_cdf
---
PDF and CDF functions.
```


- discrete r.v.:
    - *Probability mass function* (PMF) - same as the pdf but for a discrete r.v. $X$. Integrals become sums.
- $\mu = E[X]$ - *mean value* or *expected value*

$$E[X] = \int_{-\infty}^{+\infty}x \, p_X(x) \, dx$$ (mean)

- $\sigma^2 = Var[X]$ - *variance*

$$Var[X] = \int_{-\infty}^{+\infty}x^2 \, p_X(x) \, dx = E[(X-\mu)^2]$$ (variance)

- $Cov[X,Y]=E[(X-\mu_X)(Y-\mu_Y)]$ - *covariance*
- *Change of variables* - if $X \sim p_X$ and $Y=h(X)$, then the distribution of $Y$ becomes:

$$p_Y(y)=\frac{p_X(h^{-1}(y))}{\left|\frac{dh}{dx}\right|}$$  (change_of_vars)

### Catalogue of Important Distributions

- *Binomial*, $X\in\{0,1,...,n\}$. Describes how often we get $k$ positive outcomes out of $n$ independent experiments. Parameter $\lambda$ is the success probability of each trial.

$$\mathbb{P}(X=k|\lambda)=\binom{n}{k}\lambda^k(1-\lambda)^{n-k}, \quad \text{ with } k\in(1,2,..., n).$$ (binomial)

- *Bernoulli* - special case of Binomial with $n=1$.
- *Normal* (aka *Gaussian*), $X \in \mathbb{R}$.

$$p(x| \mu, \sigma)=\mathcal{N}(x|\mu, \sigma^2) = \frac{1}{\sqrt{2 \pi \sigma^2}}\exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$ (gaussian)

- *Multivariate Gaussian* $\mathcal{N}(\mathbf{\mu}, \mathbf{\Sigma})$ of $\mathbf{X}\in \mathbb{R}^n$ with mean $\mathbf{\mu}\in \mathbb{R}^n $ and covariance $\mathbb{\Sigma} \in \mathbb{R}_{+}^{n\times n}$.

$$p_X(x)= \frac{1}{(2\pi)^{n/2}\sqrt{\det (\mathbf{\Sigma})}} \exp \left(-\frac{1}{2}(\mathbf{x}-\mathbf{\mu})^{\top}\mathbf{\Sigma}^{-1}(\mathbf{x}-\mathbf{\mu})\right).$$ (multivariate_gaussian)

### Exponential Family

The exponential family of distributions is a large family of distributions with shared properties, some of which we have already encountered in other courses before. Prominent members of the exponential family include:

- Bernoulli
- Gaussian
- Dirichlet
- Gamma
- Poisson
- Beta

At their core, members of the exponential family all fit the same general probability distribution form

$$p(x|\eta) = h(x) \exp \left\{ \eta^{\top} t(x) - a(\eta) \right\},$$ (exponential_pdfs)

where the individual components are:

- $\eta$ - *natural parameter*
- $t(x)$ - *sufficient statistic*
- $h(x)$ - *probability support measure*
- $a(\eta)$ - *log normalizer*; guarantees that the probability density integrates to 1.

> If you are unfamiliar with the concept of probability measures, then $h(x)$ can safely be disregarded. Conceptually it describes the area in the probability space over which the probability distribution is defined.

**Why is this family of distributions relevant to this course?** 

> The exponential family has a direct connection to graphical models, which are a formalism favored by many people to visualize machine learning models, and the way individual components interact with each other. As such they are highly instructive, and at the same time foundational to many probabilistic approaches covered in this course.

Let's inspect the practical example of the Gaussian distribution to see how the theory translates into practice. Taking the probability density function which we have also previously worked with

$$p(x|\mu, \sigma^{2}) = \frac{1}{\sqrt{2 \pi \sigma^{2}}} \exp \left\{ \frac{(x - \mu)^{2}}{2 \sigma^{2}} \right\}.$$ (gaussian_2)

We can then expand the square in the exponent of the Gaussian to isolate the individual components of the exponential family

$$p(x|\mu, \sigma^{2}) = \frac{1}{\sqrt{2 \pi}} \exp \left\{ \frac{\mu}{\sigma^{2}}x - \frac{1}{2 \sigma^{2}}x^{2} - \frac{1}{2 \sigma^{2}} \mu^{2} - \log \sigma \right\}.$$ (gaussian_expanded)

Then the individual components of the Gaussian are

$$\eta = \langle \frac{\mu}{\sigma^{2}}, - \frac{1}{2 \sigma^{2}} \rangle$$
$$t(x) = \langle x, x^{2} \rangle$$
$$a(\eta) = \frac{\mu^{2}}{2 \sigma^{2}} + \log \sigma$$
$$h(x) = \frac{1}{\sqrt{2 \pi}}$$ (gaussian_as_exponential)

For the sufficient statistics, we then need to derive the derivative of the log normalizer, i.e 

$$\frac{d}{d\eta}a(\eta) = \mathbb{E}\left[ t(X) \right]$$ (gaussian_as_exponential_suff_stats)

Which yields

$$\frac{da(\eta)}{d\eta_{1}} = \mu = \mathbb{E}[X] $$
$$\frac{da(\eta)}{d\eta_{2}} = \sigma^2 - \mu^2 = \mathbb{E}[X^{2}] $$ (gaussian_as_exponential_suff_stats_2)

**Exercise: Exponential Family 1**

Show that the Dirichlet distribution is a member of the exponential family.


**Exercise: Exponential Family 2**

Show that the Bernoulli distribution is a member of the exponential family


## Gaussian Mixture Models

Assume that we have a set of measurements $\{x^{(1)}, \dots x^{(m)}\}$. This is one of the few unsupervised learning examples in this lecture, thus, we do not know the true labels $y$. 

Gaussian Mixture Models (GMMs) assume that the data comes from a mixture of $K$ Gaussian distributions in the form

$$p(x) = \sum_{k=1}^K \pi_k \mathcal{N}(x|\mu_k, \Sigma_k),$$ (gmm_model)

with 

- $\pi = (\pi_1,...,\pi_k)$ called mixing coefficients, or cluster probabilities,
- $\mu = (\mu_1,...,\mu_k)$ the cluster means, and
- $\Sigma = (\Sigma_1,...,\Sigma_k)$ the cluster covariance matrices.

We define a K-dimensional r.v. $z$ which satisfies $z\in \{0,1\}$ and $\sum_k z_k=1$ (i.e. with only one of its dimensions being 1, while all others are 0), such that $z_k~\sim \text{Multinomial}(\pi_k)$ and $p(z_k=1) = \pi_k$ . For Eq. {eq}`gmm_model` to be a valid probability density, the parameters $\{\pi_k\}$ must satisfy $0\le\pi_k\le 1$ and $\sum_k \pi_k=1$.

The marginal distribution of $z$ can be equivalently written as 

$$p(z)=\prod_{k=1}^{K} \pi_k^{z_k},$$ (gmm_marginal_z)

whereas the conditional $p(x|z_k=1) = \mathcal{N}(x|\mu_k, \Sigma_k)$ becomes

$$p(x|z) = \prod_{k=1}^{K}\mathcal{N}(x|\mu_k, \Sigma_k)^{z_k}.$$ (gmm_conditional)

If we then express the distribution of interest $p(x)$ as the marginalized joint distribution, we obtain

$$
\begin{aligned}
p(x) &= \sum_z p(x,z) \\
& = \sum_z p(x|z) p(z) \\
& = \sum_{k=1}^K \pi_k\mathcal{N}(x| \mu_k, \Sigma_k)
\end{aligned}
$$ (gmm_marginalization)

Thus, the unknown parameters are $\{\pi_k, \mu_k, \Sigma_k\}_{k=1:K}$. We can write the log likelihood of the data as

$$
\begin{aligned}
l(x | \pi,\mu,\Sigma) &= \sum_{i=1}^{m}\log p(x^{(i)}|\pi,\mu,\Sigma) \\
&= \sum_{i=1}^{m}\log \left\{ \sum_{k=1}^K \pi_k \mathcal{N}(x^{(i)}|\mu_k,\Sigma_k) \right\}
\end{aligned}$$ (gmm_mle)

However, if we try to analytically solve this problem, we will see that there is no closed form solution. The problem is that we do not know which $z_k$ each of the measurements comes from.

### Expectation-Maximization

```{figure} ../imgs/cc1/em_algorithm.png
---
width: 600px
align: center
name: em_algorithm
---
EM algorithm for a GMM with $k=2$ (Source: [Bishop, 2006](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf), Section 9.2).
```

There is an iterative algorithms that can solve the maximum likelihood problem by alternating between two steps. The algorithm goes as follows:

0. Guess the number of modes $K$
1. Randomly initialize the means $\mu_k$, covariances $\Sigma_k$, and mixing coefficients $\pi_k$, and evaluate the likelihood
2. **(E-step)**. Evaluate $\omega_k^{(i)}$ assuming constant $\pi, \mu, \Sigma$ (see expression after the algorithm)
    $$w_k^{(i)} := p(z^{(i)}=k| x^{(i)}, \pi, \mu, \Sigma).$$ (gmm_e_step)
3. **(M-step)**. Update the parameters by solving the maximum likelihood probelms for fixed $z_k$ values.
    $$\begin{aligned}
    \pi_k &:= \frac{1}{m}\sum_{i=1}^m w_k^{(i)} \\
    \mu_k &:= \frac{\sum_{i=1}^{m} w_k^{(i)}x^{(i)}}{\sum_{i=1}^{m} w_k^{(i)}} \\
    \Sigma_k &:= \frac{\sum_{i=1}^{m} w_k^{(i)}(x^{(i)}-\mu_k)(x^{(i)}-\mu_k)^{\top}}{\sum_{i=1}^{m} w_k^{(i)}}
    \end{aligned}
    $$ (gmm_m_step)
4. Evaluate the log likelihood 
    $$l(x | \pi,\mu,\Sigma) = \sum_{i=1}^{m}\log \left\{ \sum_{k=1}^K \pi_k \mathcal{N}(x^{(i)}|\mu_k,\Sigma_k) \right\}$$ (gmm_lig_likelihood)
    and check for convergence. If not converged, return to step 2.

In the E-step, we compute the posterior probability of $z^{(i)}_k$ given the data point $x^{(i)}$ and the current $\pi$, $\mu$, $\Sigma$ values as 

$$
\begin{aligned}
p(z^{(i)}=k| x^{(i)},\pi,\mu,\Sigma) &= \frac{p(x^{(i)}|z^{(i)}=k, \mu, \Sigma)p(z^{(i)}=k,\pi)}{\sum_{l=1}^K p(x^{(i)}|z^{(i)}=l, \mu, \Sigma)p(z^{(i)}=l,\pi)} \\
 &= \frac{\pi_k \mathcal{N}(x^{(i)}|\mu_K, \Sigma_k)}{\sum_{l=1}^K \pi_l \mathcal{N}(x^{(i)}|\mu_l, \Sigma_l)}
\end{aligned}$$ (gmm_responsibilities)

The values of $p(x^{(i)}|z^{(i)}=k, \mu, \Sigma)$ can be computed by evaluating the $k$th Gaussian with parameters $\mu_k$ and $\Sigma_k$. And $p(z^{(i)}=k,\pi)$ is just $\pi_k$. 

**Exercise: derive the M-step update equations following the maximum likelihood approach.**

> Hint: look at [Bishop, 2006](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf), Section 9.2. 

### Applications and Limitations

Once we have fitted a GMM on $p(x)$, we can use it for:

1. Density estimation: by evaluate the probability $p(\tilde{x})$ of any new point $\tilde{x}$, we can say how probable it is that this point comes from the same distribution as the training data.
2. Clustering: so far we have talked about density estimation, but GMMs are typically used for clustering. Given a new query point $\tilde{x}$, we can evaluate each of the $K$ Gaussians and scale their probability by the respective $\pi_k$. These will be the probabilities of $\tilde{x}$ to be part of cluster $k$.

Most limitations of this approach arive from the assumption that the indivudual clusters follow the Gaussian distribution:

- If the data does not follow a Gaussian distribution, e.g. heavy-tailed ditribution with outliers, then too much weight will be given to the outliers
- If there is an outlier, eventually one mode will focus only on this one data point. But if a Gaussian describes only one data point, then its variance will be zero and we recover a singularity/Dirac function.
- The choice of $K$ is crucial and this parameters need to be optimized in a outer loop.
- GMMs do now scale well to high dimensions.


## Sampling Methods

The opposite process to density estimation is sampling. Here, we will look at the general case of an unnormalized ($\int_{\Omega}p(\cdot)\ne 1$) *target distribution*

$$p(y,\theta) = g(\theta)f(y|\theta).$$ (target_density)

The task we will be trying to solve here is how to generate samples from this distribution. How these samples are useful and why we choose this particular form of a target distribution will be the topic of next week's lecture.

### Acceptance-Rejection Sampling

Acceptance-rejection sampling draws its random samples directly from the target posterior distribution, as we only have access to the unscaled target distribution initially we will have to draw from the unscaled target. _The acceptance-rejection algorithm is specially made for this scenario._ The acceptance-rejection algorithm draws random samples from an easier-to-sample starting distribution and then successively reshapes its distribution by only selectively accepting candidate values into the final sample. For this approach to work the *candidate distribution* $g_{0}(\theta)$ has to dominate the posterior distribution, i.e. there must exist an $M$ s.t.

$$M \times g_{0}(\theta) \geq g(\theta) f(y|\theta), \quad \forall \theta$$ (acceptance_rejection)

Taking an example candidate density for an unscaled target as an example to show the "dominance" of the candidate distribution over the posterior distribution.

```{figure} ../imgs/cc1/acceptance_rejection.png
---
width: 450px
align: center
name: acceptance_rejection
---
Acceptance-rejection candidate and target distributions (Source: Bolstad, 2009).
```

To then apply acceptance-rejection sampling to the posterior distribution we can write out the algorithm as follows:

1. Draw a random sample of size $N$ from the candidate distribution $g_{0}(\theta)$.
2. Calculate the value of the unscaled target density at each random sample.
3. Calculate the candidate density at each random sample, and multiply by $M$.
4. Compute the weights for each random sample

$$ w_{i} = \frac{g(\theta_{i}) \times f(y_{1}, \ldots, y_{n}| \theta_{i})}{M \times g_{0}(\theta_{i})}$$ (acceptance_rejection_weights)

5. Draw $N$ samples from the $U(0, 1)$ uniform distribution.
6. If $u_{i} < w_{i}$ accept $\theta_{i}$


### Sampling-Importance-Resampling / Bayesian Bootstrap

The sampling-importance-resampling algorithm is a two-stage extension of the acceptance-rejection sampling which has an improved weight-calculation, but most importantly employs a _resampling_ step. This resampling step resamples from the space of parameters. The weight is then calculated as

$$w_{i} = \frac{\frac{g(\theta_{i})f(y_{1}, \ldots\ y_{n} | \theta_{i})}{g_{0}(\theta_{i})}}{\left( \sum_{i=1}^{N} \frac{g(\theta_{i})f(y_{1}, \ldots, y_{n}| \theta_{i})}{g_{0}(\theta_{i})} \right)} $$ (sampling_importance_resampling_weights)

The algorithm to sample from the posterior distribution is then:

1. Draw $N$ random samples $\theta_{i}$ from the starting density $g_{0}(\theta)$
2. Calculate the value of the unscaled target density at each random sample.
3. Calculate the starting density at each random sample, and multiply by $M$.
4. Calculate the ratio of the unscaled posterior to the starting distribution

$$r_{i} = \frac{g(\theta_{i})f(y_{1}, \ldots, y_{n}| \theta_{i})}{g_{0}(\theta_{i})}$$ (sampling_importance_resampling_weights_r)

5. Calculate the importance weights

$$w_{i} = \frac{r_{i}}{\sum r_{i}}$$ (sampling_importance_resampling_weights_w)

6. Draw $n \leq 0.1 \times N$ random samples with the sampling probabilities given by the importance weights. 


### Adaptive Rejection Sampling

If we are unable to find a candidate/starting distribution, which dominates the unscaled posterior distribution immediately, then we have to rely on _adaptive rejection sampling_.

> This approach only works for a log-concave posterior!

See below for an example of a log-concave distribution.

```{figure} ../imgs/cc1/adaptive_rejection_sampling.png
---
width: 450px
align: center
name: adaptive_rejection_sapling
---
(Not) log-concave function (Source: Bolstad, 2009).
```

Using the tangent method, our algorithm then takes the following form:

1. Construct an upper bound from piecewise exponential functions, which dominate the log-concave unscaled posterior
2. With the envelope giving us the initial candidate density we draw $N$ random samples
3. Rejection sampling, see the preceding two subsections for details.
4. If rejected, add another exponential piece which is tangent to the target density.

> As all three presented sampling approaches have their limitations, practitioners often rely more on Markov chain Monte Carlo methods such as Gibbs sampling, and Metropolis-Hastings.

### Markov Chain Monte Carlo

The idea of Markov Chain Monte Carlo (MCMC) is to construct an ergodic Markov chain of samples $\{\theta^0, \theta^1, ...,\theta^N\}$ distributed according to the posterior distribution $g(\theta|y) \propto g(\theta)f(y|\theta)$. This chain evolves according to a transition kernel given by $q(x_{next}|x_{current})$. Let's look at one of the most popular MCMC algorithms: Metropolis Hastings

**Metropolis-Hastings**

The general Metropolis-Hastings prescribes a rule which guarantees that the constructed chain is representative of the target distribution $g(\theta|y)$. This is done by following the algorithm:

0. Start at an initial point $\theta_{current} = \theta^0$.
1. Sample $\theta' \sim q(\theta_{next}|\theta_{current})$
2. Compute the ecceptance probability
    $$\alpha = min \left\{ 1, \frac{g(\theta'|y) q(\theta_{current}|\theta')}{g(\theta_{current}|y) q(\theta'|\theta_{current})} \right\}$$
3. Sample $u\sim \text{Uniform}(0,1)$
4. If $\alpha > u$, then $\theta_{current} = \theta'$, else $\theta_{current} = \theta_{current}$
5. Repeat $N$ times from step 1.

A special choice of $q(\cdot | \cdot)$ is for example the normal distribution $\mathcal{N}(\cdot | \theta_{current}, \sigma^2)$, which results in the so-called Random Walk Metropolis algorithm. Other special cases include the Metropolis-Adjusted Langevin Algorithm (MALA), as well as the Hamiltonian Monte Carlo (HMC) algorithm.

```{figure} ../imgs/cc1/metropolis_hastings.png
---
width: 450px
align: center
name: metropolis_hastings
---
Metropolis-Hastings trajectory (Source: [relguzman.blogpost.com](https://relguzman.blogspot.com/2018/04/sampling-metropolis-hastings.html)).
```

--- 

 In summary:

- The unscaled posterior $g(\theta|y) \propto g(\theta)f(y|\theta)$ contains the *shape information* of the posterior
- For the true posterior, the unscaled posterior needs to be divided by an integral over the whole parameter space.
- Integral has to be evaluated numerically for which we rely on the just presented Monte Carlo sampling techniques.

## Further References

**Probability Theory**

- [CS229 Lecture notes](https://sgfin.github.io/files/notes/CS229_Lecture_Notes.pdf), Andrew Ng, Parts III, "The exponential family"


**Gaussian Mixture Models**

- [CS229 Lecture notes](https://sgfin.github.io/files/notes/CS229_Lecture_Notes.pdf), Andrew Ng, Parts VIII, "Mixtures of Gaussians and the EM algorithm"
- [Pattern Recognition and Machine Learning](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf), Bishop, 2006,
Section 9.2

**Markov Chain Monte Carlo**

- [Interactive MCMC visualizations](https://chi-feng.github.io/mcmc-demo/app.html?algorithm=RandomWalkMH&target=banana)
- [Monte Carlo Statistical Methods](https://link.springer.com/book/10.1007/978-1-4757-4145-2), Rober & Casella, 2004

