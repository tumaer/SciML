# GMM and MC

This lesson first recaps on Probability Theory and then introduces Gaussian Mixture Models (GMM) and Markov chain Monta Carlo (MCMC).


## Probability Theory

### Basic Building Blocks
- $\Omega$ - sample space: the set of all outcomes of a random experiment.
- $\mathbb{P}(E)$ - probability measure of an event $E \in \Omega$: a function $\mathbb{P}: \Omega \rightarrow \mathbb{R}$ which satisfies the following three properties:
    - $0 \le \mathbb{P}(E) \le 1 \quad \forall E \in \Omega$
    - $\mathbb{P}(\Omega)=1$
    - $\mathbb{P}(\cup_{i=1}^n E_i) = \sum_{i=1}^n \mathbb{P}(E_i) \;$ for disjoint events ${E_1, ..., E_n}$
- $\mathbb{P}(A, B)$ - joint probability: probability that both $A$ and $B$ occur simultaneously.
- $\mathbb{P}(A | B)$ - conditional probability: probability that $A$ occurs, if $B$ has occured.
- Product rule of probabilities:
    - general case: <br>

    $$\mathbb{P}(A, B) = \mathbb{P}(A | B)\cdot  \mathbb{P}(B) = \mathbb{P}(B | A) \cdot \mathbb{P}(A)$$

    - independent events: <br>

    $$\mathbb{P}(A, B) = \mathbb{P}(A) \cdot \mathbb{P}(B)$$

- Sum rule of probabilities: 

$$\mathbb{P}(A)=\sum_{B}\mathbb{P}(A, B)$$

- Bayes rule: solving the general case of the product rule for $\mathbb{P}(A)$ results in:

    $$ \mathbb{P}(B|A) = \frac{\mathbb{P}(A|B) \mathbb{P}(B)}{\mathbb{P}(A)} = \frac{\mathbb{P}(A|B) \mathbb{P}(B)}{\sum_{i=1}^n \mathbb{P}(A|B_i)\mathbb{P}(B_i)}$$

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

### Exponential Family

The exponential family of distributions is a large family of distributions with shared properties, some of which we have already encountered in other courses before. Prominent members of the exponential family include

- Bernoulli
- Gaussian
- Dirichlet
- Gamma
- Poisson
- Beta

At their core, members of the exponential family all fit the same general probability distribution form

$$p(x|\eta) = h(x) \exp \left\{ \eta^{\top} t(x) - a(\eta) \right\}$$

where the individual components are

- Natural parameter $\eta$
- Sufficient statistic $t(x)$
- Probability support measure $h(x)$
- Log normalizer $a(\eta)$ to guarantee that the probability density integrates to 1.

> If you are unfamiliar with the concept of probability measures, then $h(x)$ can safely be disregarded. Conceptually it describes the area in the probability space over which the probability distribution is defined.

**Why is this family of distributions relevant to this course?** 

> The exponential family has a direct connection to graphical models, which are a formalism favored by many people to visualize machine learning models, and the way individual components interact with each other. As such they are highly instructive, and at the same time foundational to many probabilistic approaches covered in this course.

Let's inspect the practical example of the Gaussian distribution to see how the theory translates into practice. Taking the probability density function which we have also previously worked with

$$p(x|\mu, \sigma^{2}) = \frac{1}{\sqrt{2 \pi \sigma^{2}}} \exp \left\{ \frac{(x - \mu)^{2}}{2 \sigma^{2}} \right\}$$

We can then expand the square in the exponent of the Gaussian to isolate the individual components of the exponential family

$$p(x|\mu, \sigma^{2}) = \frac{1}{\sqrt{2 \pi \sigma^{2}}} \exp \left\{ \frac{\mu}{\sigma^{2}}x - \frac{1}{2 \sigma^{2}}x^{2} - \frac{1}{2 \sigma^{2}} \mu^{2} - \log \sigma \right\}$$

Then the individual components of the Gaussian are

$$\eta = \langle \frac{\mu}{\sigma^{2}}, - \frac{1}{2 \sigma^{2}} \rangle$$
$$t(x) = \langle x, x^{2} \rangle$$
$$a(\eta) = \frac{\mu^{2}}{2 \sigma^{2}} + \log \sigma$$
$$h(x) = \frac{1}{\sqrt{2 \pi}}$$

For the sufficient statistics, we then need to derive the derivative of the log normalizer, i.e 

$$\frac{d}{d\eta}a(\eta) = \mathbb{E}\left[ t(X) \right]$$

Which yields

$$\frac{da(\eta)}{d\eta_{1}} = \mu = \mathbb{E}[X] $$
$$\frac{da(\eta)}{d\eta_{2}} = \mu = \mathbb{E}[X^{2}] $$

**Exercise: Exponential Family 1**

Show that the Dirichlet distribution is a member of the exponential family.


**Exercise: Exponential Family 2**

Show that the Bernoulli distribution is a member of the exponential family


## GMM


## MCMC

### Monte Carlo Sampling

To approximate this quantity with Monte Carlo sampling techniques we then need to draw from the posterior $\int f(y|\theta) g(\theta) d\theta$. A process which given enough samples always converges to the true value of the denominator according to the [Monte Carlo theorem](http://www-star.st-and.ac.uk/~kw25/teaching/mcrt/MC_history_3.pdf). 

Monte Carlo integration is a fundamental tool first developed by Physicists dealing with the solution of high-dimensional integrals. The main objective is solving integrals like

$$E[h(\mathbf{x})]=\int h(\mathbf{x}) p(\mathbf{x}) d\mathbf{x}, \quad \mathbf{x}\in \mathbb{R}^d$$

with some function of interest $h$ and $\mathbf{x}$ being a r.v.

The approach consists of the following three steps:
1. Generate i.i.d. random samples $\mathbf{x}^{(i)}\in \mathbb{R}^d, \; i=1,2,...,N$ from the density $p(\mathbf{x})$.
2. Evaluate $h^{(i)}=h(\mathbf{x}^{(i)}), \; \forall i$.
3. Approximate

$$E[h(\mathbf{x})]\approx \frac{1}{N}\sum_{i=1}^{N}h^{(i)}$$

---

Bayesian approaches based on random Monte Carlo sampling from the posterior have a number of advantages for us:

- Given a large enough number of samples, we are not working with an approximation, but with an estimate which can be made as precise as desired (given the requisite computational budget)
- Sensitivity analysis of the model becomes easier.

#### Acceptance-Rejection Sampling

Acceptance-rejection sampling draws its random samples directly from the target posterior distribution, as we only have access to the unscaled target distribution initially we will have to draw from the unscaled target. _The acceptance-rejection algorithm is specially made for this scenario._ The acceptance-rejection algorithm draws random samples from an easier-to-sample starting distribution and then successively reshapes its distribution by only selectively accepting candidate values into the final sample. For this approach to work the candidate distribution $g_{0}(\theta)$ has to dominate the posterior distribution, i.e. there must exist an $M$ s.t.

$$M \times g_{0}(\theta) \geq g(\theta) f(y|\theta), \quad \forall \theta$$

Taking an example candidate density for an unscaled target as an example to show the "dominance" of the candidate distribution over the posterior distribution.

<center>
<img src = "https://i.imgur.com/1xzbFI5.png" width = "450"></center>

(Source, Bolstad _Understanding Computational Bayesian Statistics_)

To then apply acceptance-rejection sampling to the posterior distribution we can write out the algorithm as follows:

1. Draw a random sample of size $N$ from the candidate distribution $g_{0}(\theta)$.
2. Calculate the value of the unscaled target density at each random sample.
3. Calculate the candidate density at each random sample, and multiply by $M$.
4. Compute the weights for each random sample

$$ w_{i} = \frac{g(\theta_{i}) \times f(y_{1}, \ldots, y_{n}| \theta_{i})}{M \times g_{0}(\theta_{i})}$$

5. Draw $N$ samples from the $U(0, 1)$ uniform distribution.
6. If $u_{i} < w_{i}$ accept $\theta_{i}$


#### Sampling-Importance-Resampling / Bayesian Bootstrap

The sampling-importance-resampling algorithm is a two-stage extension of the acceptance-rejection sampling which has an improved weight-calculation, but most importantly employs a _resampling_ step. This resampling step resamples from the space of parameters. The weight is then calculated as

$$w_{i} = \frac{\frac{g(\theta_{i})f(y_{1}, \ldots\ y_{n} | \theta_{i})}{g_{0}(\theta_{i})}}{\left( \sum_{i=1}^{N} \frac{g(\theta_{i})f(y_{1}, \ldots, y_{n}| \theta_{i})}{g_{0}(\theta_{i})} \right)} $$

The algorithm to sample from the posterior distribution is then:

1. Draw $N$ random samples $\theta_{i}$ from the starting density $g_{0}(\theta)$
2. Calculate the value of the unscaled target density at each random sample.
3. Calculate the starting density at each random sample, and multiply by $M$.
4. Calculate the ratio of the unscaled posterior to the starting distribution

$$r_{i} = \frac{g(\theta_{i})f(y_{1}, \ldots, y_{n}| \theta_{i})}{g_{0}(\theta_{i})}$$

5. Calculate the importance weights

$$w_{i} = \frac{r_{i}}{\sum r_{i}}$$

6. Draw $n \leq 0.1 \times N$ random samples with the sampling probabilities given by the importance weights. 


#### Adaptive Rejection Sampling

If we are unable to find a candidate/starting distribution, which dominates the unscaled posterior distribution immediately, then we have to rely on _adaptive rejection sampling_.

> This approach only works for a log-concave posterior!

See below for an example of a log-concave distribution.

<center>
<img src = "https://i.imgur.com/8j4zCVo.png" width = "550"></center>

(Source: Bolstad, _Understanding Computational Bayesian Statistics_)

Using the tangent method our algorithm then takes the following form:

1. Construct an upper bound from piecewise exponential functions, which dominate the log-concave unscaled posterior
2. With the envelope giving us the initial candidate density we draw $N$ random samples
3. Rejection sampling, see the preceding two subsections for details.
4. If rejected, add another exponential piece which is tangent to the target density.

As all three presented sampling approaches have their limitations, practitioners tend to rely more on Markov chain Monte Carlo methods such as Gibbs sampling, and Metropolis-Hastings.
#### Markov Chain Monte Carlo

The idea of Markov Chain Monte Carlo (MCMC) is to construct an ergodic Markov chain of samples $\{\theta^0, \theta^1, ...,\theta^N\}$ distributed according to the posterior distribution $g(\theta|y)$. This chain evolves according to a transition kernel given by $q(x_{next}|x_{current})$. Let's look at one of the most popular MCMC algorithms: Metropolis Hastings

**Metropolis-Hastings**

The general Metropolis-Hastings prescribes a rule which guarantees that the constructed chain is representative of the target distribution $g(\theta|y)$. This is done by following the algorithm:

0. Start at an initial point $\theta_{current} = \theta^0$.
1. Sample $\theta' \sim q(\theta_{next}|\theta_{current})$
2. Compute 
    $$\alpha = min \left\{ 1, \frac{g(\theta'|y) q(\theta_{current}|\theta')}{g(\theta_{current}|y) q(\theta'|\theta_{current})} \right\}$$
3. Sample $u\sim \text{Uniform}(0,1)$
4. If $\alpha > u$, then $\theta_{current} = \theta'$, else $\theta_{current} = \theta_{current}$
5. Repeat $N$ times from step 1.

A special choice of $q(\cdot | \cdot)$ is for example the normal distribution $\mathcal{N}(\cdot | \theta_{current}, \sigma^2)$, which results in the so-called Random Walk Metropolis algorithm. Other special cases include the Metropolis-Adjusted Langevin Algorithm (MALA), as well as the Hamiltonian Monte Carlo (HMC) algorithm. For more information, refer to [Monte Carlo Statistical Methods](https://link.springer.com/book/10.1007/978-1-4757-4145-2) by Rober & Casella.


--- 

 In summary:

- The unscaled posterior $g(\theta|y) \propto g(\theta)f(y|\theta)$ contains the *shape information* of the posterior
- For the true posterior, the unscaled posterior needs to be divided by an integral over the whole parameter space.
- Integral has to be evaluated numerically for which we rely on the just presented Monte Carlo sampling techniques.

## Further References
