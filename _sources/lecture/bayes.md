# Bayesian methods

We start by introducing Bayesian statistics and the closely related sampling methods, e.g. Markov Chain Monte Carlo (MCMC). We then present Bayesian inference and its applications to regression and classification. 

## Bayesian Statistics

The main ideals upon which Bayesian statistics is founded are:

- Uncertainty over parameters $\rightarrow$ treatment as random variables
- A probability over a parameter essentially expresses a degree of belief
- Inference over parameters using rules of probability
- We combine the prior knowledge and the observed data with Bayes' theorem

> Refresher - Bayes' theorem is given by $\mathbb{P}(A|B) = \frac{\mathbb{P}(B|A)\mathbb{P}(A)}{\mathbb{P}(B)}$

So what we are interested in is the **posterior** distribution over our parameters, which can be found using Bayes' theorem. While this may at first glance look straightforward, and holds for the **unscaled posterior** i.e the distribution which has not been normalized by dividing by $\mathbb{P}(B)$, obtaining the scaled posterior is much much harder due to the difficulty in computing the divisor $\mathbb{P}(B)$. To evaluate this divisor we often rely on Monte Carlo sampling.

> What is a _scaled posterior_? A scaled posterior is a distribution whose integral over the entire distribution evaluates to 1. 

Taking Bayes' theorem and using the probability theorems in their conditional form, we then obtain the following formula for the posterior density

$$g(\theta | y) = \frac{g(\theta) \times f(y | \theta)}{\int g(\theta) \times f(y | \theta) d\theta},$$ (posterior_density)

with data $y$, parameters as random variable $\theta$, *prior* $g(\theta)$, *likelihood* $f(y|\theta)$, and *posterior* $g(\theta|y)$. If we now seek to compute the denominator (aka *evidence*), then we have to compute the integral

$$\int f(y|\theta) g(\theta) d\theta.$$ (evidence)

This integration may be very difficult, and is in most practical cases infeasible.

### Monte Carlo Integration

Luckily, we do not necessarily need to compute the denominator. Computational Bayesian statistics relies on Monte Carlo samples from the posterior, which **does not require knowing the denominator** of the posterior.

What we actually care about is making predictions about the output of a model $h_{\theta}(x)$, whose parameters $\theta$ are random variables. Without loss of generality, we can rewrite $h_{\theta}(x)$ to $h_x(\theta)$ as a function of the random variable $\theta$ evaluated at a particular $x$. Then the problem can be formulated as the expectation

$$E[h_x(\theta|y)]=\int h_{x}(\theta) g(\theta|y) d\theta.$$ (mc_expectation)

To approximate this integral with Monte Carlo sampling techniques, we then need to draw samples from the posterior $g(\theta|y)$. A process which given enough samples always converges to the true value of the denominator according to the [Monte Carlo theorem](http://www-star.st-and.ac.uk/~kw25/teaching/mcrt/MC_history_3.pdf). 

The approach consists of the following three steps:
1. Generate i.i.d. random samples $\theta^{(i)}, \; i=1,2,...,N$ from the posterior $g(\theta|y)$.
2. Evaluate $h_x^{(i)}=h_x(\theta^{(i)}), \; \forall i$.
3. Approximate

$$E[h_x(\theta|y)]\approx \frac{1}{N}\sum_{i=1}^{N}h_x^{(i)}.$$ (mc_sum)

Bayesian approaches based on random Monte Carlo sampling from the posterior have a number of advantages for us:

- Given a large enough number of samples, we are not working with an approximation, but with an estimate which can be made as precise as desired (given the requisite computational budget)
- Sensitivity analysis of the model becomes easier.

*The only question left is how to sample from $g(\theta|y)$?*

## Sampling Methods

Looking at the denominator of Eq. {eq}`posterior_density`, we notice that it is independent of $\theta$, and is thus only a scaling factor from the perspective of the posterior $g(\theta|y)$. This means that the nominator contains all the information describing the shape of the posterior. Thus, if we can generate sufficiently many samples from the unnormalized posterior (i.e. nominator), then these will have the exact same distribution as the true posterior. That is why you will often see the posterior written as "proportional to the prior times lihelihood"

$$g(\theta | y) \propto g(\theta)f(y|\theta).$$ (target_density)

The problem we will be trying to solve here is how to generate samples from such unnormalized distributions. We use the term *target* distributin to describe the distribution we want to sample from.

### Acceptance-Rejection Sampling

Acceptance-rejection sampling draws its random samples directly from the target posterior distribution. As we only have access to the unscaled target distribution, we will have to draw from it. _The acceptance-rejection algorithm is specially made for this scenario._ The acceptance-rejection algorithm draws random samples from an easier-to-sample starting distribution and then selectively accepts candidate values into the final sample. For this approach to work, the *candidate distribution* $g_{0}(\theta)$ has to dominate the posterior distribution, i.e. there must exist an $M$ s.t.

$$M \times g_{0}(\theta) \geq g(\theta) f(y|\theta), \quad \forall \theta.$$ (acceptance_rejection)

{numref}`acceptance_rejection` shows a candidate density for an unscaled target. It would take $M\approx 5$ to reach a "dominance" of the candidate distribution over the target distribution.

```{figure} ../imgs/acceptance_rejection.png
---
width: 450px
align: center
name: acceptance_rejection
---
Acceptance-rejection candidate and target distributions (Source: {cite}`bolstad2009`).
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

> This approach only works for a log-concave posterior! Log-concave means that the second derivative of the log of a log-concave density is always non-positive.

See below for an example of a log-concave distribution.

```{figure} ../imgs/adaptive_rejection_sampling.png
---
width: 450px
align: center
name: adaptive_rejection_sapling
---
(Not) log-concave function (Source: {cite}`bolstad2009`).
```

Using the tangent method, our algorithm then takes the following form:

1. Construct an upper bound from piecewise exponential functions, which dominate the log-concave unscaled posterior
2. With the envelope giving us the initial candidate density we draw $N$ random samples
3. Rejection sampling, see the preceding two subsections for details.
4. If rejected, add another exponential piece which is tangent to the target density.

> As all three presented sampling approaches have their limitations, practitioners often rely on Markov chain Monte Carlo methods such as Gibbs sampling, and Metropolis-Hastings.

### Markov Chain Monte Carlo

The idea of Markov Chain Monte Carlo (MCMC) is to construct an ergodic Markov chain of samples $\{\theta^0, \theta^1, ...,\theta^N\}$ distributed according to the posterior distribution $g(\theta|y) \propto g(\theta)f(y|\theta)$. This chain evolves according to a transition kernel given by $q(\theta_{next}|\theta_{current})$. Let's look at one of the most popular MCMC algorithms: Metropolis Hastings

**Metropolis-Hastings**

The general Metropolis-Hastings prescribes a rule which guarantees that the constructed chain is representative of the target distribution $g(\theta|y)$. This is done by following the algorithm:

0. Start at an initial point $\theta_{current} = \theta^0$.
1. Sample $\theta' \sim q(\theta_{next}|\theta_{current})$
2. Compute the acceptance probability

    $$\alpha = min \left\{ 1, \frac{g(\theta'|y) q(\theta_{current}|\theta')}{g(\theta_{current}|y) q(\theta'|\theta_{current})} \right\}$$ (mcmc_acceptance_prob)

3. Sample $u\sim \text{Uniform}(0,1)$
4. If $\alpha > u$, then $\theta_{current} = \theta'$, else $\theta_{current} = \theta_{current}$
5. Repeat $N$ times from step 1.

A special choice of $q(\cdot | \cdot)$ is for example the normal distribution $\mathcal{N}(\cdot | \theta_{current}, \sigma^2)$, which results in the so-called Random Walk Metropolis algorithm. Other special cases include the Metropolis-Adjusted Langevin Algorithm (MALA), as well as the Hamiltonian Monte Carlo (HMC) algorithm.

```{figure} ../imgs/metropolis_hastings.png
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


## Bayesian Inference

In the Bayesian framework, everything centers around the posterior distribution and our ability to relate our previous knowledge with newly gained evidence to the next stage of our belief (of a probability distribution). With the posterior being our entire inference about the parameters given the data, there exist multiple inference approaches with their roots in frequentist statistics.

### Bayesian Point Estimation

Bayesian point estimation chooses a single value to represent the entire posterior distribution. Potential choices here are locations like the posterior mean, and posterior median. The **posterior mean** $\hat{\theta}$ minimizes the *posterior mean squared error*

$$PMS(\hat{\theta}) = \int (\theta - \hat{\theta})^{2} g(\theta | y_{1}, \ldots, y_{n})d\theta,$$ (mps_error)

$$\hat{\theta} = \int_{-\infty}^{\infty} \theta g(\theta | y_{1}, \ldots, y_{n})d \theta \qquad \text{(posterior mean)}.$$ (posterior_mean)

```{figure} ../imgs/posterior_mean.png
---
width: 450px
align: center
name: posterior_mean
---
Posterior mean (Source: {cite}`bolstad2009`, Chapter 3).
```

And the **posterior median** $\tilde{\theta}$ minimizes the *posterior mean absolute deviation*

$$PMAD(\hat{\theta}) = \int |\theta - \hat{\theta}| g(\theta| y_{1}, \ldots, y_{n})d\theta,$$ (pmad)

$$.5 = \int_{-\infty}^{\tilde{\theta}} g(\theta | y_{1}, \ldots, y_{n}) d\theta \qquad \text{(posterior median)}.$$ (posterior_median)


```{figure} ../imgs/posterior_median.png
---
width: 450px
align: center
name: posterior_median
---
Posterior median (Source: {cite}`bolstad2009`, Chapter 3).
```

Another point estimate is given by the **maximum a posteriory** (MAP) estimate $\theta_{MAP}$, which is a generalization over the maximum likelihood estimate (MLE) in the sense that MAP entails a prior $g(\theta)$.

$$\theta_{MLE} = \underset{\theta}{\arg \max}\;  \prod_{i=1}^N f(y^{(i)}|\theta). \qquad$$ (mle_estimate)

$$\theta_{MAP} = \underset{\theta}{\arg \max}\;  \prod_{i=1}^N f(y^{(i)}|\theta) g(\theta).$$ (map_estimate)

If the prior $g(\theta)$ is uniform (aka *uninformative prior*), then the MLE and MAP estimates coincide. But as soon as we have some prior knowledge about the problem, the prior regularizes the maximization problem. More on regularization in lecture [](./tricks.md).

### Bayesian Interval Estimation

Another type of Bayesian inference is the one in which we seek to find an interval that, with a pre-determined probability, contains the true value. In Bayesian statistics, these are called credible intervals. Finding the interval with equal tail areas to both sides $(\theta_{l}, \theta_{u})$, and which has the probability to contain the true value of the parameter, i.e.

$$
\begin{aligned}
\int_{-\infty}^{\theta_{l}} g(\theta | y_{1}, \ldots, y_{n}) d\theta &= \frac{\alpha}{2}, \\
\int_{\theta_{u}}^{\infty} g(\theta | y_{1}, \ldots, y_{n}) d\theta &= \frac{\alpha}{2},
\end{aligned}
$$ (confidence_intervals)

which we then only need to solve. A visual example of such a scenario is in the following picture:

```{figure} ../imgs/credible_interval.png
---
width: 450px
align: center
name: credible_intervals
---
The 95\% credible interval (Source: {cite}`bolstad2009`, Chapter 3).
```

### Predictive Distribution of a New Observation

If we obtain a new observation, then we can compute the updated predictive distribution by combining the conditional distribution of the new observation, and conditioning it on the previous observations. Then we only need to integrate the parameter out of the joint posterior

$$
\begin{aligned}
f(y_{n+1}|y_{1}, \ldots, y_{n}) &\propto \int g(\theta) \times f(y_{n+1}| \theta) \times \ldots \times f(y_{1}|\theta) d\theta \\
&\propto \int f(y_{n+1}|\theta)g(\theta|y_{1}, \ldots, y_{n}) d\theta
\end{aligned}$$ (predictive_posterior_verbose)

and marginalize it out.


### Bayesian Inference from a Posterior Random Sample

When we only have a random sample from the posterior instead of a numerical approximation of the posterior, we are still able to apply the same techniques (i.e. point and interval estimates), but just apply them to the posterior sample.

The generated sample only constitute an approximation, but given the sampling budget, this approximation can be made as accurate as desired. In summary, Bayesian inference can be condensed to the following main take-home knowledge:

- The posterior distribution is the current summary of beliefs about the parameter in the Bayesian framework.
- Bayesian inference is then performed using the probabilities calculated from the posterior distribution of the parameter.
    - To get an approximation of the scaling factor for the posterior we have to utilize sampling-based Monte Carlo techniques to approximate the requisite integral.


## Bayesian Approaches to Regression

If we are faced with the scenario of having very little data, then we ideally seek to quantify the uncertainty of our model and preserve the predictive utility of our machine learning model. The right approach to this is to extend Linear Regression, and Logistic Regression with the just presented Bayesian Approach utilizing Bayesian Inference.

### Bayesian Logistic Regression

If we now want to capture the uncertainty over our logistic regression predictions, then we have to resort to the Bayesian approach. To make the Bayesian approach work for logistic regression, we choose to apply something called the _Laplace Approximation_ in which we approximate the posterior using a Gaussian

$$p(\omega | \mathcal{D}) \approx \mathcal{N}({\bf{\omega}}| {\bf{\hat{\omega}}}, {\bf{H}}^{-1}),$$ (laplace_approx)

where $\omega$ corresponds to the learned parameters $\theta$, $\hat{\omega}$ is the MAP estimate of $\theta$, and $H^{-1}$ is the inverse of the Hessian computed at $\hat{\omega}$. There exist many different modes representing viable solutions for this problem when we seek to optimize it.


```{figure} ../imgs/bayesian_log_reg_data.png
---
width: 450px
align: center
name: bayesian_log_reg_data
---
Bayesian logistic regression data (Source: {cite}`murphy2022`, Chapter 10).
```

In practical applications, we are interested in predicting the output $y$ given an input $x$. Thus, we need to compute the **posterior predictive distribution**

$$p(y|x, \mathcal{D}) = \int p(y | x, \omega) p(\omega | \mathcal{D})) d\omega.$$ (predictive_posterior)

To now compute the uncertainty in our predictions we perform a _Monte Carlo Approximation_ of the integral using $S$ samples from the posterior $\omega_s \sim p(\omega|\mathcal{D})$ as

$$p(y=1 | x, \mathcal{D}) = \frac{1}{S} \sum_{s=1}^{S} \text{sigmoid} \left( \omega_{s}^{\top} x \right).$$ (blr_mc)

Looking at a larger visual example of Bayesian Logistic Regression applied.

```{figure} ../imgs/bayesian_log_reg.jpg
---
width: 600px
align: center
name: bayesian_log_reg
---
Bayesian logistic regression (Source: {cite}`murphy2022`, Chapter 10).
```

### Bayesian Linear Regression

To now introduce the Bayesian approach to linear regression we have to assume that we already know the variance $\sigma^{2}$, so the posterior which we actually want to compute at that point is

$$p(\omega | \mathcal{D}, \sigma^{2}),$$ (blr_formulation)

where $\mathcal{D} = \left\{ (x_{n}, y_{n}) \right\}_{n=1:N}$. For simplicity we assume a Gaussian prior distribution

$$p(\omega) = \mathcal{N}(\omega | \breve{\omega}, \breve{\Sigma}),$$ (blr_normal_prior)

and a likelihood given by the Multivariate-Normal distribution

$$p(\mathcal{D} | \omega, \sigma^{2}) = \prod_{n=1}^{N}p(y_{n}|{\bf{\omega^{\top}}}{\bf{x}}, \sigma^{2}) = \mathcal{N}({\bf{y}} | {\bf{X} \bf{\omega}}, \sigma^{2} {\bf{I}}_{N}).$$ (blr_likelihood)

The posterior can then be analytically derived using Bayes' rule for Gaussians (see {cite}`murphy2022`, Eq. 3.37) to

$$
\begin{aligned}
p({\bf{\omega}} | {\bf{X}}, {\bf{y}}, \sigma^{2}) &\propto \mathcal{N}(\omega | \breve{\omega}, \breve{\Sigma}) \mathcal{N}({\bf{y}} | {\bf{X} \bf{\omega}}, \sigma^{2} {\bf{I}}_{N}) = \mathcal{N}({\bf{\omega}} | {\bf{\hat{\omega}}}, {\bf{\hat{\Sigma}}}) \\
\text{with } \quad {\bf{\hat{\omega}}} &\equiv {\bf{\hat{\Sigma}}} \left( {\bf{\breve{\Sigma}}}^{-1} {\bf{\breve{\omega}}} + \frac{1}{\sigma^{2}} {\bf{X^{\top} y}}  \right), \\
{\bf{\hat{\Sigma}}} &\equiv \left( {\bf{\breve{\Sigma}}}^{-1} + \frac{1}{\sigma^{2}} {\bf{X^{\top} X}} \right)^{-1},
\end{aligned}
$$ (blr_posterior)

where $\hat{\omega}$ is the posterior mean, and $\hat{\Sigma}$ is the posterior covariance. A good visual example of this is the sequential Bayesian inference on a linear regression model shown below.

```{figure} ../imgs/bayesian_lin_reg.png
---
width: 550px
align: center
name: bayesian_lin_reg
---
Bayesian linear regression (Source: {cite}`murphy2022`, Chapter 11).
```

## Bayesian Machine Learning

Let's consider the setup we have encountered so far in which we have labels $x$, hyperparameters $\theta$, and seek to predict labels $y$. Probabilistically expressed this amounts to $p(y|x, \theta)$. Then the posterior is defined as $p(\theta| \mathcal{D})$, where $\mathcal{D}$ is our labeled dataset $\mathcal{D} = \left\{ (x_{n}, y_{n}) \right\}_{n=1:N}$. 
Applying the previously discussed Bayesian approaches to these problems, and the respective model parameters, are called **Bayesian Machine Learning**.

While we lose computational efficiency at first glance, as we have to perform a sampling-based inference procedure, what we gain is a principled approach to discuss uncertainties within our model. This can help us most especially when we move in the *small-data limit*, where we can not realistically expect our model to converge. See e.g. below a Bayesian logistic regression example in which the posterior distribution is visualized.

```{figure} ../imgs/bayesian_nn.png
---
width: 550px
align: center
name: bayesian_nn
---
Bayesian machine learning (Source: {cite}`murphy2022`, Chapter 4).
```

## Further References

**Bayesian Methods**

There exist a wide number of references to the herein presented Bayesian approach, most famously introductory treatment of Probabilistic Programming frameworks, which utilize the herein presented modeling approach to obtain posteriors over programs.

- [Introduction to Pyro](http://pyro.ai/examples/intro_long.html)
- [A Practical Example with Stan](https://m-clark.github.io/bayesian-basics/example.html#posterior-predictive)

In addition, there exists highly curated didactic material from Michael Betancourt:

- [Sampling](https://betanalpha.github.io/assets/case_studies/sampling.html): Section 3, 4, and 5
- [Towards a Principled Bayesian Workflow](https://betanalpha.github.io/assets/case_studies/principled_bayesian_workflow.html)
- [Markov Chain Monte Carlo](https://betanalpha.github.io/assets/case_studies/markov_chain_monte_carlo.html): Section 1, 2, and 3

**Bayesian Statistics**

- {cite}`bolstad2009`, Chapters 2

**Sampling**

- {cite}`bolstad2009`, Chapters 2, 5, and 6 - all presented sampling methods and background
- {cite}`robert2004` - deeper dive into MCMC
- [Interactive MCMC visualizations](https://chi-feng.github.io/mcmc-demo/app.html?algorithm=RandomWalkMH&target=banana)


**Bayesian Inference**

- {cite}`bolstad2009`, Chapters 3

**Bayesian Regression / Classification**

- {cite}`murphy2022`
    - Section 10.5 on Bayesian logistic regression 
    - Section 11.7 on Bayesian linear regression