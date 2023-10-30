# Bayesian methods

We present Bayesian Inference and its applications to regression and classification.



## Bayesian Inference

The main ideals upon which Bayesian statistics is founded are

- Uncertainty over parameters -> treatment as random variables
- A probability over a parameter essentially expresses a degree of belief
- Inference over parameters using rules of probability
- We combine the prior knowledge and the observed data with Bayes' theorem

> Refresher - Bayes' theorem is given by $\mathbb{P}(A|B) = \frac{\mathbb{P}(B|A)\mathbb{P}(A)}{\mathbb{P}(B)}$

So what we are interested in is the **posterior** distribution over our parameters, which can be found using Bayes' theorem. While this may at first glance look straightforward, and holds for the **unscaled posterior** i.e the distribution which has not been normalized by dividing by $\mathbb{P}(B)$, obtaining the scaled posterior is much much harder due to the difficulty in computing the divisor $\mathbb{P}(B)$. To evaluate this divisor we draw on Monte Carlo sampling.

> What is a _scaled posterior_? A scaled posterior is a distribution whose integral over the entire distribution evaluates to 1. 

Taking Bayes' theorem, and using the probability theorems in their conditional form we then obtain the following formula for the posterior density

$$g(\theta | y) = \frac{g(\theta) \times f(y | \theta)}{\int g(\theta) \times f(y | \theta) d\theta}$$

If we now seek to compute the denominator, then we have to integrate

$$\int f(y|\theta) g(\theta) d\theta.$$


### Monte Carlo Integration
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

---

In the Bayesian framework, everything centers around the posterior distribution and our ability to relate our previous knowledge with newly gained evidence to the next stage of our belief (of a probability distribution). With the posterior being our entire inference about the parameters given the data there exist multiple inference approaches with their roots in frequentist statistics.

### Bayesian Point Estimation

Bayesian point estimation chooses a single value to represent the entire posterior distribution. Potential choices here are locations like the posterior mean, and posterior median. For the posterior mean squared error, the posterior mean is then the first moment of the posterior distribution

$$PMS(\hat{\theta}) = \int (\theta - \hat{\theta})^{2} g(\theta | y_{1}, \ldots, y_{n})d\theta$$
$$\hat{\theta} = \int_{-\infty}^{\infty} \theta g(\theta | y_{1}, \ldots, y_{n})d \theta$$

<center>
<img src = "https://i.imgur.com/C5WdqqR.png" width = "450"></center>

and for the posterior median $\tilde{\theta}$

$$PMAD(\hat{\theta}) = \int |\theta - \hat{\theta}| g(\theta| y_{1}, \ldots, y_{n})d\theta$$
$$.5 = \int_{-\infty}^{\tilde{\theta}} g(\theta | y_{1}, \ldots, y_{n}) d\theta$$

<center>
<img src = "https://i.imgur.com/u2b81gQ.png" width = "450"></center>


### Bayesian Interval Estimation

Another type of Bayesian inference is the one in which we seek to find an interval that, with a pre-determined probability, contains the true value. In Bayesian statistics, these are called credible intervals. Finding the interval with equal tail areas to both sides $(\theta_{l}, \theta_{u})$, and which has the probability to contain the true value of the parameter, i.e.

$$\int_{-\infty}^{\theta_{l}} g(\theta | y_{1}, \ldots, y_{n}) d\theta = \frac{\alpha}{2}$$
$$\int_{\theta_{u}}^{\infty} g(\theta | y_{1}, \ldots, y_{n}) d\theta = \frac{\alpha}{2}$$

which we then only need to solve. A visual example of such a scenario is in the following picture:

<center>
<img src = "https://i.imgur.com/ofkfuF0.png" width = "450"></center>

(Source: Bolstad, _Understanding Computational Bayesian Statistics_)


<!---
#### Maximum A Posteriori Estimation (MAP)

Blubba blub
--->


### Predictive Distribution of a New Observation

If we obtain a new observation, then we can compute the updated predictive distribution by combining the conditional distribution of the new observation, and conditioning it on the previous observations. Then we only need to integrate the parameter out of the joint posterior

$$f(y_{n+1}|y_{1}, \ldots, y_{n}) \propto \int g(\theta) \times f(y_{n+1}| \theta) \times \ldots \times f(y_{1}|\theta) d\theta$$
$$\propto \int f(y_{n+1}|\theta)g(\theta|y_{1}, \ldots, y_{n}) d\theta$$

and marginalize it out.


### Bayesian Inference from a Posterior Random Sample

When we only have a random sample from the posterior instead of the approximation, we are still able to apply the same techniques, but just apply them to the posterior sample.

With our rudimentary approximations of the denominator by sampling from the posterior. This only constitutes an approximation, but given the sampling budget, this approximation can be made as accurate as desired. In summary Bayesian inference can be condensed to the following main take-home knowledge:

- The posterior distribution is the current summary of beliefs about the parameter in the Bayesian framework.
- Bayesian inference is then performed using the probabilities calculated from the posterior distribution of the parameter.
    - To get an approximation of the scaling factor for the posterior we have to utilize sampling-based Monte Carlo techniques to approximate the requisite integral.


## Bayesian Approaches to Regression

If we are faced with the scenario of having very little data, then we ideally seek to quantify the uncertainty of our model and preserve the predictive utility of our machine learning model. The right approach to this is to extend Linear Regression, and Logistic Regression with the just presented Bayesian Approach utilizing Bayesian Inference.

### Bayesian Logistic Regression

If we now want to capture the uncertainty over our predictions in our logistic regression, then we have to resort to the Bayesian approach. To make the Bayesian approach work for logistic regression, we have to apply something called the _Laplace Approximation_ in which we approximate the posterior using a Gaussian

$$p(\omega | \mathcal{D}) \approx \mathcal{N}({\bf{\omega}}| {\bf{\hat{\omega}}}, {\bf{H}}^{-1})$$

where $H^{-1}$ is the inverse of the Hessian, $\omega$ corresponds to the learned parameters $\vartheta$, and $\hat{\omega}$ is the MLE of $\vartheta$. There exist many different modes representing viable solutions for this problem when we seek to optimize it.

<center>
<img src = "https://i.imgur.com/1nGR1Ju.png" width = "450"></center>

(Source, [Murphy](https://github.com/probml/pml-book))

Using a Gaussian prior centered at the origin, we can then multiply our prior with the likelihood to obtain the unnormalized posterior. Which yields us the posterior predictive distribution

$$p(y|x, \mathcal{D}) = \int p(y | x, \omega) p(\omega | \mathcal{D})) d\omega$$

To now compute the uncertainty in our predictions we use a Gaussian prior, and then perform a _Monte Carlo Approximation_ of the integral using $S$ samples from the posterior $\omega_s \sim p(\omega|\mathcal{D})$

$$p(y=1 | x, \mathcal{D}) = \frac{1}{S} \sum_{s=1}^{S} \sigma \left( \omega_{s}^{\top} x \right)$$

Looking at a larger visual example of Bayesian Logistic Regression applied.

<center>
<img src = "https://i.imgur.com/oBsUrVi.jpg" width = "600"></center>

(Source: [Murphy](https://github.com/probml/pml-book))


### Bayesian Linear Regression

To now introduce the Bayesian approach to linear regression we have to assume that we already know the variance $\sigma^{2}$, so the posterior which we actually compute at that point is

$$p(\omega | \mathcal{D}, \sigma^{2})$$

If we then take a Gaussian distribution as our prior distribution $p(\omega)$

$$p(\omega) = \mathcal{N}(\omega | \breve{\omega}, \breve{\Sigma})$$

Then we can write down the likelihood as a Multivariate-Normal distribution.

$$p(\mathcal{D} | \omega, \sigma^{2}) = \prod_{n=1}^{N}p(y_{n}|{\bf{\omega^{\top}}}{\bf{x}}, \sigma^{2}) = \mathcal{N}({\bf{y}} | {\bf{X} \bf{\omega}}, \sigma^{2} {\bf{I}}_{N})$$

The posterior can then be analytically derived using Bayes' rule for Gaussians (see [Murphy](https://github.com/probml/pml-book), eq. 3.37)

$$p({\bf{\omega}} | {\bf{X}}, {\bf{y}}, \sigma^{2}) \propto \mathcal{N}(\omega | \breve{\omega}, \breve{\Sigma}) \mathcal{N}({\bf{y}} | {\bf{X} \bf{\omega}}, \sigma^{2} {\bf{I}}_{N}) = \mathcal{N}({\bf{\omega}} | {\bf{\hat{\omega}}}, {\bf{\hat{\Sigma}}}) $$
$${\bf{\hat{\omega}}} \equiv {\bf{\hat{\Sigma}}} \left( {\bf{\breve{\Sigma}}}^{-1} {\bf{\breve{\omega}}} + \frac{1}{\sigma^{2}} {\bf{X^{\top} y}}  \right)$$
$${\bf{\hat{\Sigma}}} \equiv \left( {\bf{\breve{\Sigma}}}^{-1} + \frac{1}{\sigma^{2}} {\bf{X^{\top} X}} \right)^{-1}$$

where $\hat{\omega}$ is the posterior mean, and $\hat{\Sigma}$ is the posterior covariance. A good visual example of this is the sequential Bayesian inference on a linear regression model:

<center>
<img src = "https://i.imgur.com/87em7Vz.png" width = "550"></center>

(Source, [Murphy](https://github.com/probml/pml-book))


## Bayesian Machine Learning

Let's consider the setup we have encountered so far in which we have labels $x$, hyperparameters $\theta$, and seek to predict labels $y$. Probabilistically expressed this amounts to $p(y|x, \theta)$. Then the posterior is defined as $p(\theta| \mathcal{D})$, where $\mathcal{D}$ is our labeled dataset

$$\mathcal{D} = \left\{ (x_{n}, y_{n}):n=1:N \right\}$$

Applying the previously discussed Bayesian approaches to these problems, and the respective model parameters, are called **Bayesian Machine Learning**.

While we lose computational efficiency at first glance, as we have to perform a sampling-based inference procedure, what we gain is a principled approach to discuss uncertainties within our model. This can help us most especially when we move in the *small-data limit*, where we can not realistically expect our model to converge. See e.g. below a Bayesian logistic regression example in which the posterior distribution is visualized.


<center>
<img src = "https://i.imgur.com/AfikBRy.png" width = "550"></center>

(Source: [Murphy](https://github.com/probml/pml-book))



## Further References

**Bayesian Methods**
There exist a wide number of references to the herein presented Bayesian approach, most famously introductory treatment of Probabilistic Programming frameworks, which utilize the herein presented modeling approach to obtain posteriors over programs.

- [Introduction to Pyro](http://pyro.ai/examples/intro_long.html)
- [A Practical Example with Stan](https://m-clark.github.io/bayesian-basics/example.html#posterior-predictive)

In addition, there exists highly curated didactic material from Michel Betancourt:

- [Sampling](https://betanalpha.github.io/assets/case_studies/sampling.html): Section 3, 4, and 5
- [Towards a Principled Bayesian Workflow](https://betanalpha.github.io/assets/case_studies/principled_bayesian_workflow.html)
- [Markov Chain Monte Carlo](https://betanalpha.github.io/assets/case_studies/markov_chain_monte_carlo.html): Section 1, 2, and 3
