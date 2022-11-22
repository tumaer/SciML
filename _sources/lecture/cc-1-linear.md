# Core Content 1: Linear Models

## Linear Regression

Linear regression belongs to the family of **supervised learning** approaches, as it inherently requires labeled data. With it being the simplest regression approach. The simplest example to think of would be "Given measurement pairs $\left\{x^{(i)}, y^{\text {(i)}}\right\}_{i=1,...m}$, how to fit a line $h(x)$ to best approximate $y$?"


<center><img src = "https://i.imgur.com/kCnveaq.png" width = "350">
<img src = "https://i.imgur.com/pqga0NA.png" width = "350"></center>
<!-- 
x = np.linspace(1.0, 5, N)
X0 = x.reshape(N, 1)
X = np.c_[np.ones((N, 1)), X0]
w = np.array([1., 1 / 9.0])
y = 15 + w[0] * x + w[1] * np.square(x)
y = y + np.random.normal(0, 1, N)  -->

(Source: adapted from [Murphy](https://github.com/probml/pml-book))



*How does it work?*

1. We begin by formulating a hypothesis for the relation between input $x \in \mathbb{R}^{n}$, and output $y \in \mathbb{R}^{p}$. For conceptual clarity, we will initially set $p=1$. Then our hypothesis is represented by
    
    $$h(x)=\vartheta^{\top} x, \quad h \in \mathbb{R}^{p}$$

    as above $p=1$ in this case, and $\vartheta \in \mathbb{R}^{n}$ are the parameters of our hypothesis.

2. Then we need a strategy to fit our hypothesis parameters $\vartheta$ to the data points we have $\left\{x^{(i)}, y^{\text {(i)}}\right\}_{i=1,...m}$.

    1. Define a suitable cost function $J$, which emphasizes the importance of certain traits to the model. I.e. if a certain data area is of special importance to our model we should penalize modeling failures for those points much more heavily than others. A typical choice is the *Least Mean Square* (LMS) i.e.
    
        $$J(\vartheta)=\frac{1}{2} \sum_{i=1}^{m}\left(h\left(x^{(i)}\right)-y^{(i)}\right)^{2}$$

    2. Through an iterative application of gradient descent (more on this later in the course) find a $\vartheta$ which minimizes the cost function $J(\vartheta)$. If we apply gradient descent, our update function for the hypothesis parameters then takes the following shape

    $$\vartheta^{(k+1)}=\vartheta^{(k)}-\alpha\frac{\partial J}{\partial \vartheta^{k}}.$$

The iteration scheme can then be computed as:

$$
\begin{aligned}
\frac{\partial J}{\partial \vartheta_{j}}&=\frac{\partial}{\partial \vartheta_{j}} \frac{1}{2} \sum_{i}\left(h\left(x^{(i)}\right)-y^{(i)}\right)^{2} \\
&=\underset{i}{\sum} \frac{\partial }{\partial \vartheta_{j}}\left(h\left(x^{(i)}\right)-y^{(i)}\right)  \\
&=\sum_{i}\left(h\left(x^{(i)}\right)-y^{(i)}\right) x_{j}^{(i)}.
\end{aligned}
$$

resubstituting the iteration scheme into the update function, we then obtain the formula for **batch gradient descent**

$$\vartheta^{(k+1)}_j=\vartheta^{(k)}_j-\alpha\sum_{i}\left(h^{(k)}\left(x^{(i)}\right)-y^{(i)}\right) x_{j}^{(i)}.$$

If we choose an alternative update rate and seek to update every hypothesis parameter individually, then we can apply stochastic gradient descent.


If we now utilize matrix-vector calculus, then we can find the optimal $\vartheta$ in one shot. To do this we begin by defining our **design matrix $X$**

$$X_{m \times n}=\left[\begin{array}{c}x^{(1) \top }\\ \vdots \\ x^{(i) \top} \\ \vdots \\ x^{(m) \top}\end{array}\right]$$

and then define the feature vector from all samples as

$$Y_{m \times 1}=\left[\begin{array}{c}y^{(1)} \\ \vdots \\ y^{(i)} \\ \vdots \\ y^{(m)}\end{array}\right]$$

Connecting the individual pieces we then get the update function as

$$\left[\begin{array}{c}
h\left(x^{(0)}\right)-y^{(1)} \\
\vdots \\
h\left(x^{(i)}\right)-y^{(i)} \\
\vdots \\
h\left(x^{(m)}\right)-y^{(m)}
\end{array}\right]=X \vartheta _ {n\times 1} -Y$$

According to which, the cost function then becomes

$$J(\vartheta)=\frac{1}{2}(X \vartheta-Y)^{\top}(X \vartheta-Y).$$

As our cost function $J(\vartheta)$ is convex, we now only need to check that there exists a minimum i.e.

$$\nabla_{\vartheta} J(\vartheta) \stackrel{!}{=} 0.$$

Computing the derivative

$$\begin{align}
\nabla_{\vartheta} J(\vartheta)&=\frac{1}{2} \nabla_{\vartheta}(X \vartheta-Y)^{\top}(X \vartheta-Y) \\
& =\frac{1}{2} \nabla_{\vartheta}(\underbrace{\vartheta^{\top} X^{\top} X \vartheta-\vartheta^{\top} X^{\top} Y-Y^{\top} X \vartheta}_{\text {this is in fact a scalar for $p=1$}}+Y^{\top} Y) \quad \operatorname{as} {\nabla}_{\vartheta} Y^{\top} Y=0 \\
&=\frac{1}{2}\left(2X^{\top} X \vartheta-2 X^{\top} Y\right)
\\
&=X^{\top} X \vartheta-X^{\top} Y \stackrel{!}{=} 0
\end{align}
$$

<!-- &=\frac{1}{2} \nabla_{\vartheta} \operatorname{tr}(\cdots) -->

From which follows

$$
\begin{align}
\Rightarrow & \quad X^{\top} X \vartheta=X^{\top} Y
\\
\Rightarrow &\quad\vartheta=\left(X^{\top}X\right)^{-1}X^{\top}Y
\end{align}
$$


**Exercise: Linear Regression Implementations**
Implement the three approaches (batch gradient descent, stochastic gradient descent, and the matrix approach) to linear regression and compare their performance.
1. Batch Gradient Descent
2. Stochastic Gradient Descent
3. Matrix Approach


### Probabilistic Interpretation
With much data in practice, having errors over the collected data itself, we want to be able to include a data error in our linear regression. The approach for this is **Maximum Likelihood Estimation** as introduced in the *Introduction* lecture. I.e. this means data points are modeled as 

<!-- _{\uparrow} -->

$$y^{(i)}=\vartheta^{\top} x^{(i)}+\varepsilon^{(i)}$$

Presuming that all our data points are **independent, identically distributed (i.i.d)** random variables. The noise is modeled with a normal distribution.

$$p\left(\varepsilon^{(i)}\right)=\frac{1}{\sqrt{2 \pi \sigma^{2}}} e^{-\frac{\varepsilon^{(i) 2}}{2 \sigma^{2}}}$$

> While most noise distributions in practice are not normal, the normal (a.k.a. Gaussian) distribution has many nice theoretical properties making it much friendlier for theoretical derivations.

Using the data error assumption, we can now derive the probability density function (pdf) for the data

$$p\left(y^{(i)} \mid x^{(i)} ; \vartheta\right)=\frac{1}{\sqrt{2 \pi \sigma^{2}}} e^{-\frac{\left(y^{(i)}-\vartheta^{\top} x^{(i)}\right)^{2}}{2\sigma^{2}}}$$

where $y^{(i)}$ is conditioned on $x^{(i)}$. If we now consider not just the individual sample $i$, but the entire dataset, we can define the likelihood for our hypothesis parameters as

$$
\begin{align}
L(\vartheta) &=p(Y \mid X ; \vartheta)=\prod_{i=1}^{m} p\left(y^{(i)} \mid x^{(i)} ; \vartheta\right)
\\
&=\prod_{i=1}^{m} \frac{1}{\sqrt{2 \pi \sigma^{2}}} e^{-\frac{\left(y^{(i)}-\vartheta^{T} x^{(i)}\right)^{2}}{2 \sigma^{2}}}
\end{align}
$$

The probabilistic strategy to determine the optimal hypothesis parameters $\vartheta$ is then the maximum likelihood approach for which we resort to the **log-likelihood** as it is monotonically increasing, and easier to optimize for.


$$
\begin{aligned}
l(\vartheta)&=\log L(\vartheta)=\log \prod_{i=1}^{m} \frac{1}{\sqrt{2 \pi \sigma^{2}}} e^{\frac{\left(y^{(i)}-\vartheta^{\top} x^{(i)}\right)^{2}}{2 \sigma^{2}}}\\
&=\sum_{i=1}^{m} \log \frac{1}{\sqrt{2 \pi \sigma^{2}}} e^{ \frac{\left(y^{(i)}-\vartheta^{\top} x^{(i)}\right)^{2}}{2 \sigma^{2}}}\\
&=m \log \frac{1}{\sqrt{2 \pi \sigma^{2}}}-\frac{1}{2 \sigma^{2}} \sum_{i=1}^{m}\left(y^{(i)}-\vartheta^{\top} x^{(i)}\right)^{2}\\
&\Rightarrow \vartheta=\operatorname{argmax} l(\vartheta)=\operatorname{argmin} \sum_{i=1}^{m}\left(y^{(i)}-\vartheta^{\top} x^{(i)}\right)^{2}\\
\end{aligned}
$$

**This is the same result as minimizing $J(\vartheta)$ from before.** Interestingly enough, the Gaussian i.i.d. noise used in the maximum likelihood approach is entirely independent of $\sigma^{2}$.

> Least squares method (LMS), as well as maximum likelihood regression as above are **parametric learning** algorithms. 

> If the number of parameters is **not** known beforehand, then the algorithms become **non-parametric** learning algorithms.

Can maximum likelihood estimation (MLE) be made more non-parametric? Following intuition the linear MLE, as well as the LMS critically depends on the selection of the features, i.e the dimension of the parameter vector $\vartheta$. I.e. when the dimension of $\vartheta$ is too low we tend to underfit, where we do not capture some of the structure of our data. An approach to cope with the problem of underfitting here is to give more weight to new, unseen data $x$. E.g. for x, where we want to estimate y:

$$\hat{\delta}=\operatorname{argmax} \sum_{i=1}^{m} w^{(i)}\left(y^{(i)}-\vartheta^{\top} x^{(i)}\right)^{2}$$

where $\omega$ is given by

$$\omega^{(i)}=e^{-\frac{\left(x^{(i)}-x\right)^{2}}{2 \tau^{2}}}.$$

This approach naturally gives more weight to new datapoints in $x$. Hence making $\vartheta$ crucially depend on $x$, and making it more non-parametric.


## Classification & Logistic Regression

Summarizing the differences between regression and classification:

| Regression | Classification | 
| -------- | -------- |
| $x \in \mathbb{R}^{n}$    | $x \in \mathbb{R}^{n}$     |
| $y \in \mathbb{R}$  | $y \in\{0,1\}$ |


<div style="text-align:center">
    <img src="https://i.imgur.com/ZIb72vK.png" alt="drawing" width="500"/>
</div>

(Source: [Murphy](https://github.com/probml/pyprobml/blob/master/notebooks/book1/02/iris_logreg.ipynb))

To achieve such classification ability we have to introduce a new hypothesis function $h(x)$. A reasonable choice would be to model the probability that $y=1$ given $x$ with a function $h:\mathbb{R}\rightarrow [0,1]$. The logistic regression approach chooses

$$
h(x) = \varphi \left( \vartheta^{\top} x \right) = \frac{1}{1+e^{-\vartheta^{\top} x}}
$$

where 

$$\varphi(x)=\frac{1}{1+e^{-x}}=\frac{1}{2}\left(1+\tanh\frac{x}{2}\right)$$

is the logistic function, also called the sigmoid function. 


<div style="text-align:center">
    <img src="https://upload.wikimedia.org/wikipedia/commons/8/88/Logistic-curve.svg" alt="drawing" width="400"/>
</div>

(Source: [Wikipedia](https://en.wikipedia.org/wiki/Sigmoid_function))

The advantage of this function lies in its many nice properties, such as its derivative:

$$\varphi^{\prime} (x)=\frac{1}{1+e^{-x}} e^{-x}=\frac{1}{1+e^{-x}}\left(1-\frac{1}{1+e^{-x}}\right)=\varphi(x)(1-\varphi(x))$$

If we now want to apply the *Maximum Likelihood Estimation* approach, then we need to use our hypothesis to assign probabilities to the discrete events

$$\begin{cases}p(y=1 \mid x ; \vartheta)=h(x) & \\ p(y=0 \mid x ; \vartheta)=1-h(x) &
\end{cases}$$

Our probability density function then becomes

$$p(y \mid x ; \vartheta)=h^{y}(x)(1-h(x))^{1-y}$$

> This will look quite different for other types of labels, so be cautious in just copying this form of the pdf!

With our pdf we can then again construct the likelihood function

$$L(\vartheta) = p(y | x ; \vartheta) =\prod_{i=1}^{m} p\left(y^{(i)} \mid x^{(i)}, \vartheta\right)$$

Assuming the previously presumed classification buckets, and that the data is i.i.d.

$$
L(\vartheta)=\prod_{i=1}^{m} h^{y^{(i)}}(x^{(i)})\left(1-h\left(x^{(i)}\right)\right)^{1-y^{(i)}}
$$

and then the log-likelihood decomposes to

$$
l(\vartheta)=\log L(\vartheta)=\sum_{i=1}^{m}\left(y^{(i)} \log h\left(x^{(i)}\right)+\left(1-y^{(i)}\right) \log \left(1-h\left(x^{(i)}\right)\right)\right).
$$

Again we can find $\operatorname{argmax} l(\vartheta)$ e.g. by gradient ascent (batch or stochastic):

$$\vartheta_{j}^{(k+1)}=\vartheta_{j}^{(k)}+\left.\alpha \frac{\partial l(\vartheta)}{\partial \vartheta}\right|^{(k)}$$

$$\begin{align}
\frac{\partial \ell(\vartheta)}{\partial \vartheta_j} &=\left(y \frac{1}{h(x)}-(1-y) \frac{1}{1-h(x )}\right) \frac{\partial h(x)}{\partial \vartheta_j }\\
&=\left(\frac{y-h(x)}{h(x)(1-h(x))}\right) h (x)(1-h (x)) x_j\\
&=(y- h(x)) x_j
\end{align}$$

$$\Rightarrow \vartheta_{j}^{(k+1)}=\vartheta_{j}^{(k)}+\alpha \left( y^{(i)}-h(x^{(i)}) \right) x_j^{(i)}$$

which we can then solve with either batch gradient descent or stochastic gradient descent.

> The algorithm formally associated with the least squares method for logistic regression is slightly different!

The alternative here is to utilize Newton's method, which we have encountered before in numerical methods lectures. An application of Newton's method then looks like the following:

$$\vartheta_{j}^{(k+1)}=\vartheta_{j}^{(k)}\left(\left.\frac{\partial^{2} \ell}{\partial \vartheta_{i} \partial \vartheta_{j}}\right|^{(k)}\right)^{-1} \frac{\partial l}{\partial \vartheta_{j}}$$

Newton's method converges quadratically, and our problem class is sufficiently smooth for Newton's method to be applicable. The downside to this approach is that we have to compute the inverse Hessian matrix, which is an expensive computational operation.


**Exercise: Vanilla Indicator**

Using the "vanilla" indicator function instead of the sigmoid:

$$
g(x)= \begin{cases}1, & x \geqslant 0 \\ 0, & x<0\end{cases}
$$

derive the update functions for the gradient methods, as well as the Maximum Likelihood Estimator approach.


## Exponential Family

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


### Bayesian Inference

In the Bayesian framework, everything centers around the posterior distribution and our ability to relate our previous knowledge with newly gained evidence to the next stage of our belief (of a probability distribution). With the posterior being our entire inference about the parameters given the data there exist multiple inference approaches with their roots in frequentist statistics.

#### Bayesian Point Estimation

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


#### Bayesian Interval Estimation

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


#### Predictive Distribution of a New Observation

If we obtain a new observation, then we can compute the updated predictive distribution by combining the conditional distribution of the new observation, and conditioning it on the previous observations. Then we only need to integrate the parameter out of the joint posterior

$$f(y_{n+1}|y_{1}, \ldots, y_{n}) \propto \int g(\theta) \times f(y_{n+1}| \theta) \times \ldots \times f(y_{1}|\theta) d\theta$$
$$\propto \int f(y_{n+1}|\theta)g(\theta|y_{1}, \ldots, y_{n}) d\theta$$

and marginalize it out.


#### Bayesian Inference from a Posterior Random Sample

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

**Linear & Logistic Regression**

- Machine Learning Basics [video](https://www.youtube.com/watch?v=73RL3WPPFE0&list=PLQ8Y4kIIbzy_OaXv86lfbQwPHSomk2o2e&index=2) and [slides](https://niessner.github.io/I2DL/slides/2.Linear.pdf) from the "Introduction to Deep Learning" course for Informatics students at TUM.
- [What happens if a linear regression is underdetermined i.e. we have fewer observations than parameters?](https://betanalpha.github.io/assets/case_studies/underdetermined_linear_regression.html)

**Bayesian Methods**
There exist a wide number of references to the herein presented Bayesian approach, most famously introductory treatment of Probabilistic Programming frameworks, which utilize the herein presented modeling approach to obtain posteriors over programs.

- [Introduction to Pyro](http://pyro.ai/examples/intro_long.html)
- [A Practical Example with Stan](https://m-clark.github.io/bayesian-basics/example.html#posterior-predictive)

In addition, there exists highly curated didactic material from Michel Betancourt:

- [Sampling](https://betanalpha.github.io/assets/case_studies/sampling.html): Section 3, 4, and 5
- [Towards a Principled Bayesian Workflow](https://betanalpha.github.io/assets/case_studies/principled_bayesian_workflow.html)
- [Markov Chain Monte Carlo](https://betanalpha.github.io/assets/case_studies/markov_chain_monte_carlo.html): Section 1, 2, and 3
