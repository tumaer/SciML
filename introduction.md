# Introduction

## Famous Recent Examples of Scientific Machine Learning

With scientific machine learning becoming an ever larger mainstay at machine learning conferences, and ever more venues and research centres at the intersection of machine learning and the natural sciences / engineering appearing there exist ever more impressive examples of algorithms which connect the very best of machine learning with deep scientific insight into the respective underlying problem to advance the field.

Below are a few prime examples of recent flagship algorithms in scientific machine learning, of which every single one of them personifies the very best algorithmic approaches we have available to us today.

### [AlphaFold](https://www.nature.com/articles/s41586-021-03819-2) - predicts 3D protein structure given its sequence:


```{figure} imgs/alphafold.png
---
width: 500px
align: center
name: alphafold
---
AlphaFold model. (Source:  [Jumper et al., 2021](https://www.nature.com/articles/s41586-021-03819-2))
```

### [GNS](https://proceedings.mlr.press/v119/sanchez-gonzalez20a.html) - capable of simulating the motion of water particles:

```{figure} imgs/gns.png
---
width: 500px
align: center
name: gns
---
GNS model. (Source:  [Sanchez-Gonzalez et al., 2020](https://proceedings.mlr.press/v119/sanchez-gonzalez20a.html))
```

### [Codex](https://arxiv.org/abs/2107.03374) - translating natural language to code:

```{figure} imgs/codex.gif
---
width: 500px
align: center
name: codex
---
Codex demo (Source: [openai.com](https://openai.com/blog/openai-codex))
```

### [Geometric Deep Learning](https://geometricdeeplearning.com/) 
Geometric deep learning aims to generalize neural network models to non-Euclidean domains such as graphs and manifolds. Good examples of this line of research include:

#### [SFCNN](https://arxiv.org/abs/1711.07289) - steerable rotation equivariant CNN, e.g. for image segmentation

```{figure} imgs/sfcnn.png
---
width: 500px
align: center
name: sfcnn
---
SFCNN model. (Source: [Weiler et al., 2018](https://arxiv.org/abs/1711.07289))
``` 

#### [SEGNN](https://arxiv.org/abs/2110.02905) - molecular property prediction model

```{figure} imgs/segnn.png
---
width: 500px
align: center
name: segnn
---
SEGNN model. (Source: [Brandstetter et al., 2022](https://arxiv.org/abs/2110.02905))
```

### [Stable Diffusion](https://stability.ai/blog/stable-diffusion-public-release) - generating images from natural text description

```{figure} imgs/stable_diffusion_art.jpg
---
width: 500px
align: center
name: stablediffusion
---
Stable Diffusion art. (Source: [stability.ai](https://stability.ai/blog/stable-diffusion-public-release))
```

### [Stable Diffusion reconstructing visual experiences from human brain activity](https://www.biorxiv.org/content/10.1101/2022.11.18.517004v3)

```{figure} imgs/stable_diffusion_brain.jpg
---
width: 700px
align: center
name: stablediffusion_brain
---
Stable Diffusion brain signal reconstruction. (Source: [Takagi & Nishimito, 2023](https://sites.google.com/view/stablediffusion-with-brain/?s=09))
```

### [ImageBind](https://arxiv.org/abs/2305.05665) - Holistic AI learning across six modalities

```{figure} imgs/imagebind.gif
---
width: 600px
align: center
name: imagebind
---
ImageBind modalities. (Source: [ai.meta.com](https://ai.meta.com/blog/imagebind-six-modalities-binding-ai/))
```

## Definition

Machine learning at the intersection of engineering, physics, chemistry, computational biology etc. and core machine learning to improve existing scientific workflows, derive new scientific insight, or bridge the gap between scientific data and our current state of knowledge.

Important here to recall is the difference in approaches between engineering & physics, and machine learning on the other side:

**Engineering & Physics**

Models are derived from conservation laws, observations, and established physical principles.

**Machine Learning**

Models are derived from data with imprinted priors on the model space either through the data itself, or through the design of the machine learning algorithm.


## Supervised vs Unsupervised

There exist 3 main types of modern day machine learning:

1. Supervised Learning
2. Unsupervised Learning
3. Reinforcement Learning


### Supervised Learning

In supervised learning we have a mapping $f: X \rightarrow Y$, where the inputs $x \in X$ are also called **features**, **covariates**, or **predictors**. The outputs $y \in Y$ are often also called the **labels**, **targets**, or **responses**. The correct mapping is then learned from a **labeled** training set

$$\mathcal{D}_{N} = \left\{ \left( x_{n}, y_{n} \right) \right\}_{n=1:N}$$ (supervised_data)

with $N$ the number of observations. Depending on the type of the response vector $y$, we can then perform either **regression**, or **classification**

> Some also call it "glorified curve-fitting"

#### Regression

In regression, the target $y$ is real-valued, i.e. $y \in \mathbb{R}$

```{figure} imgs/2d_regression_ex.jpg
---
width: 600px
align: center
name: 2d_regression_ex
---
2D regression example. (Source: {cite}`murphy2022`, Introduction)
```

Example of a response surface being fitted to a number of data points in 3 dimensions, where in this instance the x- and y-axes are a two-dimensional space, and the z-axis is the temperature in the two-dimensional space.

#### Classification

In classification the labels $y$ are categorical i.e. $y \in \mathcal{C}$, where $\mathcal{C}$ defines a set of classes.

```{figure} imgs/iris_classification.png
---
width: 350px
align: center
name: iris_classification
---
Classification example. (Source: {cite}`murphy2022`, Introduction)
```

Example of flower classification, where we aim to find the decision boundaries which will sort each individual node into the respective class.


### Unsupervised Learning

In unsupervised learning, we only receive a dataset of inputs

$$\mathcal{D}_{N} = \left\{ x_{n} \right\}_{n=1:N}$$ (unsupervised_data)

without the respective outputs $y_{n}$, i.e. we only have **unlabelled** data.

> The implicit goal here is to describe the system, and identify features in the **high-dimensional inputs**.

Two famous examples of unsupervised learning are **clustering** (e.g. k-means) and especially **dimensionality reduction** (e.g. principal component analysis) which is commonly used in engineering and scientific applications.

#### Clustering of Principal Components

```{figure} imgs/pca_clustering.png
---
width: 400px
align: center
name: pca_clustering
---
Clustering based on principal components. (Source: {cite}`brunton2019`, Section 1.5)
```

Combining clustering with principal component analysis to show the samples which have cancer in the first three principal component coordinates.


### Supervised vs Unsupervised, the tl;dr in Probabilistic Terms

The difference can furthermore be expressed in probabilistic terms, i.e., in supervised learning we are fitting a model over the outputs conditioned on the inputs $p(y|x)$, whereas in unsupervised learning we are fitting an unconditional model $p(x)$.


### Reinforcement Learning

In reinforcement learning, an agent sequentially interact with an unknown environment to obtain an interaction trajectory $T$, or a batch thereof. Reinforcement learning then seeks to optimize the way the agent interacts with the environment through its actions $a_{t}$ to maximize for a (cumulative) reward function to obtain an optimal strategy.

```{figure} imgs/rl.png
---
width: 500px
align: center
name: rl
---
Reinforcement learning overview. (Source: [lilianweng](https://lilianweng.github.io/posts/2018-02-19-rl-overview/))
```

## Polynomial Curve Fitting

Let's presume we have a simple regression problem, e.g.

```{figure} imgs/lin_reg_1d.png
---
width: 400px
align: center
name: lin_reg_1d
---
Linear regression example. (Source: {cite}`murphy2022`, Introduction)
```

Then we have a number of scalar observations ${\bf{x}} = (x_{1}, \ldots, x_{N})$ and targets ${\bf{y}} = (y_{1}, \ldots, y_{N})$. Then the tool we have probably seen before in the mechanical engineering curriculum is the simple approach to fit a polynomial function

$$y(x, \mathbf{w}) = \omega_{0} + \omega_{1}x + \omega_{2} x^{2} + \ldots + \omega_{M}x^{M} = \sum_{j=0}^{M}\omega_{j}x^{j}$$ (polynomial_regression)

Then a crucial choice is the degree of the polynomial function.

> This class of models is called **Linear Models** because we want to learn only the linear scaling coefficients $w_i$, given any choice of basis for the variable $x$ like the polinomial basis shown here. 

We can then construct an error function with the sum of squares approach in which we are computing the distance of every target data point to our polynomial

$$E(\mathbf{w}) = \frac{1}{2} \sum_{n=1}^{N} \{ y(x_{n}, \mathbf{w}) - y_{n} \}^{2}$$ (l2_loss)

in which we are then optimizing for the value of $w$.

```{figure} imgs/lin_reg_1d_distances.png
---
width: 400px
align: center
name: lin_reg_1d_distances
---
Linear regression error computation. (Source: {cite}`murphy2022`, Introduction)
```

To minimize this we then have to take the derivative with respect to the coefficients $\omega_{i}$, i.e.

$$\frac{\partial E(w)}{\partial \omega_{i}}=\sum_{n=1}^{N}\{ y(x_{n}, w) - y_{n} \}x_{n}^{i}=\sum_{n=1}^{N}\{ \sum_{j=0}^{M}\omega_{j} x_{n}^{j} - y_{n} \}x_{n}^{i}$$ (grad_l2_loss)

which we are optimizing for and by setting to 0, we can then find the minimum

$$\sum_{n=1}^{N}\sum_{j=0}^{M}\omega_{j}x_{n}^{i}x_{n}^{j}=\sum_{n=1}^{N}y_{n}x_{n}^{i}$$ (lin_reg_polynomial_solution)

This can be solved by the trusty old Gaussian elimination. A general problem with this approach is that the degree of the polynomial is a decisive factor which often leads to over-fitting and hence makes this a less desirable approach. Gaussian elimination, or a matrix inversion approach when implemented on a computer can also be a highly expensive computational operation for large datasets.

> This is a special case of the **Maximum Likelihood** method.


## Bayesian Curve Fitting

**Recap: Bayes Theorem**

$$\mathbb{P}(A|B) = \frac{\mathbb{P}(B|A)\mathbb{P}(A)}{\mathbb{P}(B)}$$ (bayes_theorem)

If we now seek to reformulate the curve-fitting in probabilistic terms, then we have to begin by expressing our uncertainty over the target $y$ with a probability distribution. For this we presume a Gaussian distribution over each target where the mean is the point value we previously considered, i.e.

$$p(y|x, \mathbf{w}, \beta)=\mathcal{N}(y|y(x, \mathbf{w}), \beta^{-1}),$$ (bayesian_lin_reg_sample_likelihood)

where $\beta$ corresponds to the inverse variance of the normal distribution $\mathcal{N}$. We can then apply the maximum likelihood principle to find the optimal parameters $\mathbf{w}$ with our new likelihood function

$$p(y|x, \mathbf{w}, \beta)=\prod^{N}_{n=1}\mathcal{N}(y_{n}|y(x_{n},\mathbf{w}), \beta^{-1}).$$ (bayesian_lin_reg_joint_likelihood)

```{figure} imgs/bayesian_reg_1d.png
---
width: 400px
align: center
name: bayesian_reg_1d
---
Bayesian regression example. (Source: {cite}`bishop2006`, Section 1.2)
```

Taking the log likelihood we are then able to find the definitions of the optimal parameters

$$\text{ln } p(y|x, \mathbf{w}, \beta) = - \frac{\beta}{2} \sum^{N}_{2} \{ y(x_{n}, \mathbf{w}) - y_{n} \}^{2} + \frac{N}{2} \text{ln } \beta - \frac{N}{2} \text{ln }(2 \pi)$$ (bayesian_lin_reg_joint_likelihood_log)

Which we can then optimize for the $\mathbf{w}$.

> If we consider the special case of $\frac{\beta}{2}=\frac{1}{2}$, and instead of maximizing, minimizing the negative log-likelihood, then this is equivalent to the sum-of-squares error function. 

The herein obtained optimal maximum likelihood parameters $\mathbf{w}_{ML}$ and $\beta_{ML}$ can then be resubstituted to obtain the **predictive distribution** for the targets $y$.

$$p(y|x, \mathbf{w}_{ML}, \beta_{ML})=\mathcal{N}(y|y(x, \mathbf{w}_{ML}),\beta_{ML}^{-1})$$ (bayesian_lin_reg_ml_sol)

To arrive at the full Bayesian curve-fitting approach we now have to apply the sum and product rules of probability

**Recap: Sum Rules of (disjoint) Probability**

$$\mathbb{P}(A \cap B) = \mathbb{P}(A) + \mathbb{P}(B)$$ (sum_rule_disjoint)


**Recap: Product Rules of Probability** - for **independent** events $A,B$.

$$\mathbb{P}(A, B) = \mathbb{P}(A) \cdot \mathbb{P}(B)$$ (product_rule_iid)

The Bayesian curve fitting formula is hence

$$p(y|x, {\bf{x}}, {\bf{y}}) = \int p(y|x, \mathbf{w})p(\mathbf{w}|{\bf{x}}, {\bf{y}})d\mathbf{w}$$ (bcf_posterior)

with the dependence on $\beta$ omitted for legibility reasons. This integral can be solved analytically hence giving the following predictive distribution 

$$p(y|x, {\bf{x}}, {\bf{y}})=\mathcal{N}(y|m(x), s^{2}(x))$$ (bcf_posterior_gaussians)

with mean and variance 

$$m(x) = \beta \phi(x)^{T} {\bf{S}} \sum_{n=1}^{N}\phi(x_{n})y_{n},$$ (bcf_posterior_gaussians_mean)

$$s^{2}(x) = \beta^{-1} + \phi(x)^{T} {\bf{S}} \phi(x),$$ (bcf_posterior_gaussians_var)

and ${\bf{S}}$ defined as

$${\bf{S}}^{-1} = \alpha {\bf{I}} + \beta \sum_{n=1}^{N} \phi(x_{n}) \phi(x)^{T},$$ (bcf_posterior_gaussians_sol_s)

and ${\bf{I}}$ the unit matrix, while $\phi(x)$ is defined by $\phi_{i}(x) = x^{i}$. Examining the variance $s^{2}(x)$, then the benefits of the Bayesian approach become readily apparent:

- $\beta^{-1}$ represents the prediction uncertainty
- The second term of the variance is caused by the uncertainty in the parameters $w$.


## Maximum Likelihood & Log-Likelihood

Formalizing the principle of maximum likelihood estimation, we seek to act in a similar fashion to function extrema calculation in high school in that we seek to differentiate our likelihood function, and then find the maximum value of it by setting the derivative equal to zero.

$$\hat{\theta} = \underset{\theta \in \Theta}{\arg \max} \mathcal{L}_{n}(\theta; y),$$ (maximum_likelihood_objective)

where $\mathcal{L}$ is the likelihood function. In the derivation of the Bayesian Curve Fitting approach we have already utilized this principle by exploiting the independence between data points and then taking log of the likelihood to then utilize the often nicer properties of the log-likelihood

$$l(\theta;y) = \ln \mathcal{L}_{n}(\theta;y).$$ (log_likelihood)



## Recap

- Supervised Learning: Mapping from inputs $x$ to outputs $y$
- Unsupervised Learning: Only receives the datapoints $x$ with **no** access to the true labels $y$
- Maximum Likelihood Principle
    - Polynomial Curve Fitting a special case
    - Wasteful of training data, and tends to overfit
    - Bayesian approach less prone to overfitting


## Further References

- [Using AI to Accelerate Scientific Discovery](https://youtu.be/AU6HuhrC65k) - inspirational video by Demis Hassabis
