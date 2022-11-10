# Introduction

## Famous Recent Examples of Scientific Machine Learning

With scientific machine learning becoming an ever larger mainstay at machine learning conferences, a/nd ever more venues and research centres at the intersection of machine learning and the natural sciences / engineering appearing there exist ever more impressive examples of algorithms which connect the very best of machine learning with deep scientific insight into the respective underlying problem to advance the field.

Below are a few prime examples of recent flagship algorithms in scientific machine learning, of which every single one of them personifies the very best algorithmic approaches we have available to us today.

### [AlphaFold](https://www.nature.com/articles/s41586-021-03819-2) - predicts 3D protein structure given its sequence:

<div style="text-align:center">
    <img src="https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fs41586-021-03819-2/MediaObjects/41586_2021_3819_Fig1_HTML.png?as=webp" alt="drawing" width="500"/>
</div><br/>


### [GNS](http://proceedings.mlr.press/v119/sanchez-gonzalez20a/sanchez-gonzalez20a.pdf) - capable of simulating the motion of water particles:

<div style="text-align:center">
    <img src="https://user-images.githubusercontent.com/544269/88065293-60aaba80-cba7-11ea-8f1e-ce3b30774775.png" alt="drawing" width="500"/>
</div><br/>


### [Codex](https://arxiv.org/abs/2107.03374) - translating natural language to code:

<div style="text-align:center">
    <img src="https://miro.medium.com/max/720/0*yfxfvwdlLBoGlmyU.gif" alt="drawing" width="500"/>
</div>
<br/>

### [Geometric Deep Learning](https://geometricdeeplearning.com/) 
Geometric deep learning aims to generalize neural network models to non-Euclidean domains such as graphs and manifolds. Good examples of this line of research include:

#### [SFCNN](https://arxiv.org/abs/1711.07289) - steerable rotation equivariant CNN, e.g. for image segmentation

<div style="text-align:center">
        <img src="https://i.imgur.com/t7DL28y.png" alt="drawing" width="500"/>
    <br/>
    </div>
    
    
#### [SEGNN](https://arxiv.org/abs/2110.02905) - molecular property prediction algorithm

<div style="text-align:center">
        <img src="https://i.imgur.com/bbJThJf.png" alt="drawing" width="500"/>
</div>


### [Stable Diffusion](https://stability.ai/blog/stable-diffusion-public-release) - generating images from natural text description

<div style="text-align:center">
    <img src="https://images.squarespace-cdn.com/content/v1/6213c340453c3f502425776e/0715034d-4044-4c55-9131-e4bfd6dd20ca/2_4x.png" alt="drawing" width="500"/>
</div><br/>

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

In supervised learning we have a mapping $x \longrightarrow y$, where the inputs $x$ are also called **features**, **covariates**, or **predictors**. The outputs $y$ are often also called the **labels**, **targets**, or **responses**. The correct mapping is then learned from a **labeled** training set

$$\mathcal{D}_{n} = \left\{ \left( x_{i} \right)_{i=1,n} \right\}$$

with $n$ the number of observations. Depending on the type of the response vector $y$, we can then perform either **regression**, or **classification**

> Some also call it "glorified curve-fitting"

#### Regression

In regression the target $y$ is is real-valued, i.e. $y \in \mathbb{R}$

<div style="text-align:center">
    <img src="https://i.imgur.com/sjxzmNa.jpg" alt="drawing" width="600"/>
</div>

(Source: [Murphy](https://github.com/probml/pml-book))

at the example of a response surface being fitted to a number of data points in 3 dimensions, where in this instance the $x$, and $y$ axes are a two-dimensional space, and the $z$-axis is the temperature in the two-dimensional space.

#### Classification

In classification the labels $y$ are categorical i.e. $y \in \mathcal{C}$, where $\mathcal{C}$ defines a set of classes.

<div style="text-align:center">
    <img src="https://i.imgur.com/DHD5IvI.png" alt="drawing" width="350"/>
</div>

(Source: [Murphy](https://github.com/probml/pml-book))

for the example of flower classification, where we aim to find the decision boundaries which will sort each individual node into the respective class.


### Unsupervised Learning

In unsupervised learning we only receive a dataset of inputs

$$\mathcal{D} = \{x_{n}: n = 1:N\}$$

without the respective outputs $y_{n}$, i.e. we only have **unlabelled** data.

> The implicit goal here is to describe the system, and identify features in the **high-dimensional inputs**.

Two famous examples of unsupervised learning are **clustering**, and especially **principal component analysis** which is commonly used in engineering and scientific applications.

#### Clustering of Principal Components

<div style="text-align:center">
    <img src="https://i.imgur.com/PyetOat.png" alt="drawing" width="400"/>
</div>

(Source: [Data-Driven Science and Engineering](http://databookuw.com))

Combining clustering with principal component analysis to show the samples which have cancer in the first three principal component coordinates.


### Supervised vs Unsupervised, the tl;dr in Probabilistic Terms

The difference can furthermore be expressed in probabilistic terms, i.e., in supervised learning we are fitting a model over the outputs conditioned on the inputs $p(y|x)$, whereas in unsupervised learning we are fitting an unconditional model $p(x)$.


### Reinforcement Learning

In reinforcement learning one sequentially interact with an unknown environment to obtain an interaction trajectory $T$, or a batch thereof. Reinforcement learning then seeks to optimize the way it interacts with the environment through its actions $a_{t}$ to maximize for a (cumulative) reward function to obtain an optimal strategy.

<div style="text-align:center">
    <img src="https://i.imgur.com/xexKADj.png" alt="drawing" width="500"/>
</div>

(Source: [lilianweng](https://lilianweng.github.io/posts/2018-02-19-rl-overview/))

## Polynomial Curve Fitting

Let's presume we have a simple regression problem, e.g.

<div style="text-align:center">
    <img src="https://i.imgur.com/ZU886t2.png" alt="drawing" width="400"/>
</div>

(Source: [Murphy](https://github.com/probml/pml-book))

then we have a number of observations ${\bf{x}} = (x_{1}, \ldots, x_{N})$, and a target ${\bf{y}} = (y_{1}, \ldots, y_{N})$. Then the tool we have probably seen before in the mechanical engineering curriculum is the simple approach to fit a polynomial function

$$y(x, w) = \omega_{0} + \omega_{1}x + \omega_{2} x^{2} + \ldots + \omega_{M}x^{M} = \sum_{j=0}^{M}\omega_{j}x^{j}$$

Then a crucial choice is the degree of the polynomial function.

> This class of models is called **Linear Models** because we want to learn only the linear scaling coefficients $w_i$, given any choice of basis for the variable $x$ like the polinomial basis shown here. 

We can then construct an error function with the sum of squares approach in which we are computing the distance of every target data point to our polynomial

$$E(w) = \frac{1}{2} \sum_{n=1}^{N} \{ y(x_{n}, w) - y_{n} \}^{2}$$

in which we are then optimizing for the value of $w$.

<div style="text-align:center">
    <img src="https://i.imgur.com/yfnGi4C.png" alt="drawing" width="400"/>
</div>

(Source: [Murphy](https://github.com/probml/pml-book))


To minimize this we then have to take the derivative with respect to the coefficients $\omega_{i}$, i.e.

$$\frac{\partial E(w)}{\partial \omega_{i}}=\sum_{n=1}^{N}\{ y(x_{n}, w) - y_{n} \}x_{n}^{i}=\sum_{n=1}^{N}\{ \sum_{j=0}^{M}\omega_{j} x_{n}^{j} - y_{n} \}x_{n}^{i}$$

which we are optimizing for and by setting to 0, we can then find the minimum

$$\sum_{n=1}^{N}\sum_{j=0}^{M}\omega_{j}x_{n}^{i}x_{n}^{j}=\sum_{n=1}^{N}y_{n}x_{n}^{i}$$

this can be solved by the trusty old Gaussian elimination. A general problem with this approach is that the degree of the polynomial is a decisive factor which often leads to over-fitting and hence makes this a less desirable approach. Gaussian elimination, or a matrix inversion approach when implemented on a computer can also be a highly expensive computational operation for large datasets.

> This is a special case of the **Maximum Likelihood** method.


## Bayesian Curve Fitting

**Recap: Bayes Theorem**

$$\mathbb{P}(A|B) = \frac{\mathbb{P}(B|A)\mathbb{P}(A)}{\mathbb{P}(B)}$$

If we now seek to reformulate the curve-fitting in probabilistic terms, then we have to begin by expressing our uncertainty over the target y with a probability distribution. For this we presume a Gaussian distribution over each target where the mean is the point value we previously considered, i.e.

$$p(y|x, w, \beta)=\mathcal{N}(y|y(x, w), \beta^{-1})$$

$\beta$ corresponds to the inverse variance of the distribution $\mathcal{N}$. We can then apply the maximum likelihood principle to find the optimal parameter $w$ with our new likelihood function

$$p(y|x, w, \beta)=\prod^{N}_{n=1}\mathcal{N}(y_{n}|y(x_{n},w), \beta^{-1}).$$

<div style="text-align:center">
    <img src="https://i.imgur.com/RVl2Z8R.png" alt="drawing" width="400"/>
</div>

(Source: [Bishop](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf))

Taking the log likelihood we are then able to find the definitions of the optimal parameters

$$\text{ln } p(y|x, w, \beta) = - \frac{\beta}{2} \sum^{N}_{2} \{ y(x_{n}, w) - y_{n} \}^{2} + \frac{N}{2} \text{ln } \beta - \frac{N}{2} \text{ln }(2 \pi)$$

Which we can then optimize for the $w$.

> If we consider the special case of $\frac{\beta}{2}=\frac{1}{2}$, and instead of maximizing, minimizing the negative log-likelihood, then this is equivalent to the sum-of-squares error function. 

The herein obtained optimal maximum likelihood parameter $w_{ML}$, and $\beta_{ML}$ can then be resubstituted to obtain the **predictive distribution** for the targets $y$.

$$p(y|x, w_{ML}, \beta_{ML})=\mathcal{N}(y|y(x, w_{ML}),\beta_{ML}^{-1})$$

To arrive at the full Bayesian curve-fitting approach we now have to apply the sum and product rules of probability

**Recap: Sum Rules of Probability**

$$\mathbb{P}(A \cap B) = \mathbb{P}(A) + \mathbb{P}(B)$$


**Recap: Product Rules of Probability**

$$\mathbb{P}(A, B) = \mathbb{P}(A) \cdot \mathbb{P}(B),$$
where $A$, and $B$ must be **independent** events.


The Bayesian curve fitting formula is hence

$$p(y|x, {\bf{x}}, {\bf{y}}) = \int p(y|x, w)p(w|{\bf{x}}, {\bf{y}})dw$$

with the dependence on $\beta$ omitted for legibility reasons. This integral can be solved analytically hence giving the following predictive distribution 

$$p(y|x, {\bf{x}}, {\bf{y}})=\mathcal{N}(y|m(x), s^{2}(x))$$

with mean and variance 

$$m(x) = \beta \phi(x)^{T} {\bf{S}} \sum_{n=1}^{N}\phi(x_{n})y_{n},$$
$$s^{2}(x) = \beta^{-1} + \phi(x)^{T} {\bf{S}} \phi(x),$$

and ${\bf{S}}$ defined as

$${\bf{S}}^{-1} = \alpha {\bf{I}} + \beta \sum_{n=1}^{N} \phi(x_{n}) \phi(x)^{T},$$

and ${\bf{I}}$ the unit matrix, while $\phi(x)$ is defined by $\phi_{i}(x) = x^{i}$. Examining the variance $s^{2}(x)$, then the benefits of the Bayesian approach become readily apparent:

- $\beta^{-1}$ represents the prediction uncertainty
- The second term of the variance is caused by the uncertainty in the parameters $w$.


## Maximum Likelihood, Information Theory & Log-Likelihood

### Maximum Likelihood & Log-Likelihood

Formalizing the principle of maximum likelihood estimation, we seek to act in a similar fashion to function extrema calculation in high school in that we seek to differentiate our likelihood function, and then find the maximum value of it by setting the derivative equal to zero.

$$\hat{\theta} = \underset{\theta \in \Theta}{\arg \max} \mathcal{L}_{n}(\theta; y)$$

where $\mathcal{L}$ is the likelihood function. In the derivation of the Bayesian Curve Fitting approach we have already utilized this principle by exploiting the independence between data points and then taking log of the likelihood to then utilize the often nicer properties of the log-likelihood

$$l(\theta;y) = \ln \mathcal{L}_{n}(\theta;y).$$


### Information Theory

The core concept of information theory to consider here is that not every data point is equal to us! If an individual data point is outside of the previously estimated dynamics model, then it is a much more valuable data point than a data point which is e.g. directly on the response surface. Such information content can be measure with a function which is a logarithm of the probability of us observing a specific data point $p(x)$.

$$h(x) = - \text{log}_{2} p(x)$$

i.e. the $-$ ensures that provided information value is either net-positive or neutral. This can be further formalized with the concept of **information entropy** which is the expectation of $h(x)$

$$H[x] = - \sum_{x} p(x) \text{log}_{2} p(x).$$

Using statistical mechanics one can then derive the definitions of the entropy for continuous, as well as discrete variables.

**Entropy of Discrete Variables**

$$H[p]= - \sum_{i} p(x_{i}) \text{ln } p(x_{i})$$

**Entropy of Continuous Variables**

$$H[x] = - \int p(x) \text{ln }p(x) dx$$


## Recap

- Supervised Learning: Mapping from inputs $x$ to outputs $y$
- Unsupervised Learning: Only receives the datapoints $x$ with **no** access to the true labels $y$
- Maximum Likelihood Principle
    - Polynomial Curve Fitting a special case
    - Wasteful of training data, and tends to overfit
    - Bayesian approach less prone to overfitting
- Information theory, and more specifically the concept of entropy, is the vehicle with which we can quantify just how valuable an added datapoint is to us.


## Further References

- [Using AI to Accelerate Scientific Discovery](https://youtu.be/AU6HuhrC65k) - inspirational video by Demis Hassabis
