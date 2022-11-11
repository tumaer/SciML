# Core Content 2: Optimization

<!-- 
[link](https://mlstory.org/optimization.html) 
[link](https://www.d2l.ai/chapter_optimization/index.html)
vis.ensmallen.org
-->

In Core Content 1, we already saw the main building blocks of a supervised learning algorithm: 
1. **Model** $h$ of the relationship between inputs $x$ and outputs $y$,
2. **Loss** (aka **Cost**, **Error**) function $J(\vartheta)$ quantifying the discrepancy between $h_{\vartheta}(x^{(i)})$ and $y^{(i)}$ for each of the measurement pairs $\left\{x^{(i)}, y^{\text {(i)}}\right\}_{i=1,...m}$, and 
3. Optimization algorithm (aka **Optimizer**) minimizing the loss. 

Point 1 is a matter of what we know about the world in advance and how we include that knowledge into the model, e.g. choose CNNs when working with images because the same pattern might appear at different locations of an image. In Core Contents 3 and 4, we will look at different models each of which is by construction better suited for different problem types.

After selecting a model $h_{\vartheta}$, points 2 and 3 are critical to the success of the learning as the loss function (point 2) defines how we measure success and the optimizer (point 3) guides the process of moving from a random initial guess of the parameters to a parameter configuration with much smaller loss. These two point are the topic of Core Content 2: how do the loss influence training and which optimization methods exist?

## Basics of Optimization

First, we define the general (unconstrained) minimization problem

$$\text{argmin}_{\vartheta} \; J(\vartheta),$$

where $J:\mathbb{R}^n \rightarrow \mathbb{R}$ is a real-valued function. We call the optimal solution of this problem the *minimizer* of $J$ and denote it as $\vartheta_{\star}$. The minimizer is defined as $J(\vartheta_{\star}) \le J(\vartheta)$ for all $\vartheta$. If the relation $J(\vartheta_{\star}) \le J(\vartheta)$ holds in a local neighborhood $||\vartheta - \vartheta_{\star}|| \le \epsilon$, for some $\epsilon>0$, then we call $\vartheta_{\star}$ a local optimizer.

In the figure below we see examples of functions with (left) one global minimum, (middle) infinitely many global minima, and (right) multiple local as well as one global minimum.

<div style="text-align:center">
    <img src="https://i.imgur.com/kw9yqs6.png" alt="drawing" width="500"/>
</div>

(Source: [mlstory](https://mlstory.org/optimization.html))

### Convexity
Convexity is a property of a function and has a very clear geometric meaning. 

If the function $J$ fulfills the following inequality

$$J(\alpha \vartheta_l+(1-\alpha)\vartheta_r) \le \alpha J(\vartheta_l)+(1-\alpha)J(\vartheta_r), \quad \forall \vartheta_l, \vartheta_r \text{ and } \alpha \in [0,1],$$

then $J$ is said to be convex. Geometrically, this inequality implies that a line segment between $(\vartheta_l, J(\vartheta_l))$ and $(\vartheta_r, J(\vartheta_r))$ lies above the graph of $J$ in the range $(\vartheta_l, \vartheta_r)$.

<div style="text-align:center">
    <img src="https://mlstory.org/assets/convex.svg" alt="drawing" width="350"/>
</div>

(Source: [mlstory](https://mlstory.org/optimization.html))

There is exhaustive literature on convex functions and convex optimization, e.g. [Boyd & Vandenberghe 2004](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf), due to the mathematical properties of such functions. An important result from this theory is that gradient descent is guaranteed to find an optimum of convex functions.

Two examples of convex functions are the Least Mean Square loss in linear regression and the negative log-likelihood in logistic regression. However, modern deep learning is in general non-convex. Thus, we optimize towards a local minimum in the proximity of an initial configuration. 

> Note: Convexity is a property of the loss $J(\vartheta)$ w.r.t. $\vartheta$, i.e. it is now about the convexity of the loss w.r.t. the output of the function $h(x)$. This means, for $J(\vartheta)$ to be convex, the combination of model and loss functions has to result in a convex function in $\vartheta$.

<!-- If we assume that $h(x)$ is a very complicated non-convex function, then we could assume that the choice of loss function should not matter too much. 

 -->

Apparently, the choice of $h(x)$ plays an important role on the shape of $J(\vartheta)$, but how does the choice of loss function influence $J$?

### Cost Functions

Some of the most common loss functions include:
- Regression loss
    - L1 loss: $\; \; \; J(h_{\vartheta}(x), y)=1/m \sum_{i=1}^m |y-h_{\vartheta}(x)|$
    - MSE loss: $J(h_{\vartheta}(x), y)=1/m \sum_{i=1}^m (y-h_{\vartheta}(x))^2$

- Classification loss
    - Cross Entropy loss $J(h_{\vartheta}(x),y)= -\sum_{i=1}^m\sum_{k=1}^K(y_{ik}\cdot \log h_{\vartheta, k}(x_i))$

#### $l_p$ norm
To better understand the regression losses, we will look at the general $l_p$ norm of vector $w\in \mathbb{R}^m$:


$$||w||_p=\left(\sum_{i=1}^m |w_i|^p\right)^{1/p} \quad \text{for } p \ge 1.$$

In the special case $p=1$ we recover the L1 loss, and the squared version of $p=2$ corresponds to the MSE loss. Other special case is $p \to \infty$ leading to $||w||_{\infty}= \max \left\{ |w_1|,|w_2|,...,|w_{m}| \right\}$. We see that with increasing $p$ the larger terms dominate


<div style="text-align:center">
    <img src="https://i.imgur.com/Z05qdjO.png" alt="drawing" width="600"/>
</div>

*Figure*: The blue line represents the solution set of a under-determined system of equations. The red line represents the minimum-norm level sets that intersect the blue line for each norm. For norms $p=0,...,1$, the minimum-norm solution corresponds to the sparsest solution with only one coordinate active. For $p \ge 2$ the minimum-norm solution is not sparse, but all coordinates are active.

(Source: [Brunton and Kutz 2019](https://www.cambridge.org/core/books/datadriven-science-and-engineering/77D52B171B60A496EAFE4DB662ADC36E), Fig. 3.9)


## Gradient-based Methods

If we now consider the following, highly simplified, objective functions which we would now seek to optimize

<div style="text-align:center">
    <img src="https://i.imgur.com/kw9yqs6.png" alt="drawing" width="500"/>
</div>

(Source: [mlstory](https://mlstory.org/optimization.html))



then we see that to find the global/local minimum gradient descent, an approach which we already encountered in high-school curve analysis, is best suited. While being the obvious choice for the two cases on the left, the picture becomes a little more muddy in the example on the right. 

> Notation alert: For the derivation of the gradien-based optimization techniques we use the stands notation by which the function we want to find the minimum of becomes $J \rightarrow f$ and the variable $\vartheta \rightarrow x$. Don't confuse this $x$ with the input measurements.

### Gradient Descent

While the foundational concept, gradient descent is rarely used in its pure form, but mostly in its stochastic form these days. If we first consider it in its most foundational form in 1-dimension, then we can take a function $f$, and Taylor-expand it

$$f(x+\varepsilon) = f(x) + \varepsilon f'(x) + \mathcal{O}(\varepsilon^{2})$$

then our intuition would dictate that moving a small $\varepsilon$ in the direction of the negative gradient will then decrease f. Taking a step size $\eta > 0$, and using using our ability to freely choose $\varepsilon$ to set it as

$$\varepsilon = - \eta f'(x)$$

To then be plugged back into the Taylor expansion

$$f(x - \eta f'(x)) = f(x) - \eta f'^{2}(x) + \mathcal{O}(\eta^{2}f'^{2}(x))$$

Unless our gradient vanishes, we can then minimize $f$ as $\eta f'^{2}(x)>0$. Choosing a small enough $\eta$ can then make the higher-order terms irrelevant to arrive at

$$f(x - \eta f'(x)) \leq f(x)$$

I.e.

$$x \leftarrow x - \eta f'(x)$$

is the right algorithm to iterative over $x$ s.t. the value of our (objective) function $f(x)$ declines. The algorithm we hence end up with an algorithm in which we have to choose an initial value for $x$, a constant $\eta > 0$, and then continuously iterate $x$ until we reach our stopping criterion.

$\eta$ is most commonly known as our _learning rate_ and has to be set by us at the current moment of time. Now, if $\eta$ is too small $x$ will update too slowly and requiring us to perform many more costly iterations than we'd ideally like to. But if we choose a learning which is too large, then the error term $\mathcal{O}(\eta^{2}f'^{2}(x))$ at the back of the Taylor-expansion will explode, and we will overshoot the minimum.

Now, if we take a non-convex function for $f$ which might even have infinitely many local minima, then the choice of our learning rate and initialization becomes even more important. Take the following function $f$ for example.

$$f(x) = x \cdot \cos(cx)$$

then the optimization problem might end up looking like the following:


<div style="text-align:center">
    <img src="https://i.imgur.com/H2TflT5.png" alt="drawing" width="400"/>
</div>

(Source: [classic.d2l.ai](https://classic.d2l.ai/chapter_optimization/gd.html))


If we now consider the case where we do not only have a one-dimensional (objective) function, but instead a function f s.t.

$$f: \mathbb{R}^{d} \rightarrow \mathbb{R}$$

vector are mapped to scalars, then the gradient is a vector of $d$ partial derivatives

$$\nabla f({\bf{x}}) = \left[ \frac{\partial f({\bf{x}})}{\partial x_{1}}, \frac{\partial f({\bf{x}})}{\partial x_{2}}, \ldots, \frac{\partial f({\bf{x}})}{\partial x_{d}} \right]^{\top}$$

with each gradient indicating the rate of change in one of the many potential directions. Then we can use the Taylor-approximation as before, and derive the gradient descent algorithm for the multivariate case

$${\bf{x}} \leftarrow {\bf{x}} - \eta \nabla f({\bf{x}})$$

If we then construct an objective function such as

$$f({\bf{x}}) = x_{1}^{2} + 2 x_{2}^{2}$$

then our optimization will take the following shape


<div style="text-align:center">
    <img src="https://i.imgur.com/Sr833L4.png" alt="drawing" width="400"/>
</div>

(Source: [classic.d2l.ai](https://classic.d2l.ai/chapter_optimization/gd.html))

Having up until now relied on a fixed learning rate $\eta$, we now want to expand upon the previous algorithm by _adaptively_ choosing $\eta$. For this we have to go back to **Newton's method**.

For this we have to further expand the initial Taylor-expansion to the third-order term

$$f({\bf{x}} + {\bf{\varepsilon}}) = f({\bf{x}}) + {\bf{\varepsilon}}^{\top} \nabla f({\bf{x}}) + \frac{1}{2} {\bf{\varepsilon}}^{\top} \nabla^{2} f({\bf{x}}) {\bf{\varepsilon}} + \mathcal{O}(||{\bf{\varepsilon}}||^{3})$$

If we now look closer at $\nabla^{2} f({\bf{x}})$, sometimes also called the Hessian, then we recognize that for larger problems this term might be infeasible to compute due to the required $\mathcal{O}(d^{2})$ computations. 

Following the same approach as before to calculate $\varepsilon$, we then arrive at ideal value of 

$${\bf{\varepsilon}} = - (\nabla^{2} f({\bf{x}}))^{-1} \nabla f({\bf{x}})$$

I.e. we need to invert $\nabla^{2} f({\bf{x}})$, the Hessian. Computing and storing this important array turns out to be really expensive! To reduce these costs we are looking towards the _preconditioning_ of our optimization algorithm. Preconditioning we then only need to compute the diagonal entries, hence leading to the following update equation

$${\bf{x}} \leftarrow {\bf{x}} - \eta \text{diag }(\nabla^{2} f({\bf{x}}))^{-1} \nabla f({\bf{x}})$$

What the preconditioning then achieves is to in essence select a specific learning rule for every single variable.

#### Momentum

If we now have a mismatch in the scales of two variables contained in our objective function, then we end up with an unsolvable optimization problem. To solve it, we require the _momentum method_

$${\bf{v}}_{t} \leftarrow \beta {\bf{v}}_{t-1} + {\bf{g}}_{t, t-1}$$
$${\bf{x}}_{t} \leftarrow {\bf{x}}_{t-1} - \eta_{t} {\bf{v}}_{t} $$

for $\beta=0$, we then have the regular gradient descent update. To now choose the perfect effective sample weight, we have to take the limit of 

$${\bf{v}}_{t}\ = \sum_{\tau = 0}^{t-1} \beta^{\tau} {\bf{g}}_{t-\tau, t-\tau-1}$$

Taking the limit

$$\sum_{\tau=0}^{\infty} \beta^{\tau} = \frac{1}{1 - \beta}$$

Hence $\frac{\eta}{1 - \beta}$ is the perfect step size, which at the same time gives us much better gradient descent directions to follow to minimize our objective function.


### Adam

The [Adam algorithm](https://arxiv.org/abs/1412.6980), then extends beyond traditional gradient descent by combining multiple tricks into a highly robust algorith, which is one of the most well-used optimization algorithms in machine learning.

Expanding upon the previous use of momentum, Adam further utilizes the 1st, and 2nd momentum of the gradient i.e.

$${\bf{v}}_{t} \leftarrow \beta_{1} {\bf{v}}_{t-1} + (1 - \beta_{1}) {\bf{g}}_{t}$$
$${\bf{s}}_{t} \leftarrow \beta_{2} {\bf{s}}_{t-1} - (1 - \beta_{2}) {\bf{g}}_{t}^{2}$$

where both $\beta_{1}$, and $\beta_{2}$ are non-negative. A typical initialization here would be something along the lines of $\beta_{1} = 0.9$, and $\beta_{2} = 0.999$ s.t. the variance estimate moves much slower than the momentum term. As an initialization of ${\bf{v}}_{0} = {\bf{s}}_{0} = 0$ can lead to bias in the optimization algorithm, we have to re-normalize the state variables with

$${\hat{\bf{v}}}_{t} = \frac{{\bf{v}}_{t}}{1 - \beta_{1}^{t}} \text{  ,and  } {\hat{\bf{s}}}_{t} = \frac{{\bf{s}}_{t}}{1 - \beta_{2}^{t}}$$

The Adam optimization algorithm furthermore rescales the gradient to obtain

$${\bf{g}}'_{t} = \frac{\eta {\hat{\bf{v}}}_{t}}{\sqrt{{\hat{\bf{s}}}_{t}} + \varepsilon} $$

The update formula for Adam is then

$${\bf{x}}_{t} \leftarrow {\bf{x}}_{t-1} - {\bf{g}}'_{t}. $$

> The strength of the Adam optimization algorithm is the stability of its update rule.


### Stochastic Gradient Descent

In machine learning we often resort to taking the loss across an average of the entire training set. Writing down the objective function for the training set with $n$ entries

$$f({\bf{x}}) = \frac{1}{n} \sum_{i=1}^{n}f_{i}({\bf{x}})$$

The gradient of the objective function is then

$$\nabla f({\bf{x}}) = \frac{1}{n} \sum_{i=1}^{n} \nabla f_{i}({\bf{x}})$$

With the cost of each independent variable iteration being $\mathcal{O}(n)$ for gradient descent, stochastic gradient replaces this with a sampling step where we uniformly sample an index $i \in \{1, \ldots, n\}$ at random, and then compute the gradient for the sampled index, and update ${\bf{x}}$

$${\bf{x}} \leftarrow {\bf{x}} - \eta \nabla f_{i}({\bf{x}})$$

With this randomly sampled update the cost for each iteration drops to $\mathcal{O}(1)$. Due to the sampling we now have to think of our gradient as an expectation, i.e. by drawing uniform random samples we are essentially creating an unbiased estimator of the gradient

$$\mathbb{E}_{i} \nabla f_{i}({\bf{x}}) = \frac{1}{n} \sum_{i=1}^{n} \nabla f_{i}({\bf{x}}) = \nabla f({\bf{x}})$$

Looking at an example stochastic gradient descent optimization process

<div style="text-align:center">
    <img src="https://i.imgur.com/np1tRgn.png" alt="drawing" width="400"/>
</div>

(Source: [classic.d2l.ai](https://classic.d2l.ai/chapter_optimization/gd.html))
 
 
we come to realize that the stochasticity induces too much noise for our chosen learning rate. As such we want to be able to dynamically adjust the learning rate $\eta$ for a time-dependent learning rate $\eta(t)$ to then control the rate of decay of $\eta$. The most common strategies are

$$\eta(t) = \eta_{i} \text{ if } t_{i} \leq t \leq t_{i+1}, \quad \text{piecewise constant}$$
$$\eta(t) = \eta_{0} \cdot e^{-\lambda t}, \quad \text{ exponential decay}$$
$$\eta(t) = \eta_{0} \cdot \left( \beta t + 1 \right)^{- \alpha}, \quad \text{ polynomial decay}$$

Going through the different proposed options in order:

- Piecewise constant: Decrease whenever optimization progress begins to stall.
- Exponential decay: Much more aggressive, can lead to premature stopping.
- Polynomial decay: Well-behaved when $\alpha = 0.5$.

Corrected for the time-dependent learning rate, and using the exponential decay our optimization then takes the following shape:

<div style="text-align:center">
    <img src="https://i.imgur.com/UJ3J86r.png" alt="drawing" width="400"/>
</div>

(Source: [classic.d2l.ai](https://classic.d2l.ai/chapter_optimization/gd.html))


Which is much much nicer behaved!


#### Minibatching

> For data which is very similar, gradient descent is inefficient, whereas stochastic gradient descent relies on the power of vectorization.

The answer to these ailments is the use of minibatches to exploit the memory and cache hierarchy a modern computer exposes to us. In essence we seek to avoid the many single matrix-vector multiplications to reduce the overhead and improve our computational cost. The update equation then becomes

$${\bf{g}}_{t} = \partial_{\omega}f({\bf{x}}_{t}, {\bf{\omega}})$$

For computational efficiency we will now perform this update in its batched form

$${\bf{g}}_{t} = \partial_{\omega} \frac{1}{|\mathcal{B}_{t}|} \sum_{i \in \mathcal{B}_{t}} f({\bf{x}}_{i}, {\bf{\omega}})$$

As both ${\bf{g}}_{t}$, and ${\bf{x}}_{t}$ are drawn uniformly at random from the training set, we retain our unbiased gradient estimator. For size $b$ of the dataset, i.e $b = | \mathcal{B}_{t} |$ we obtain a reduction of the standard deviation by $b^{-\frac{1}{2}}$, while this is desirable we should in practice choose the minibatch-size s.t. our underlying hardware gets utilized as optimally as possible.



## Further References

**Gradient-Based Optimization**

- [Patterns, Predictions, and Actions](https://mlstory.org/optimization.html), Chapter 5. Optimization; M. Hardt and B. Recht; 2022
- [Dive into Deep Learning](https://d2l.ai/chapter_optimization/index.html), Chapter 12. Optimization Algorithms
