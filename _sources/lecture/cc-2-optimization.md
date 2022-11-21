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

After selecting a model $h_{\vartheta}$, points 2 and 3 are critical to the success of the learning as the loss function (point 2) defines how we measure success and the optimizer (point 3) guides the process of moving from a random initial guess of the parameters to a parameter configuration with a much smaller loss. These two points are the topic of Core Content 2: how do the loss influence training and which optimization methods exist?

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

Apparently, the choice of $h(x)$ plays an important role in the shape of $J(\vartheta)$, but how does the choice of loss function influence $J$?

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

*Figure*: The blue line represents the solution set of an under-determined system of equations. The red line represents the minimum-norm level sets that intersect the blue line for each norm. For norms $p=0,...,1$, the minimum-norm solution corresponds to the sparsest solution with only one coordinate active. For $p \ge 2$ the minimum-norm solution is not sparse, but all coordinates are active.

(Source: [Brunton and Kutz 2019](https://www.cambridge.org/core/books/datadriven-science-and-engineering/77D52B171B60A496EAFE4DB662ADC36E), Fig. 3.9)


## Gradient-based Methods

If we now consider the following, highly simplified, objective functions which we would now seek to optimize

<div style="text-align:center">
    <img src="https://i.imgur.com/kw9yqs6.png" alt="drawing" width="500"/>
</div>

(Source: [mlstory](https://mlstory.org/optimization.html))



then we see that to find the global/local minimum gradient descent, an approach that we already encountered in high-school curve analysis, is best suited. While being the obvious choice for the two cases on the left, the picture becomes a little muddier in the example on the right. 

```{note}
Notation alert: For the derivation of the gradien-based optimization techniques we use the stands notation by which the function we want to find the minimum of becomes $J \rightarrow f$ and the variable $\vartheta \rightarrow x$. Don't confuse this $x$ with the input measurements $\{x^{(i)},y^{(i)}\}_{i=0,...,m}$.
```

### Gradient Descent

While the foundational concept, gradient descent is rarely used in its pure form, but mostly in its stochastic form these days. If we first consider it in its most foundational form in 1-dimension, then we can take a function $f$, and Taylor-expand it

$$f(x+\varepsilon) = f(x) + \varepsilon f'(x) + \mathcal{O}(\varepsilon^{2})$$

then our intuition would dictate that moving a small $\varepsilon$ in the direction of the negative gradient will then decrease f. Taking a step size $\eta > 0$, and using using our ability to freely choose $\varepsilon$ to set it as

$$\varepsilon = - \eta f'(x)$$

to then be plugged back into the Taylor expansion

$$f(x - \eta f'(x)) = f(x) - \eta f'^{2}(x) + \mathcal{O}(\eta^{2}f'^{2}(x))$$

Unless our gradient vanishes, we can then minimize $f$ as $\eta f'^{2}(x)>0$. Choosing a small enough $\eta$ can then make the higher-order terms irrelevant to arrive at

$$f(x - \eta f'(x)) \leq f(x)$$

I.e.

$$x \leftarrow x - \eta f'(x)$$

is the right algorithm to iterative over $x$ s.t. the value of our (objective) function $f(x)$ declines. The algorithm we hence end up with an algorithm in which we have to choose an initial value for $x$, a constant $\eta > 0$, and then continuously iterate $x$ until we reach our stopping criterion.

$\eta$ is most commonly known as our _learning rate_ and has to be set by us at the current moment of time. Now, if $\eta$ is too small $x$ will update too slowly and require us to perform many more costly iterations than we'd ideally like to. But if we choose a learning rate that is too large, then the error term $\mathcal{O}(\eta^{2}f'^{2}(x))$ at the back of the Taylor-expansion will explode, and we will overshoot the minimum.

Now, if we take a non-convex function for $f$ which might even have infinitely many local minima, then the choice of our learning rate and initialization becomes even more important. Take the following function $f$ for example.

$$f(x) = x \cdot \cos(cx)$$

then the optimization problem might end up looking like the following:


<div style="text-align:center">
    <img src="https://i.imgur.com/H2TflT5.png" alt="drawing" width="400"/>
</div>

(Source: [classic.d2l.ai](https://classic.d2l.ai/chapter_optimization/gd.html))


If we now consider the case where we do not only have a one-dimensional (objective) function but instead a function f s.t.

$$f: \mathbb{R}^{d} \rightarrow \mathbb{R}$$

vector are mapped to scalars, then the gradient is a vector of $d$ partial derivatives

$$\nabla f({\bf{x}}) = \left[ \frac{\partial f({\bf{x}})}{\partial x_{1}}, \frac{\partial f({\bf{x}})}{\partial x_{2}}, \ldots, \frac{\partial f({\bf{x}})}{\partial x_{d}} \right]^{\top}$$

with each gradient indicating the rate of change in one of the many potential directions. Then we can use the Taylor-approximation as before and derive the gradient descent algorithm for the multivariate case

$${\bf{x}} \leftarrow {\bf{x}} - \eta \nabla f({\bf{x}})$$

If we then construct an objective function such as

$$f({\bf{x}}) = x_{1}^{2} + 2 x_{2}^{2}$$

then our optimization will take the following shape


<div style="text-align:center">
    <img src="https://i.imgur.com/Sr833L4.png" alt="drawing" width="400"/>
</div>

(Source: [classic.d2l.ai](https://classic.d2l.ai/chapter_optimization/gd.html))

Having up until now relied on a fixed learning rate $\eta$, we now want to expand upon the previous algorithm by _adaptively_ choosing $\eta$. For this, we have to go back to **Newton's method**.

For this, we have to further expand the initial Taylor-expansion to the third-order term

$$f({\bf{x}} + {\bf{\varepsilon}}) = f({\bf{x}}) + {\bf{\varepsilon}}^{\top} \nabla f({\bf{x}}) + \frac{1}{2} {\bf{\varepsilon}}^{\top} \nabla^{2} f({\bf{x}}) {\bf{\varepsilon}} + \mathcal{O}(||{\bf{\varepsilon}}||^{3})$$

If we now look closer at $\nabla^{2} f({\bf{x}})$, sometimes also called the Hessian, then we recognize that for larger problems this term might be infeasible to compute due to the required $\mathcal{O}(d^{2})$ computations. 

Following the condition $\nabla_{\epsilon} f(\bf{x} + \epsilon)=0$ for the minimum to calculate $\varepsilon$, we then arrive at the ideal value of 

$${\bf{\varepsilon}} = - (\nabla^{2} f({\bf{x}}))^{-1} \nabla f({\bf{x}})$$

I.e. we need to invert $\nabla^{2} f({\bf{x}})$, the Hessian. Computing and storing this important array turns out to be really expensive! To reduce these costs we are looking towards the _preconditioning_ of our optimization algorithm. Preconditioning we then only need to compute the diagonal entries, hence leading to the following update equation

$${\bf{x}} \leftarrow {\bf{x}} - \eta \text{diag }(\nabla^{2} f({\bf{x}}))^{-1} \nabla f({\bf{x}})$$

What the preconditioning then achieves is to in essence select a specific learning rule for every single variable.

#### Momentum

If we now have a mismatch in the scales of two variables contained in our objective function, then we end up with an unsolvable optimization problem. To solve it, we require the _momentum method_

$${\bf{v}}_{t} \leftarrow \beta {\bf{v}}_{t-1} + {\bf{g}}_{t}$$
$${\bf{x}}_{t} \leftarrow {\bf{x}}_{t-1} - \eta_{t} {\bf{v}}_{t} $$

with gradients of the loss $\mathbf{g}_t = \nabla f_t(x)$ and the momentum variable $\mathbf{v}_t$. The momentum $\mathbf{v}_t$ accumulates past gradients resembling how a ball rolling down a hill integrates over past forces. For $\beta=0$, we then have the regular gradient descent update. To now choose the perfect effective sample weight, we have to take the limit of 

$${\bf{v}}_{t}\ = \sum_{\tau = 0}^{t-1} \beta^{\tau} {\bf{g}}_{t-\tau}$$

Taking the limit $t \to \infty$ results in the [geometric series](https://en.wikipedia.org/wiki/Geometric_series) solution

$$\sum_{\tau=0}^{\infty} \beta^{\tau} = \frac{1}{1 - \beta}$$

Hence using the momentum GD results in a step size $\frac{\eta}{1 - \beta}$, which at the same time gives us much better gradient descent directions to follow to minimize our objective function.

<div style="text-align:center">
    <img src="../imgs/output_momentum.svg" alt="drawing" width="400"/>
</div>

(Source: [d2l.ai](https://d2l.ai/chapter_optimization/momentum.html))
### Adam

The [Adam algorithm](https://arxiv.org/abs/1412.6980), then extends beyond traditional gradient descent by combining multiple tricks into a highly robust algorithm, which is one of the most well-used optimization algorithms in machine learning.

Expanding upon the previous use of momentum, Adam further utilizes the 1st and 2nd momentum of the gradient i.e.

$${\bf{v}}_{t} \leftarrow \beta_{1} {\bf{v}}_{t-1} + (1 - \beta_{1}) {\bf{g}}_{t}$$
$${\bf{s}}_{t} \leftarrow \beta_{2} {\bf{s}}_{t-1} - (1 - \beta_{2}) {\bf{g}}_{t}^{2}$$

where both $\beta_{1}$, and $\beta_{2}$ are non-negative. A typical initialization here would be something along the lines of $\beta_{1} = 0.9$, and $\beta_{2} = 0.999$ s.t. the variance estimate moves much slower than the momentum term. As an initialization of ${\bf{v}}_{0} = {\bf{s}}_{0} = 0$ can lead to bias in the optimization algorithm towards small initial values, we have to re-normalize the state variables with

$${\hat{\bf{v}}}_{t} = \frac{{\bf{v}}_{t}}{1 - \beta_{1}^{t}} \text{  ,and  } {\hat{\bf{s}}}_{t} = \frac{{\bf{s}}_{t}}{1 - \beta_{2}^{t}}$$

The Adam optimization algorithm furthermore rescales the gradient to obtain

$${\bf{g}}'_{t} = \frac{\eta {\hat{\bf{v}}}_{t}}{\sqrt{{\hat{\bf{s}}}_{t}} + \varepsilon} $$

The update formula for Adam is then

$${\bf{x}}_{t} \leftarrow {\bf{x}}_{t-1} - {\bf{g}}'_{t}. $$

> The strength of the Adam optimization algorithm is the stability of its update rule.

If the momentum GD algorithm can be understood as a ball rolling down a hill, the Adam algorithm behaves like a ball with friction (see [GANs Trained by a Two Time-Scale Update Rule
Converge to a Local Nash Equilibrium](https://proceedings.neurips.cc/paper/2017/file/8a1d694707eb0fefe65871369074926d-Paper.pdf)).


![Optimization](https://github.com/Jaewan-Yun/optimizer-visualization/blob/master/figures/movie11.gif)
(Source: [github.com/Jaewan-Yun/optimizer-visualization](https://github.com/Jaewan-Yun/optimizer-visualization))


### Stochastic Gradient Descent

In machine learning, we often resort to taking the loss across an average of the entire training set. Writing down the objective function for the training set with $n$ entries

$$f({\bf{x}}) = \frac{1}{n} \sum_{i=1}^{n}f_{i}({\bf{x}})$$

The gradient of the objective function is then

$$\nabla f({\bf{x}}) = \frac{1}{n} \sum_{i=1}^{n} \nabla f_{i}({\bf{x}})$$

With the cost of each independent variable iteration being $\mathcal{O}(n)$ for gradient descent, stochastic gradient replaces this with a sampling step where we uniformly sample an index $i \in \{1, \ldots, n\}$ at random, and then compute the gradient for the sampled index, and update ${\bf{x}}$

$${\bf{x}} \leftarrow {\bf{x}} - \eta \nabla f_{i}({\bf{x}})$$

With this randomly sampled update, the cost for each iteration drops to $\mathcal{O}(1)$. Due to the sampling we now have to think of our gradient as an expectation, i.e. by drawing uniform random samples we are essentially creating an unbiased estimator of the gradient

$$\mathbb{E}_{i} \nabla f_{i}({\bf{x}}) = \frac{1}{n} \sum_{i=1}^{n} \nabla f_{i}({\bf{x}}) = \nabla f({\bf{x}})$$

Looking at an example stochastic gradient descent optimization process

<div style="text-align:center">
    <img src="https://i.imgur.com/np1tRgn.png" alt="drawing" width="400"/>
</div>

(Source: [classic.d2l.ai](https://classic.d2l.ai/chapter_optimization/gd.html))
 
 
we come to realize that the stochasticity induces too much noise for our chosen learning rate. To handle that we introduce learning rate scheduling.

#### Minibatching

> For data which is very similar, gradient descent is inefficient, whereas stochastic gradient descent relies on the power of vectorization.

The answer to these ailments is the use of minibatches to exploit the memory and cache hierarchy a modern computer exposes to us. In essence, we seek to avoid the many single matrix-vector multiplications to reduce the overhead and improve our computational cost. The update equation then becomes

$${\bf{g}}_{t} = \partial_{\omega}f({\bf{x}}_{t}, {\bf{\omega}})$$

For computational efficiency, we will now perform this update in its batched form

$${\bf{g}}_{t} = \partial_{\omega} \frac{1}{|\mathcal{B}_{t}|} \sum_{i \in \mathcal{B}_{t}} f({\bf{x}}_{i}, {\bf{\omega}})$$

As both ${\bf{g}}_{t}$, and ${\bf{x}}_{t}$ are drawn uniformly at random from the training set, we retain our unbiased gradient estimator. For size $b$ of the dataset, i.e $b = | \mathcal{B}_{t} |$ we obtain a reduction of the standard deviation by $b^{-\frac{1}{2}}$, while this is desirable we should in practice choose the minibatch-size s.t. our underlying hardware gets utilized as optimally as possible.




### Second-Order Methods

With first-order methods only taking the gradient itself into account there is much information we are leaving untapped in attempting to solve our optimization problem, such as gradient curvature for which we'd need 2nd order gradients.

The reason for that is in part historic, automatic differentiation (to be explained in-depth in a later lecture) suffers from an exponential compute-graph blow-up when computing higher-order gradients. The automatic differentiation engines the first popularized machine learning engines from AlexNet, LeNet etc. were built on were unable to handle higher-order gradients for the above reason, and only modern automatic differentiation engines have been able to circumvent that problem. As such 2nd-order gradient methods have seen a recent resurgence with methods like [Shampoo](https://research.google/pubs/pub47079/), [RePAST](https://arxiv.org/pdf/2210.15255.pdf), and [Fishy](https://openreview.net/forum?id=cScb-RrBQC).


#### Newton's method (aka Newton-Raphson method)

A first example of the step toward second-order methods is the Newton method. It is an iterative method to find a zero of a differentiable function defined as $f: \mathbb{R} \rightarrow \mathbb{R}$. The Newton-method begins with an initial guess $x_{0}$ to then iteratively update with 

$$x_{t+1} = x_{t} - \frac{f(x_{t})}{f'(x_{t})}$$

Graphically speaking this amounts to the case of the tangent line of the function intersecting with the x-axis.

<div style="text-align:center">
    <img src="https://i.imgur.com/uyKmjj3.png" alt="drawing" width="350"/>
</div>

The fallacies of this method are the cases where $f'(x_{t}) = 0$, and it will diverge when $|f'(x_{t})|$ is very small. Going beyond the first-order update we can then utilize the second-order gradient if our (objective) function $f$ is twice differentiable. Hence turning the update step into

$$x_{t+1} = x_{t} - \frac{f'(x_{t})}{f''(x_{t})}, \quad t>0$$

Generalizing this update scheme in notation, we can use vector calculus to obtain

$$x_{t+1} = x_{t} - \nabla^{2} f(x_{t})^{-1} \nabla f(x_{t}), \quad t\geq 0$$

$\nabla^{2} f(x_{t})^{-1}$ constitutes the Hessian at $x_{t}$. As alluded to earlier the failure modes can then be formalized in the following fashion:

- The Hessian is not invertible.
- Gets out of control if the Hessian has a small norm.

If we further generalize the notation to a general update scheme

$$x_{t+1} = x_{t} - H(x_{t}) \nabla f(x_{t})$$

then we can simplify this update scheme to the gradient descent we already encountered in the last lecture by setting $H(x_{t}) = \gamma I$, where $I$ is the identity matrix. Newton's method hence constitutes an adaptive gradient descent approach, where the adaptation happens with respect to the local geometry of the objective function at $x_{t}$.

> Where gradient descent requires the right step-size, Newton's method converges naturally to the local minimum without the requirement of step-size tuning.

To expand upon this, assume we have a linear system of equations

$$M {\bf{x}} = {\bf{q}}$$

then Newton's method can solve this system in **one** step whereas gradient descent requires multiple steps with the right step size chosen. The downside to this is the expensive step of inverting matrix $M$. In our general case, this means we need to invert the expensive matrix

$$\nabla^{2} f(x_{0}).$$

Another advantage of Newton's method is that it does not suffer from individual coordinates being at completely different scales, e.g. the $y$-direction changes very fast, whereas the $z$-direction only changes very slowly. Gradient descent only handles these cases suboptimally, whereas Newton's method does not suffer from this shortcoming.

> Note: The Hessian has to be positive definite, otherwise we would end up in a local maximum.

A matrix $H$ is positive definite if the real number $z^{\top} H z$ is positive for every nonzero real-valued vector $z$.


#### The Quasi-Newton Approach

With the main computational bottleneck of Newton's approach being that the inversion of the Hessian matrix costs $\mathcal{O}(d^{3})$ for a matrix of size $d \times d$, there exist multiple approaches which try to circumvent this costly operation.

##### The Secant Method

The secant method is an alternative to Newton's method, which consciously does not use derivatives and has hence much less stringent requirements on our objective function such as it not needing to be differentiable. Essentially taking a finite difference approximation

$$\frac{f(x_{t}) - f(x_{t-1})}{x_{t} - x_{t-1}}$$

then we can replace the derivative in Newton's method with the finite difference approximation, i.e.

$$\frac{f(x_{t}) - f(x_{t-1})}{x_{t} - x_{t-1}} \approx f'(x_{t})$$

for a small region around $x_{t}$. The secant update step then takes the following form

$$x_{t+1} = x_{t} - f(x_{t})\frac{x_{t} - x_{t-1}}{f(x_{t}) - f(x_{t-1})}, \quad t\geq 1$$

to approximate the Newton step at the detriment of having to choose two starting values $x_{0}$, and $x_{1}$ here. Figuratively speaking the approach looks like this:

<div style="text-align:center">
    <img src="https://i.imgur.com/TrCYl2W.png" alt="drawing" width="350"/>
</div>

What this approach then does is to construct the line through the two points $(x_{t-1}, f(x_{t-1}))$, and $(x_{t}, f(x_{t}))$ on the graph of f, the next iteration is then given by the point where the line intersects the x-axis.

When our function is first-order differentiable, we can also use the secant method to derive a second-derivative-free version of Newton's method for optimization.

$$x_{t+1}=x_{t} - f'(x_{t})\frac{x_{t} - x_{t-1}}{f'(x_{t}) - f'(x_{t-1})}, \quad t \geq 1$$


##### Quasi-Newton Methods

If we consider the so-called secant condition, then we have the following approximation

$$H_{t} = \frac{f'(x_{t}) - f'(x_{t-1})}{x_{t} - x_{t-1}} \approx f''(x_{t})$$

then the secant method works with the following update step using the above approximation

$$x_{t+1} = x_{t} - H_{t}^{-1}f'(x_{t}), \quad t \geq 1.$$

For this approximation to hold we need to satisfy the secant condition 

$$f'(x_{t}) - f'(x_{t-1}) = H_{t}(x_{t} - x_{t-1})$$

Or generalized to higher dimensions

$$\nabla f({\bf{x_{t}}}) - \nabla f({\bf{x_{t-1}}}) = H_{t}({\bf{x_{t}}} - {\bf{x_{t-1}}})$$


Whenever this condition is now fulfilled in conjunction with a _symmetric matrix_, then we have a **Quasi-Newton method**. As our matrix $H_{t} \approx \nabla^{2}f(x_{t})$ only fluctuates very little during periods of very fast convergence, and Newton's method is optimal with one step, then we can presume

$$H_{t} \approx H_{t-1}$$

and equally as much 

$$H_{t}^{-1} \approx H_{t-1}^{-1}$$

then we can following Greenstadt's approach model $H_{t}$ in the following fashion

$$H_{t}^{-1} = H^{-1}_{t-1} + E_{t}.$$

I.e. our matrix $H_{t}$, the Hessian, only changes by a minor error matrix. These errors should also be as small as possible.


##### BFGS
(Broyden–Fletcher–Goldfarb–Shanno)

For the BFGS algorithm, this update matrix then assumes the form

$$E = \frac{1}{{\bf{y}}^{\top} {\bf{\sigma}}} \left( -H {\bf{y}} {\bf{\sigma}}^{\top} - {\bf{\sigma}} {\bf{y}}^{\top} H + (1 + \frac{{\bf{y}}^{\top}H{\bf{y}}}{{\bf{y}}^{\top} {\bf{\sigma}}}) {\bf{\sigma}} {\bf{\sigma}}^{\top} \right)$$

where $H=H_{t-1}^{-1}$, ${\bf{\sigma}} = x_{t} - x_{t-1}$, and $y = \nabla f(x_{t}) - \nabla f(x_{t-1})$.

One of the core advantages of BFGS is that if $H'$ is positive definite, then the update $E$ maintains this positive definite attribute and as such behaves like a proper inverse Hessian. In addition, the cost of computation drops from the original $\mathcal{O}(d^{3})$ for a matrix of size $d \times d$ to $\mathcal{O}(d^{2})$ for the BFGS approach. Scaling the update step, the individual iteration then becomes

$$x_{t+1} = x_{t} - \alpha_{t} H_{t}^{-1} \nabla f(x_{t}), \quad t \geq 1$$

where $\alpha$ can be chosen such that a line search is performed, and where $H_{t}^{-1}$ is the BFGS approximation. In a pseudo-algorithmic form this then looks the following way:

<div style="text-align:center">
    <img src="https://i.imgur.com/aEZHfdR.png" alt="drawing" width="280"/>
</div>


##### L-BFGS
(Limited-memory BFGS) 

Especially in high dimensions a cost of $\mathcal{O}(d^{2})$ may still be too prohibitive. Only using information from the past $m$ iterations for $m$ being a small value, L-BFGS then approximates the entire $H'\nabla f(x_{t})$ term. L-BFGS then builds on modeling $H'$ as

$$H' = \left( I - \frac{\sigma y^{\top}}{y^{\top}\sigma} \right) H \left( I - \frac{y \sigma^{\top}}{y^{\top}\sigma} \right) + \frac{\sigma \sigma^{\top}}{y^{\top} \sigma}$$

then we are able to utilize an oracle to compute 

$$ s= H g$$

for any vector $g$. Then $s' = H'g'$ can be computed with one oracle call and $\mathcal{O}(d)$ additional arithmetic operations, assuming that $\sigma$, and $y$ are known. We then define $\sigma$, and $y$ in the following fashion

$$\begin{align}
    \sigma_{k} &= x_{k} - x_{k-1} \\
    y_{k} &= \nabla f(x_{k}) - \nabla f(x_{k-1})
\end{align}$$

The L-BFGS algorithm is then given by 

<div style="text-align:center">
    <img src="https://i.imgur.com/W0aB3QZ.png" alt="drawing" width="300"/>
</div>

where in the case of the recursion bottoming out prematurely at a point $k=t-m$, then we pretend that we just started the computation at that point and use $H_{0}$.


## Blackbox or Derivative Free Optimization (DFO)

If we are unable to compute gradients for any reason, then we need to rely on derivative-free optimization (DFO). This is most commonly used in blackbox function optimization, or discrete optimization.

> If you as an engineer would have to optimize the design of some model given to you, but where you are not allowed to touch the code or even look at the code of the model, but only query it for outputs, then this would constitute a case of blackbox function optimization.

There exist a number of approaches for such problems, which all depend on the cost of evaluation of our function.

- Expensive function
    - Bayesian optimization
- Cheap function
    - Stochastic local search
    - Evolutionary search
    
In local search for example we replace the entire gradient update with 

$$x_{t+1} = \underset{x \in nbr(x_{t})}{\text{argmax}}$$

where $nbr$ is the neighborhood of the point $x_{t}$. This approach is also colloquially known as hill climbing, steepest ascent, or greedy search. In stochastic local search, we would then define a probability distribution over the uphill neighbors proportional to how much they improve our function and then sample at random. A second stochastic option is to start again from a different random starting point whenever we reach a local maximum. This approach is known as random restart hill climbing.

An effective strategy here is also random search, which should be the go-to baseline one attempts first when approaching a new problem. Here an iterate $x_{t+1}$ is chosen uniformly at random from the set of iterates. An alternative, which has been proven to be less efficient (see [J. Bergstra & Y. Bengio, 20212](https://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf)), is the grid search, which chooses hyperparameters equidistantly in the same range used for grid search.

<div style="text-align:center">
    <img src="../imgs/grid and random search.png" alt="drawing" width="600"/>
</div>

(Source: [Random Search for Hyper-Parameter Optimization](https://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf))

If instead of throwing away our "old" good candidates keep them in a _population_ of good candidates, then we arrive at _evolutionary algorithms_. Here we maintain a population of $K$ good candidates, which we then try to improve at each step. The advantage here is that evolutionary algorithms are embarrassingly parallel and are as such highly scalable.




## Tricks of Optimization

This is a collection of practical tools for designing and optimizing machine learning models.

```{note}
Notation alert: In this sebsection, we switch back to the notation used in Chapter 1, i.e. $x$ is the x-axis, $y$ are measurements on the y-axis, $h(x)$ is the model evaluated at $x$, and $J$ the loss.
```

**Linear Regression (revised)**

Looking back to [Chapter 1](cc-1-linear.md), the simplest linear model for $x \in \mathbb{R}$ is

$$h(x) = \vartheta_0 + \vartheta_1 \cdot x.$$

We remind the reader that the polynomial linear regression

$$h(x) = \vartheta_0 + \vartheta_1 \cdot x + ... + \vartheta_n \cdot x^n$$

also represents a linear model in terms of its parameters $\vartheta_i$ (not to be confused with the first order model for $x \in \mathbb{R}^n$). And the general linear regression could be any linear combination of x-values lifted in a predefined basis space, e.g. exponent, sine, cosine, etc.:

$$h(x)=\vartheta_0 + \vartheta_1 \cdot x + \vartheta_3 \cdot x^2 + \vartheta_3 \cdot \exp(x) + \vartheta_4 \cdot \sin(x) + \vartheta_5 \cdot \tanh(x) + \vartheta_6 \cdot \sqrt{x} + ...$$

**Nonlinear Regression** 

Any function that is more complicated than linear, e.g.

$$
h(x) = x^{\vartheta_0} + \max\{0, \vartheta_1 \cdot x\} + ...
$$

### Overfitting vs Underfitting

Dealing with real-world data containing measurement noise, we run either in over- or underfitting, depending on the choice of such basis functions. Looking at the figure below, the left regression example corresponds to $h(x) = \vartheta_0 + \vartheta_1 \cdot x$ and the left classification example corresponds to logistic regression.

<div style="text-align:center">
    <img src="../imgs/over- and underfitting table.png" alt="drawing" width="600"/>
</div>

(Source: [Techniques for handling underfitting and overfitting in Machine Learning](https://towardsdatascience.com/techniques-for-handling-underfitting-and-overfitting-in-machine-learning-348daa2380b9))

#### Bias-Variance Tradeoff

Typically, over- and underfitting are analyzed through the lens of the bias-variance decomposition.

- **Bais Error**: Difference between the average (over infinitely many same-sized datasets coming from the same distribution) model prediction and the correct values which we are trying to predict.

- **Variance Error**: Variability of the model predictions at each position $x$ (averaged over infinitely many models trained on same-sized datasets coming from the same distribution). 

- **Irreducible Error**: Originates from the noise in the measurements. Given a corupted dataset, this error cannot be reduced with ML.

In the figure below, each point corresponds to the prediction of a model trained on a different dataset.

<div style="text-align:center">
    <img src="../imgs/bias-variance bulls-eye.png" alt="drawing" width="400"/>
</div>

(Source: [Understanding the Bias-Variance Tradeoff](http://scott.fortmann-roe.com/docs/BiasVariance.html))

Mathematically, the Bias-Variance decomposition relies on a decomposition of the expected loss. The assumption is that there is a true underlying relationship between $x$ and $y$ given by $y=\tilde{h}(x)+\epsilon$ where the noise $\epsilon$ is normally distributed, i.e. $\epsilon \sim \mathcal{N}(0,\sigma_{\epsilon})$. We try to approximate $\tilde{h}$ by our model $h$ which results in the error

$$J_{\vartheta}(x) = E\left[ (y-h_{\vartheta}(x))^2\right].$$

The error can be decomposed into its bias and variance components as

$$\begin{align}
J_{\vartheta}(x) &= \left(E[h(x)]-\tilde{h}(x)\right)^2 + E\left[(h(x)-E[h(x)])^2\right] + \sigma_{\epsilon}^2 \\
&= \text{Bias}^2 + \text{Variance} + \text{Irreducible Error} \\
\end{align}
$$

<div style="text-align:center">
    <img src="../imgs/bias-variance tradeoff vs model complexity.png" alt="drawing" width="400"/>
</div>

(Source: [Understanding the Bias-Variance Tradeoff](http://scott.fortmann-roe.com/docs/BiasVariance.html))

Given the true model and enough data to calibrate it, we should be able to reduce both the bias and variance terms to zero. However, working with imperfect models and limited data, we strive for an optimum in terms of model choice.

#### Advanced Topics: Double Descent

In recent years, machine learning models have been growing extremely large, e.g. GPT-3 175B parameters. An empirical observation that has been studied by [M. Belkin et al. 2019](More data can hurt for linear regression: Sample-wise
double descent) demonstrates that contrary to the theory behind the bias-variance tradeoff if the number of parameters is too overparametrized, model performance starts improving again. Indeed, for many practical applications, this regime has not been fully explored and making ML models larger seems to improve performance further, consistently with [The Bitter Lesson](http://www.incompleteideas.net/IncIdeas/BitterLesson.html) of R. Sutton.

<div style="text-align:center">
    <img src="../imgs/double descent.png" alt="drawing" width="500"/>
</div>

(Source: [Understanding the Bias-Variance Tradeoff](http://scott.fortmann-roe.com/docs/BiasVariance.html))


In contrast to linear models, almost all such functions result in a non-convex loss surface w.r.t. the parameters $\vartheta$.

### Data Splitting
To train a machine learning model, we typically split the given data $\left\{x^{(i)}, y^{\text {(i)}}\right\}_{i=1,...m}$ into three subsets.

- **Training**: Data of which we optimize the parameters of a model.
- **Validation**: Data of which we evaluate performance during hyperparameters optimization.
- **Testing**: Data on which we evaluate the performance at the very end of tuning the model and its parameters.

Given that the dataset is large enough, typical splits for training/validation/testing data are 80/10/10 up to 60/20/20. If data is very limited, we have a very different problem and we might not want to sacrifice separate data for validation. Then we would use Cross Validation, which is explained later in this chapter.

<div style="text-align:center">
    <img src="../imgs/data splitting.png" alt="drawing" width="600"/>
</div>

(Source: [Train/Test Split and Cross Validation in Python](https://towardsdatascience.com/train-test-split-and-cross-validation-in-python-80b61beca4b6))

#### Cross Validation

If we split a dataset into $K$ pieces, we could train the model $K$ times each time excluding a different subset. We could then evaluate the model performance on the test set for each of the $K$ models and by that get a good estimate of the variance of the error. If we select the model with the least error and train it further on the whole training set, then we talk about K-fold cross-validation.

### Regularization

One possibility to counteract overfitting and still have an expressive model is regularization. There are many approaches belonging to the class of regularization techniques. 

- Adding a regularization term to the loss - add an additional term penalizing large weight values:
    - **L1 regularization** - promotes sparsity: 

    $$J_{L1}(\vartheta) = J(\vartheta) + \alpha_{L1} \cdot \sum_{i=1}^{\#params} |\vartheta_i|$$

    - **L2 regularization** - takes information from all features; typical choice: 

    $$J_{L2}(\vartheta) = J(\vartheta) + \alpha_{L2} \cdot \sum_{i=1}^{\#params} \vartheta_i^2$$

    For a visual interpretation, look back to the $l_p$ norms.

- **Dropout**: randomly set some of the parameters to zero, e.g. 5% of all $\vartheta_i$. This way the model learns to be more redundant and at the same time improved generalization. Dropout reduces co-adaptation between terms of the model. Also, this technique is very easy to apply.
- **Eary stopping**: Stop training as soon as the validation loss starts increasing. This is also a very common and easy technique.
- etc.
    
### Input Normalization and Parameter Initialization

These two topics will be discussed in much more detail in Core Content 4 in the context of more modern deep-learning methods. For the time being, the idea behind input normalization and parameter initialization is simply to speed up the learning process, i.e. reduce the number of gradient descent iterations. 

**Case Study 1 - Input Normalization**

Imagine that the input is $x\in \mathbb{R}$ and its mean and standard deviation over the whole dataset are $1$ and $0.001$. Further, imagine that we by chance choose the true underlying linear model of the form 

$$h(x) = \vartheta_0 + \vartheta_1 \cdot x_1,$$

and the true $\vartheta^{true} = [2, 1000]$. If we start a GD optimization from an initial $\vartheta^0 = [1, 1]$, we would run into a problem. To make training work, we would need a rather small learning rate to move from the initial $\vartheta^0_0=1$ to  $\vartheta_0=2$, which would then require ~3 orders of magnitude more updates to move from $\vartheta^0_1=1$ to  $\vartheta_1=1000$. That is why it makes to normalize the inputs to a standard Gaussian distribution, e.g. $\mathcal{N}(0,1)$. 

This can be achieved by precomputing the mean $\mu=1/m \sum_{i=1}^m x^{(i)}$ and variance $\sigma^2=1/m \sum_{i=1}^m \left(x^{(i)}-\mu \right)^2$ of the inputs, and then transforming each input to 

$$\hat{x} = \frac{x-\mu}{\sigma}$$

If $x \in \mathbb{R}^n$ with $n > 1$, we would do that to each of the dimensions individually. 

In signal processing, a similar transformation is called the [whitening transformation](https://en.wikipedia.org/wiki/Whitening_transformation) - the only difference in whitening is that it also considers correlations between each of the inputs. 

> Note: The same should be done with the outputs $y$.


**Case Study 2 - Parameter Initialization**

Imagine that the input is $x\in \mathbb{R}^n$ with $n>>1$ and its mean and standard deviation over the whole dataset in each of the two dimensions are $[1, ..., 1]$ and $[1, ..., 1]$ respectively. Further, imagine that we by chance choose the true underlying linear model of the form 

$$h(x) = \vartheta_0 + \vartheta_1 \cdot x_1 + ... + \vartheta_n \cdot x_n,$$

and the true $\vartheta^{true} = [0.1, ..., 0.1]$. If we start a GD optimization from an initial $\vartheta^0 = [1,2, ..., n+1]$, we would run into a problem. To make training work, we would again need a very small learning rate to move from the initial $\vartheta^0_0=0$ to  $\vartheta_0=0.1$, which would then require ~$n$ more updates to move $\vartheta^0_n=n+1$ to  $\vartheta_1=0.1$. 

Xavier initialization has been proposed to alleviate this type of issue. It essentially starts with initial values drawn from $\mathcal{N}(0,1/n)$, i.e. a zero-centered normal distribution with variance $1/n$. This way we 

1. Choose $\vartheta^0$ in the same order of magnitude, resulting in a similar weighting of each term in the model (given that the inputs are normalized beforehand)
2. End up with an output $h(x^{(i)})$ also more or less from a standard normal distribution (up to correlation impact). And if we normalized $y$ beforehand, a $\mathcal{N}(0,1)$-distributed $h(x^{(i)})$ is what we want.



### Hyperparameter Search

A hyperparameter is a parameter that controls other parameters. We typically cannot afford to train hyperparameters with gradient-based methods and resort to DFO methods (see above). 

One of the most important hyperparameters in 1st order gradient-based optimization is the learning rate $\eta$. We typically tune the learning rate by looking at the so-called training curves. To see the impact of different values of the learning rate on the **validation** loss, look at the following figure.

<div style="text-align:center">
    <img src="../imgs/loss curve vs lr.png" alt="drawing" width="400"/>
</div>

(Source: [CS231n CNNs for Visual Recognition](https://cs231n.github.io/neural-networks-3/))

Further hyperparameters are e.g. the choice of model, the optimizer itself, batch size in SGD, etc.. You will see many of them related to each model later in the lecture.

### Learning Rate Scheduling

We want to be able to dynamically adjust the learning rate $\eta$ for a time-dependent learning rate $\eta(t)$ to then control the rate of decay of $\eta$. The most common strategies are

$$\eta(t) = \eta_{i} \text{ if } t_{i} \leq t \leq t_{i+1}, \quad \text{piecewise constant}$$
$$\eta(t) = \eta_{0} \cdot e^{-\lambda t}, \quad \text{ exponential decay}$$
$$\eta(t) = \eta_{0} \cdot \left( \beta t + 1 \right)^{- \alpha}, \quad \text{ polynomial decay}$$

Going through the different proposed options in order:

- **Piecewise constant**: Decrease whenever optimization progress begins to stall.
- **Exponential decay**: Much more aggressive; can lead to premature stopping.
- **Polynomial decay**: Well-behaved when $\alpha = 0.5$.

Corrected for the time-dependent learning rate, and using the exponential decay our optimization then takes the following shape:

<div style="text-align:center">
    <img src="https://i.imgur.com/UJ3J86r.png" alt="drawing" width="400"/>
</div>

(Source: [classic.d2l.ai](https://classic.d2l.ai/chapter_optimization/gd.html))


Which is much much nicer behaved!

### Recipe for Machine Learning

If you are wondering how all of that fits together, Andrew Ng suggests this general workflow:

<div style="text-align:center">
    <img src="../imgs/recipe for ML.png" alt="drawing" width="500"/>
</div>

(Source: [Nuts and Bolts of Building Applications using Deep Learning](https://media.nips.cc/Conferences/2016/Slides/6203-Slides.pdf))

And a practical advice from [Prof. Matthias Niessner](http://niessnerlab.org/members/matthias_niessner/profile.html) at TUM is to:

1. Train the model on 1 data point to essentially learn it by heart. This way you prove that the model and output work correctly.
2. Train the model on a few samples. Proves that multiple inputs are handled correctly.
3. Move from the overfitting regime to full training.

## Further References

**Gradient-Based Optimization**

- [Patterns, Predictions, and Actions](https://mlstory.org/optimization.html), Chapter 5. Optimization; M. Hardt and B. Recht; 2022
- [Dive into Deep Learning](https://d2l.ai/chapter_optimization/index.html), Chapter 12. Optimization Algorithms
- [An overview of gradient descent optimization algorithms](https://ruder.io/optimizing-gradient-descent/index.html), S. Ruder; 2016

**Tricks of Optimization**

- [Understanding the Bias-Variance Tradeoff](http://scott.fortmann-roe.com/docs/BiasVariance.html); S. Fortmann-Roe; 2012