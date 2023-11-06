# Gradients

Gradients are a general tool of utility across many scientific domains and keep reappearing across areas. Machine learning is just one of a much larger group of examples which utilizes gradients to accelerate its optimization processes. Breaking the uses down into a few rough areas

* Machine learning (Backpropagation, Bayesian Inference, Uncertainty Quantification, Optimization)
* Scientific Computing (Modeling, Simulation)

But what are the general trends driving the continued use of automatic differentiation as compared to finite differences, or manual adjoints?

* The writing of manual derivative functions becomes intractable for large codebases or dynamically-generated programs
* We want to be able to automatically generate our derivatives

$$
\Longrightarrow \text{ Automatic Differentiation}
$$

## A Brief Incomplete History

1. 1980s/1990s: Automatic Differentiation in Scientific Computing mostly spearheaded by Griewank, Walther, and Pearlmutter
    * Adifor
    * Adol-C
    * ...
2. 2000s: Rise of Python begins
3. 2015: Autograd for the automatic differentiation of Python & NumPy is released
4. 2016/2017: PyTorch & Tensorflow are introduced with automatic differentiation at their core
5. 2018: JAX is introduced with its very thin Python layer on top of Tensorflow's compilation stack, where it performs automatic differentiation on the highest representation level
6. 2020-2022: Forward-mode estimators to replace the costly and difficult-to-implement backpropagation are being introduced

with the cost of machine learning training dominating datacenter-bills for many companies and startups alike there exist many alternative approaches out there to replace gradients, but none of them have gained significant traction so far. **But it is definitely an area to keep an eye out for.**

## The tl;dr of Gradients

Giving a brief overview of the two modes, with derivations of the properties as well as examples following later.

### Forward-Mode Differentiation

Examining a classical derivative computation

$$
\frac{\partial y}{\partial x} = \frac{\partial y}{\partial c} \left( \frac{\partial c}{\partial b} \left( \frac{\partial b}{\partial a} \frac{\partial a}{\partial x} \right) \right)
$$

then in the case of the forward-mode derivative the evaluation of the gradient is performed from the right to the left. The Jacobian of the intermediate values is then accumulated with respect to the input x

$$
\frac{\partial a}{\partial x}, \quad \frac{\partial b}{\partial x}
$$

and the information flows in the same direction as the computation. This means that we do not require any elaborate caching system to hold values in-memory for later use in the computation of a gradient, and hence require **much less memory** and are left with a **much simpler algorithm**.

<div>
    <center>
    <img = src="https://i.imgur.com/0ExLlWa.png" width="450">
</div>


### Reverse-Mode Differentiation (Backpropagation)

Taking a typical case of reverse-mode differentiation, or as it is called in machine learning "backpropagation"

$$
\frac{\partial y}{\partial x} = \left( \left( \frac{\partial y}{\partial c} \frac{\partial c}{\partial b} \right) \frac{\partial b}{\partial a} \right) \frac{\partial a}{\partial x}
$$

in the case of reverse-mode differentiation the evaluation of the gradient is then performed from the left to the right. The Jacobians of the output $y$ are then accumulated with respect to each of the intermediate variables

$$
\frac{\partial y}{\partial a}, \quad \frac{\partial y}{\partial b}
$$

and the information flows in the opposite direction of the function evaluation, which points to the main difficulty of reverse-mode differentiation. We require an elaborate caching system to hold values in-memory for when they are needed for the gradient computation, and hence require **much more memory** and are left with a **much more difficult algorithm**.

<div>
    <center>
    <img = src="https://i.imgur.com/3H2gZDl.png" width="450">
</div>


### Forward- vs. Reverse-Mode

The performance comparison between forward-mode, and reverse-mode gradient can be broken down depending on the size of our input vector, and our output vector. So for the case of abstracting our neural network as a function with takes an input vector of a certain size $n$, and generates and output vector of a certain size $m$

$$
f: \mathbb{R}^{n} \longrightarrow \mathbb{R}^{m}
$$

* Forward-mode: More efficient for gradients of scalar-to-vector functions, i.e. $m >> n$
* Reverse-mode: More efficient for gradients of vector-to-scalar functions, i.e. $m << n$

As most loss functions in machine learning output a scalar value, reverse-mode differentiation is a very natural choice for these computations. A way to circumvent these issues of forward-mode differentiation, and simplify the technical infrastructure in the background is to **compose** forward-mode with vectorization, or only compute an estimator of the gradient where multiple forward-mode samples are used.

## In-Depth Look

$$
f = f_{4} \circ f_{3} \circ f_{2} \circ f_{1}
$$

where we are essentially converting from space-to-space with each function. Each function is an abstraction for an individual neural network layer as we will see in the new year in much more depth when constructing neural networks, or in the practical example later on.

$$
\begin{align}
    f_{1}: \mathbb{R}^{n} &\longrightarrow \mathbb{R}^{m_{1}} \\
    f_{2}: \mathbb{R}^{m_{1}} &\longrightarrow \mathbb{R}^{m_{2}} \\
    f_{3}: \mathbb{R}^{m_{2}} &\longrightarrow \mathbb{R}^{m_{3}} \\
    f_{4}: \mathbb{R}^{m_{3}} &\longrightarrow \mathbb{R}^{m} \\
\end{align}
$$

where our overall network $o = f(x)$ is broken down as

$$
\begin{align}
    x_{2} &= f_{1}(x) \\
    x_{3} &= f_{2}(x_{2}) \\
    x_{4} &= f_{3}(x_{3}) \\
    o     &= f_{4}(x_{4})
\end{align}
$$

using the chain rule we can then compute the Jacobian $J_{f}(x) = \frac{\partial o}{\partial x}$

$$
\begin{align}
\frac{\partial o}{\partial x} &= \frac{\partial o}{\partial x_{4}} \frac{\partial x_{4}}{\partial x_{3}} \frac{\partial x_{3}}{\partial x_{2}} \frac{\partial x_{2}}{\partial x} \\
&= \frac{\partial f_{4}(x_{4})}{\partial x_{4}} \frac{\partial f_{3}(x_{3})}{\partial x_{3}} \frac{\partial f_{2}(x_{2})}{\partial x_{2}} \frac{\partial f_{1}(x)}{\partial x} \\
&= J_{f_{4}}(x_{4}) J_{f_{3}}(x_{3}) J_{f_{2}}(x_{2}) J_{f_{1}}(x)
\end{align}
$$

this approach to this computation would be highly inefficient, as such we rely on matrix computation for more efficiency, i.e.

$$
J_{f}(x) = \frac{\partial f(x)}{\partial x} = \left(\begin{matrix}
    \frac{\partial f_{1}}{\partial x_{1}} & \ldots & \frac{\partial f_{1}}{\partial x_{n}} \\
    \vdots & \ddots & \vdots \\
    \frac{\partial f_{m}}{\partial x_{1}} & \ldots & \frac{\partial f_{m}}{\partial x_{n}}
\end{matrix} \right) = \left( \begin{matrix}
    \nabla f_{1}(x)^{\top} \\
    \vdots \\
    \nabla f_{m}(x)^{\top}
\end{matrix}\right) = \left( \begin{matrix}
    \frac{\partial f}{\partial x_{1}}, & \ldots & , \frac{\partial f}{\partial x_{n}}
\end{matrix}\right)
$$

in practice we would **love to** have access to this Jacobian, but the reality is that in 99.99999% of the cases it is too expensive to compute and as such we have to make do with snippets from this Jacobian, namely the **Jacobian Vector Product (JVP)**, and the **Vector Jacobian Product (VJP)**.

* The i-th row of $J_{f}(x)$ gives us the vector Jacobian product (reverse-mode differentiation)
* The j-th column of $J_{f}(x)$ gives us the Jacobian vector product (forward-mode differentiation)

Examining the case for when $n<m$, then it is more efficient to compute each column using the jacobian vector product in a right-to-left manner, i.e. the right multiplication a column vector gives us

$$
J_{f}(x) v = \underbrace{J_{f_{4}}(x_{4})}_{m \times m_{3}} \underbrace{J_{f_{3}}(x_{3})}_{m_{3} \times m_{2}} \underbrace{J_{f_{2}}(x_{2})}_{m_{2} \times m_{1}} \underbrace{J_{f_{1}}(x_{1})}_{m_{1} \times n} \underbrace{v}_{n \times 1}
$$

which is then computed with forward-mode differentiation. The pseudoalgorithm is given below.

<div>
    <center>
    <img = src="https://i.imgur.com/zun2BoB.png" width="450">
</div>

Returning to the cost-advantage of forward-mode differentiation for this specific case the cost of computation in this case if $\mathcal{O}(n^{3})$. If we now have the case where $n>m$, then it is more efficient to compute $J_{f}(x)$ for each row using the vector Jacobian product (VJP) in a left-to-right manner, i.e.

$$
u^{\top} J_{f}(x) = \underbrace{u^{\top}}_{1 \times m} \underbrace{J_{f_{4}}(x_{4})}_{m \times m_{3}} \underbrace{J_{f_{3}}(x_{3})}_{m_{3} \times m_{2}} \underbrace{J_{f_{2}}(x_{2})}_{m_{2} \times m_{1}} \underbrace{J_{f_{1}}(x_{1})}_{m_{1} \times n}
$$

for the solving of which reverse-mode differentiation is the most well-suited. The pseudoalgorithm for which can be found below

<div>
    <center>
    <img = src="https://i.imgur.com/W4rgJyk.png" width="450">
</div>

The cost of computation in this case is $\mathcal{O}(n^{2})$.

## A Practical Example

Considering a simple feed-forward model with 4 layers / an MLP with one hidden layer, we now have the following computation setup represented as a directed acyclic graph:

<div>
    <center>
    <img src="https://i.imgur.com/aMbZOqb.png" width="450">
</div>

The MLP with one hidden layer is written down as

$$
\mathcal{L}\left( (x, y), \theta \right) = \frac{1}{2} || y - W_{2} \varphi(W_{1}x) ||^{2}_{2}
$$

which is then represented as the following feedforward model:

$$
\begin{align}
    \mathcal{L} &= f_{4} \circ f_{3} \circ f_{2} \circ f_{1} \\
    x_{2} &= f_{1}(x, \theta_{1}) = W_{1}x \\
    x_{3} &= f_{2}(x_{2}, \emptyset) = \varphi(x_{2}) \\
    x_{4} &= f_{3}(x_{3}, \theta_{3}) = W_{2}x_{3} \\
    \mathcal{L} &= f_{4}(x_{4}, y) = \frac{1}{2} || x_{4} - y ||^{2}
\end{align}
$$

the $\theta_{k}$ are the optional parameters for each layer. As, by construction, the final layer returns a scalar, it is much more efficient to use reverse-mode differentiation to compute the gradient vectors in this case. We begin by computing the gradients of the loss with respect to the parameters in the earlier layers

$$
\begin{align}
    \frac{\partial \mathcal{L}}{\partial \theta_{3}} &= \frac{\partial \mathcal{L}}{\partial x_{4}} \frac{\partial x_{4}}{\partial \theta_{3}} \\
    \frac{\partial \mathcal{L}}{\partial \theta_{2}} &= \frac{\partial \mathcal{L}}{\partial x_{4}} \frac{\partial x_{4}}{\partial x_{3}} \frac{\partial x_{3}}{\partial \theta_{2}} \\
    \frac{\partial \mathcal{L}}{\partial \theta_{1}} &= \frac{\partial \mathcal{L}}{\partial x_{4}} \frac{\partial x_{4}}{\partial x_{3}} \frac{\partial x_{3}}{\partial x_{2}} \frac{\partial x_{2}}{\partial \theta_{1}}
\end{align}
$$

this recursive computation procedure can subsequently be condensed down to a pseudoalgorithm:

<div>
    <center>
    <img src="https://i.imgur.com/4jWskvp.png" width="450">
</div>

what is missing from this pseudoalgorithm is the definition of the vector Jacobian product of each layer, which depends on the type and function of each layer. Or in a slightly more intricate case, please see the example below for what this computation looks like in the case of backpropagation.

<div>
    <center>
    <img src="https://i.imgur.com/meT1rHZ.png" width="450">
</div>

## What are the Core-Levers of the Alternative Approaches

* Do we actually need accurate gradients for the training, or can we actually get away with much much coarser gradients to power our training?
* Approximate the reverse-mode gradients with a construction of cheap forward-mode gradients
  * By construction of a Monte-Carlo estimator for the reverse-mode gradient using forward-mode gradient samples
  * Randomizing the forward-mode gradients and then constructing an estimator
* Taking gradients at different program abstraction levels, taking the example of JAX we have access to the following main program abstraction levels at which gradients can be computed
  * Python frontend
  * Jaxpr
  * MHLO
  * XLA

## Further References

1. [Autograd: Effortless Gradients in Numpy](https://indico.ijclab.in2p3.fr/event/2914/contributions/6483/subcontributions/180/attachments/6060/7185/automl-short.pdf)
2. [Automatic Differentiation in PyTorch](https://openreview.net/pdf?id=BJJsrmfCZ)
3. [Tangent: Automatic Differentiation Using Source Code Transformation in Python](https://arxiv.org/pdf/1711.02712.pdf)
4. [Automatic Differentiation in Machine Learning: A Survey](https://arxiv.org/pdf/1502.05767.pdf)
5. Chapter 4 Dougal MacLaurin's [PhD Thesis](https://dougalmaclaurin.com/phd-thesis.pdf)
6. A Playful Introduction to Gradients: [Jax's Autodiff Cookbook](https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html)
