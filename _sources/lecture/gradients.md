# Gradients

Gradients are a general tool of utility across many scientific domains and keep reappearing across areas. Machine learning is just one of a much larger group of examples which utilizes gradients to accelerate its optimization processes. Breaking the uses down into a few rough areas, we have:

* Machine learning (Backpropagation, Bayesian Inference, Uncertainty Quantification, Optimization)
* Scientific Computing (Modeling, Simulation)

But what are the general trends driving the continued use of automatic differentiation as compared to finite differences, or manual adjoints?

* The writing of manual derivative functions becomes intractable for large codebases or dynamically-generated programs
* We want to be able to automatically generate our derivatives

$$
\Longrightarrow \text{ Automatic Differentiation}
$$

```{figure} ../imgs/mlp.png
---
width: 600px
align: center
name: mlp_2
---
Multilayer perceptron (Source: [Techniques for handling underfitting and overfitting in Machine Learning](https://towardsdatascience.com/techniques-for-handling-underfitting-and-overfitting-in-machine-learning-348daa2380b9))
```

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

With the cost of machine learning training dominating datacenter-bills for many companies and startups alike there exist many alternative approaches out there to replace gradients, but none of them have gained significant traction so far. **But it is definitely an area to keep an eye out for.**

## The tl;dr of Gradients

Giving a brief overview of the two modes, with derivations of the properties as well as examples following later.

### Forward-Mode Differentiation

Examining a classical derivative computation

$$
\frac{\partial y}{\partial x} = \frac{\partial y}{\partial c} \left( \frac{\partial c}{\partial b} \left( \frac{\partial b}{\partial a} \frac{\partial a}{\partial x} \right) \right)
$$ (ad_forward_chain)

then in the case of the forward-mode derivative the evaluation of the gradient is performed from the right to the left. The Jacobian of the intermediate values is then accumulated with respect to the input $x$

$$
\frac{\partial a}{\partial x}, \quad \frac{\partial b}{\partial x}
$$ (ad_forward_order)

and the information flows in the same direction as the computation. This means that we do not require any elaborate caching system to hold values in-memory for later use in the computation of a gradient, and hence require **much less memory** and are left with a **much simpler algorithm**.

```{figure} ../imgs/ad_forward.png
---
width: 500px
align: center
name: ad_forward
---
Forward-mode differentiation. (Source: {cite}`maclaurin2016`, Section 2)
```
