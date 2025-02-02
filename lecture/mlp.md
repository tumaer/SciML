# Multilayer Perceptron

`````{admonition} Learning outcome
:class: tip 
- Write down the perceptron equation.
- Name two popular nonlinear activation functions.
- How does the Universal Approximation Theorem contradict empirical knowledge?
`````

## Limitations of Linear Regression

Going back to lecture [](tricks.md), we discussed that the most general linear model could be any linear combination of $x$-values lifted to a predefined basis space $\varphi(x)$, e.g., polynomial, exponent, sin, tanh, etc. basis:

$$h(x)= \vartheta^{\top} \varphi(x) = \vartheta_0 + \vartheta_1 \cdot x + \vartheta_3 \cdot x^2 + \vartheta_3 \cdot \exp(x) + \vartheta_4 \cdot \sin(x) + \vartheta_5 \cdot \tanh(x) + ...$$ (general_lin_reg)

We also saw that through hyperparameter tuning, one can find a basis that captures the underlying true relation between inputs and outputs well. In fact, for many practical problems, the approach of manually selecting meaningful basis functions and then training a linear model will be good enough, especially if we know something about the data beforehand.

The problem is that for many tasks, we don't have such a-priori information, and exploring the space of all possible combinations of basis functions ends in an infeasible combinatoric problem. Especially if our datasets are large, e.g., ImageNet, it is unrealistic to think that we can manually transform the inputs to a linearly separable space.

```{figure} ../imgs/mlp/under_overfitting.png
---
width: 600px
align: center
name: under_overfitting_duplicate
---
Under- and overfitting (Source: [Techniques for handling underfitting and overfitting in Machine Learning](https://towardsdatascience.com/techniques-for-handling-underfitting-and-overfitting-in-machine-learning-348daa2380b9))
```

*Q: If linear models are not all, what else?*

**Deep Learning** is the field that tries to systematically explore the space of possible non-linear relations $h$ between input $x$ and output $y$. As a reminder, a non-linear relation can be, for example

$$
h(x) = x^{\vartheta_0} + \max\{0, \vartheta_1 \cdot x\} + ...
$$ (nonlinear_regression)

In the scope of this class, we will look at the most popular and successful non-linear building blocks. We will see the Multilayer Perceptron, Convolutional Layer, and Recurrent Neural Network.

## Perceptron

Perceptron is a binary linear classifier and a single-layer neural network.

```{figure} ../imgs/mlp/perceptron.png
---
width: 500px
align: center
name: perceptron
---
The perceptron (Source: [What the Hell is Perceptron?](https://towardsdatascience.com/what-the-hell-is-perceptron-626217814f53))
```

The perceptron generalizes the linear hypothesis $\vartheta^{\top} x$ by subjecting it to a step function $f$ as

$$h(x) = f(\vartheta^{\top}x).$$ (perceptron)

> Note: The term $\vartheta x$ is actually an *affine* transformation, not just *linear*. We use the notation with $w_0 = b$ for brevity.

In the case of two-class classification, we use the sign function

$$f(a)=  \left\{\begin{array}{l} +1  \quad \text{if } a\ge 0,  \\ -1  \quad \text{else.} \end{array}\right.$$ (sign_fn)

$f(a)$ is called *activation function* as it represents a simple model of how neurons respond to input stimuli. Other common activation functions are the sigmoid, tanh, and ReLU ($=\max(0,x)$).

```{figure} ../imgs/mlp/activation_functions.png
---
width: 450px
align: center
name: activation_functions
---
Activation functions (Source: [Introduction to Different Activation Functions for Deep Learning](https://medium.com/@shrutijadon/survey-on-activation-functions-for-deep-learning-9689331ba092))
```

## Multilayer Perceptron  (MLP)

If we stack multiple perceptrons after each other with a user-defined dimension of the intermediate (a.k.a. latent or hidden) space, we get a multilayer perceptron.

```{figure} ../imgs/mlp/mlp.png
---
width: 700px
align: center
name: mlp
---
Multilayer Perceptron (Source: [Training Deep Neural Networks](https://towardsdatascience.com/training-deep-neural-networks-9fdb1964b964))
```

We could write down the stack of such layer-to-layer transformations as

$$h(x) = W_4 f_3 ( W_3 f_2(W_2 f_1(W_1 \mathbf{x}))).$$ (mlp_stack)

In the image above, the black line connecting entry $i$ of the input with the corresponding entry $j$ of hidden layer 1 corresponds to the row $i$ and column $j$ of the learnable weights matrix $W_{1}$. Note that the output layer does not have an activation function - in the case of regression, we typically stop with the linear transformation, and in the case of classification, we typically apply the softmax function to the output vector.

It is crucial to have non-linear activation functions. Why? Simply concatenating linear functions results in a new linear function! You immediately see it if you remove the activations $f_i$ in the equation above.

By the **Universal Approximation Theorem**, a single-layer perceptron with any "squashing" activation function (i.e., $h(x)=W_2 f(W_1 x)$) can approximate essentially any functional $h: x \to y$. More on that in {cite}`goodfellow2016`, Section 6.4.1. However, empirically we see improved performance when we stack multiple layers, adding the depth (number of hidden layers) and width (dimension of hidden layers) of a neural network to the hyperparameters of paramount importance.

**Exercise: Learning XOR**

Find the simplest MLP capable of learning the XOR function, and fit its parameters.

```{figure} ../imgs/mlp/xor_function.png
---
width: 300px
align: center
name: xor_function
---
XOR function (Source: {cite}`goodfellow2016`, Section 6.1)
```

## Further References

- {cite}`goodfellow2016`, Chapter 6
