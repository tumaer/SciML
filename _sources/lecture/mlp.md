# Multilayer Perceptron


## Limitations of Linear Regression

Looking back to [Chapter 2](optimization.md) and the Tricks of Optimization, we discussed that the most general linear model could be any linear combination of $x$-values lifted to a predefined basis space $\varphi(x)$, e.g. polynomial, exponent, sin, tanh, etc. basis:

$$h(x)= \vartheta^{\top} \varphi(x) = \vartheta_0 + \vartheta_1 \cdot x + \vartheta_3 \cdot x^2 + \vartheta_3 \cdot \exp(x) + \vartheta_4 \cdot \sin(x) + \vartheta_5 \cdot \tanh(x) + ...$$

We also saw that through hyperparameter tuning one can find a basis that captures the underlying true relation between inputs and outputs quite well. In fact, for many practical problems, the approach of manually selecting meaningful basis functions and then training a linear model will be good enough, especially if we know something about the true relationship beforehand.

The problem is that for many tasks we don't have such a-priori information and exploring the space of all possible combinations of basis functions ends in an infeasible combinatoric problem. Especially if the datasets we have are large, e.g. ImageNet, it is unrealistic to think that we can manually transform the inputs to a linearly-separable space.

<div style="text-align:center">
    <img src="https://i.imgur.com/nPti5Rg.png" alt="drawing" width="600"/>
</div>

(Source: [Techniques for handling underfitting and overfitting in Machine Learning](https://towardsdatascience.com/techniques-for-handling-underfitting-and-overfitting-in-machine-learning-348daa2380b9))

*Q: If linear models are not all, what else?*

**Deep Learning** is the field that tries to systematically explore the space of possible non-linear relations $h$ between inputs $x$ and output $y$. As a reminder, a non-linear relation can be for example

$$
h(x) = x^{\vartheta_0} + \max\{0, \vartheta_1 \cdot x\} + ...
$$

In the scope of this class, we will look at the most popular and successful non-linear building blocks. We will see the Multilayer Perceptron, Convolutional Layer, and Recurrent Neural Network.


## Perceptron

Perceptron is a binary linear classifier and a single-layer neural network.

```{figure} ../imgs/mlp.png
---
width: 600px
align: center
name: mlp
---
Multilayer perceptron (Source: [What the Hell is Perceptron?](https://towardsdatascience.com/what-the-hell-is-perceptron-626217814f53))
```

The perceptron generalizes the linear hypothesis $\vartheta^{\top} x$ by subjecting it to a step function $f$ as

$$h(x) = f(\vartheta^{\top}x).$$

In the case of two class classification, we use the sign function


$$f(a)=  \left\{\begin{array}{l} +1 , \quad a\ge 0  \\ -1 , \quad a<0 \end{array}\right.$$

$f(a)$ is called activation function as it represents a simple model of how neurons respond to input stimuli. Other common activation functions are the sigmoid, tanh, and ReLU ($=\max(0,x)$) activation functions

<div style="text-align:center">
    <img src="https://i.imgur.com/4opuLgP.png" alt="drawing" width="500"/>
</div>

(Source: [Introduction to Different Activation Functions for Deep Learning](https://medium.com/@shrutijadon/survey-on-activation-functions-for-deep-learning-9689331ba092))


## Multilayer Perceptron  (MLP)

If we stack multiple perceptrons after each other with a user-defined dimension of the intermediate (a.k.a. latent, hidden) space, we get a multilayer perceptron.


<div style="text-align:center">
    <img src="https://i.imgur.com/JXtd2fy.png" alt="drawing" width="800"/>
</div>

(Source: [Training Deep Neural Networks](https://towardsdatascience.com/training-deep-neural-networks-9fdb1964b964))

We could write down the stack of such layer-to-layer transformations as 

$$h(x) = f_4 ( W_4 f_3 ( W_3 f_2(W_2 f_1(W_1 \mathbf{x})))).$$

In the image above, the black line connecting entry $i$ of the input with the corresponding entry $j$ of hidden layer 1 corresponds to the row $i$ and column $j$ of the learnable weights matrix $W_{1}$. For regression, typically the last activation, here $f_4$, is the identity.

It is crucial to have non-linear activation function. Why? Simply concatenating linear functions results in a new linear function! You immediately see it if you remove the activations $f_i$ in the equation above.

By the **Universal Approximation Theorem**, a single-layer perceptron with any "squashing" activation function, can approximate essentially any functional $h: x \to y$. More on that in the [deeplearningbook.org](https://www.deeplearningbook.org/contents/mlp.html). However, empirically we see improved performance when we stack multiple layers, adding the depth (number of hidden layers) and width (dimension of hidden layers) of a neural network to the hyperparameters of paramount importance.


## Further References

- [Deep Learning](https://www.deeplearningbook.org/), Chapters 6; Goodgellow, Bengio, Courville; 2016
