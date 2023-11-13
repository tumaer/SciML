# Tricks of Optimization


## Tricks of Optimization

This is a collection of practical tools for designing and optimizing machine learning models.

> Notation alert: In this sebsection, we switch back to the notation used in Chapter 1, i.e. $x$ is the x-axis, $y$ are measurements on the y-axis, $h(x)$ is the model evaluated at $x$, and $J$ the loss.

**Linear Regression (revised)**

Looking back to [Chapter 1](linear.md), the simplest linear model for $x \in \mathbb{R}$ is

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
    <img src="https://i.imgur.com/nPti5Rg.png" alt="drawing" width="600"/>
</div>

(Source: [Techniques for handling underfitting and overfitting in Machine Learning](https://towardsdatascience.com/techniques-for-handling-underfitting-and-overfitting-in-machine-learning-348daa2380b9))

#### Bias-Variance Tradeoff

Typically, over- and underfitting are analyzed through the lens of the bias-variance decomposition.

- **Bais Error**: Difference between the average (over infinitely many same-sized datasets coming from the same distribution) model prediction and the correct values which we are trying to predict.

- **Variance Error**: Variability of the model predictions at each position $x$ (averaged over infinitely many models trained on same-sized datasets coming from the same distribution). 

- **Irreducible Error**: Originates from the noise in the measurements. Given a corupted dataset, this error cannot be reduced with ML.

In the figure below, each point corresponds to the prediction of a model trained on a different dataset.

<div style="text-align:center">
    <img src="https://i.imgur.com/Y2IscaH.png" alt="drawing" width="400"/>
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
    <img src="https://i.imgur.com/Pm1otyT.png" alt="drawing" width="400"/>
</div>

(Source: [Understanding the Bias-Variance Tradeoff](http://scott.fortmann-roe.com/docs/BiasVariance.html))

Given the true model and enough data to calibrate it, we should be able to reduce both the bias and variance terms to zero. However, working with imperfect models and limited data, we strive for an optimum in terms of model choice.

#### Advanced Topics: Double Descent

In recent years, machine learning models have been growing extremely large, e.g. GPT-3 175B parameters. An empirical observation that has been studied by [M. Belkin et al. 2019](More data can hurt for linear regression: Sample-wise
double descent) demonstrates that contrary to the theory behind the bias-variance tradeoff if the number of parameters is too overparametrized, model performance starts improving again. Indeed, for many practical applications, this regime has not been fully explored and making ML models larger seems to improve performance further, consistently with [The Bitter Lesson](http://www.incompleteideas.net/IncIdeas/BitterLesson.html) of R. Sutton.

<div style="text-align:center">
    <img src="https://i.imgur.com/CFdVvIq.png" alt="drawing" width="500"/>
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
    <img src="https://i.imgur.com/9xHVNt9.png" alt="drawing" width="600"/>
</div>

(Source: [Train/Test Split and Cross Validation in Python](https://towardsdatascience.com/train-test-split-and-cross-validation-in-python-80b61beca4b6))

#### Cross Validation

If we split a dataset into $K$ pieces, we could train the model $K$ times each time excluding a different subset. We could then evaluate the model performance on the test set for each of the $K$ models and by that get a good estimate of the variance of the error. If we select the model with the least error and train it further on the whole training set, then we talk about K-fold cross-validation.

### Regularization

One possibility to counteract overfitting and still have an expressive model is regularization. There are many approaches belonging to the class of regularization techniques. 

- Adding a regularization term to the loss - add an additional term penalizing large weight values:
    - **L1 regularization** - promotes sparsity: 

    $$J_{L1}(\vartheta) = J(\vartheta) + \alpha_{L1} \cdot \sum_{i=1}^{\#params} |\vartheta_i|$$

    - **(squared) L2 regularization** - takes information from all features; typical choice: 

    $$J_{L2}(\vartheta) = J(\vartheta) + \alpha_{L2} \cdot \sum_{i=1}^{\#params} \vartheta_i^2$$

#### $l_p$ norm
To better understand the regression losses, we will look at the general $l_p$ norm of vector $w\in \mathbb{R}^m$:


$$||w||_p=\left(\sum_{i=1}^m |w_i|^p\right)^{1/p} \quad \text{for } p \ge 1.$$

In the special case $p=1$ we recover the L1 loss, and the squared version of $p=2$ corresponds to the MSE loss. Other special case is $p \to \infty$ leading to $||w||_{\infty}= \max \left\{ |w_1|,|w_2|,...,|w_{m}| \right\}$. We see that with increasing $p$ the larger terms dominate


<div style="text-align:center">
    <img src="https://i.imgur.com/Z05qdjO.png" alt="drawing" width="600"/>
</div>

*Figure*: The blue line represents the solution set of an under-determined system of equations. The red line represents the minimum-norm level sets that intersect the blue line for each norm. For norms $p=0,...,1$, the minimum-norm solution corresponds to the sparsest solution with only one coordinate active. For $p \ge 2$ the minimum-norm solution is not sparse, but all coordinates are active.

(Source: [Brunton and Kutz 2019](https://www.cambridge.org/core/books/datadriven-science-and-engineering/77D52B171B60A496EAFE4DB662ADC36E), Fig. 3.9)


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
    <img src="https://i.imgur.com/2NVKmG8.png" alt="drawing" width="400"/>
</div>

(Source: [CS231n CNNs for Visual Recognition](https://cs231n.github.io/neural-networks-3/))

Further hyperparameters are e.g. the choice of model, the optimizer itself, batch size in SGD, etc. You will see many of them related to each model later in the lecture.

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
    <img src="https://i.imgur.com/ir6Mdmm.png" alt="drawing" width="600"/>
</div> 

(Source: [Nuts and Bolts of Building Applications using Deep Learning](https://media.nips.cc/Conferences/2016/Slides/6203-Slides.pdf))

And a practical advice from [Prof. Matthias Niessner](http://niessnerlab.org/members/matthias_niessner/profile.html) at TUM is to:

1. Train the model on 1 data point to essentially learn it by heart. This way you prove that the model and output work correctly.
2. Train the model on a few samples. Proves that multiple inputs are handled correctly.
3. Move from the overfitting regime to full training.

## Further References

- [Understanding the Bias-Variance Tradeoff](http://scott.fortmann-roe.com/docs/BiasVariance.html); S. Fortmann-Roe; 2012