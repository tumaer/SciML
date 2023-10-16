# Linear Models

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
l(\vartheta)&=\log L(\vartheta)=\log \prod_{i=1}^{m} \frac{1}{\sqrt{2 \pi \sigma^{2}}} e^{-\frac{\left(y^{(i)}-\vartheta^{\top} x^{(i)}\right)^{2}}{2 \sigma^{2}}}\\
&=\sum_{i=1}^{m} \log \frac{1}{\sqrt{2 \pi \sigma^{2}}} e^{- \frac{\left(y^{(i)}-\vartheta^{\top} x^{(i)}\right)^{2}}{2 \sigma^{2}}}\\
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


## Further References

**Linear & Logistic Regression**

- Machine Learning Basics [video](https://www.youtube.com/watch?v=73RL3WPPFE0&list=PLQ8Y4kIIbzy_OaXv86lfbQwPHSomk2o2e&index=2) and [slides](https://niessner.github.io/I2DL/slides/2.Linear.pdf) from the "Introduction to Deep Learning" course for Informatics students at TUM.
- [What happens if a linear regression is underdetermined i.e. we have fewer observations than parameters?](https://betanalpha.github.io/assets/case_studies/underdetermined_linear_regression.html)
