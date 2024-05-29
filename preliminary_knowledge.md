# Preliminary Knowledge

## What do you need?

A basic handling of _Calculus_, _Linear Algebra_, and _Probability Theory_ are required to understand the theoretical part of the course.

Exercises, and practicising the practical parts of the course require a basic handling of Python.

> If you are an ardent believer of programming language X, we will not force you to use Python. Problem sets, as well as model solutions will only be provided in Python, but there is nothing keeping you from doing the exercises in e.g. Julia, Mojo, Rust, etc.

## Sources for Preliminary Reading

To refresh your memory on the preliminary knowledge required for this course, we strongly recommend the concise, accessible notes from the [_Dive into Deep Learning_](https://d2l.ai) online book by Aston Zhang, Zachary C Lipton, Mu Li, and Alexander J Smola.

* [Linear Algebra](https://www.d2l.ai/chapter_preliminaries/linear-algebra.html)
* [Calculus](https://www.d2l.ai/chapter_preliminaries/calculus.html)
* [Probability and Statistics](https://www.d2l.ai/chapter_preliminaries/probability.html)

If you prefer a more expansive book/lecture style, we can furthermore recommend the following chapter from [_Mathematical Foundations of Machine Learning_](https://nowak.ece.wisc.edu/MFML.pdf) by Robert Nowak:

* Chapter 1: Probability in Machine Learning
* Chapter 2: Discrete Probability Distributions and Classification

## Exercises to Test Preliminary Knowledge

If you seek to refresh your memory by the way of exercises, please find below a problem set to test your preliminary knowledge. The solutions can be found in the Appendix (link to go here).

### Linear Algebra

#### Moore-Penrose Pseudoinverse: Applied

Complete the pseudoinverse of the following two matrices:

$$
\begin{bmatrix}
    1 & 1 \\
    2 & 2
\end{bmatrix}, \quad
\begin{bmatrix}
    4 & 3 \\
    8 & 6
\end{bmatrix}
$$

#### Singular Value Decomposition: Applied

Compute the singular value decomposition of the following two matrices:

$$
\begin{bmatrix}
    3 & 1 & 1 \\
    -1 & 3 & 1
\end{bmatrix}, \quad
\begin{bmatrix}
    2 & 4 \\
    1 & 3 \\
    0 & 0 \\
    0 & 0
\end{bmatrix}
$$

#### Singular Value Decomposition: Theoretical

Show that for a matrix $A$:

1. The rank of $A$ is $r$, where $r$ is the minimum $i$ such that $\underset{\underset{|\bf{v}|=1}{\bf{v} \perp \bf{v_{1}}, \bf{v_{2}}, \ldots, \bf{v_{i}}}}{\arg \max} |A \hspace{1pt} \bf{v} | = 0$.
2. $\left| \bf{u}^{T}_{1} A \right| = \underset{|u|=1}{\max} \left| \bf{u}^{T} A \right| = \sigma_{1}$.

> Hint: Recall the definition of the singular value decomposition.

### Probability Theory

#### Variance of a Sum

Show that the variance of a sum is $var\left[X + Y\right] = var[X] + var[Y] + 2 cov[X, Y]$, where $cov[X, Y]$ is the covariance between $X$ and $Y$.

#### Pairwise Independence Does not Imply Mutual Independence

We say that two random variables are pairwise independent if

$$
p(X_{2} | X_{1}) = p(X_{2})
$$

and hence

$$
p(X_{2}, X_{1}) = p(X_{1}) p(X_{2}|X_{1}) = p(X_{1}) p(X_{2}).
$$

We say that $n$ random variables are mutually independent if

$$
p(X_{i}|X_{\mathcal{S}}) = p(X_{i}) \quad \forall \mathcal{S} \subseteq \left\{1, \ldots, n \right\} \backslash \{ i\}
$$

and hence

$$
p(X_{1:n}) = \prod^{n}_{i=1} p(X_{i}).
$$

Show that pairwise independence between all pairs of variables does not necessarily imply mutual independence.

> Hint: It suffices to give a counter example.

#### Bernoullie Distribution

The form of the Bernoulli distribution given by

$$
Bern(x|\mu) = \mu^{x} \left( 1 - \mu \right)^{1-x}
$$

is not symmetric between the two values of x. In some situations, it will be more convenient to use an equivalent formulation for which $x \in \{-1, 1\}$, in which case the distribution can be written

$$
p(x|\mu) = \left( \frac{1 - \mu}{2} \right)^{\frac{1 -x}{2}} \left( \frac{1 + \mu}{2} \right)^{\frac{1+x}{2}}
$$

where $\mu \in [-1, 1]$. Show that the distribution is normalized, and evaluate its mean, variance, and entropy.

#### Beta Distribution

Prove that the beta distribution, given by

$$Beta(\mu | a, b) = \frac{\Gamma(a+b)}{\Gamma(a)\Gamma(b)} \mu^{a-1} \left( 1 - \mu \right)^{b-1}$$ (beta-distribution)

is correctly normalized, so that

$$\int_{0}^{1} Beta(\mu |a, b) d\mu = 1$$ (beta-probability-condition)

holds. This is equivalent to showing that

$$
\int_{0}^{1} \mu^{a-1} \left( 1 - \mu \right)^{b-1} d\mu = \frac{\Gamma(a) \Gamma(b)}{\Gamma(a+b)}
$$

From the definition of the gamma function, we have

$$
\Gamma(a) \Gamma(b) = \int_{0}^{\infty} \exp(-x) x^{a-1} dx \hspace{2pt} \int_{0}^{\infty} \exp(-y) y^{b - 1} dy.
$$

Use expression {eq}`beta-probability-condition` to prove {eq}`beta-distribution` as follows. First bring the integral over y inside the integrand of the integral over x, next make the change of variable $t = y + x$ where $x$ is fixed, then interchange the order of the $x$ and $t$ integrations, and finally make the change of variable $x = t \mu$ where $t$ is fixed.

#### Mean, Mode, and Variance of the Beta Distribution

Suppose $\theta \sim Beta(a, b)$. Derive the mean, mode and variance.

> Hint: See the definition above, and consider the moment generating function.

#### Deriving the Inverse Gamma Density

Let $X \sim Ga(a, b)$, i.e.

$$Ga(x|a, b) = \frac{b^{a}}{\Gamma(a)} x^{a-1} e^{-xb}$$

Let $Y = 1 / X$. Show that $Y \sim IG(a, b)$, i.e.

$$IG(x | \text{shape}=a, \text{scale}=b) = \frac{b^{a}}{\Gamma(a)} x^{-(a+1)} e^{-b / x}.$$

#### Normalization Constant for a 1-Dimensional Gaussian

The normalization constant for a zero-mean Gaussian is given by

$$Z = \int_{a}^{b} \exp \left( - \frac{x^{2}}{2 \sigma^{2}} \right) dx$$

where $a = -\infty$ and $b=\infty$. To compute this, consider its square

$$Z^{2} = \int_{a}^{b} \int_{a}^{b} \exp \left( - \frac{x^{2} + y^{2}}{2\sigma^{2}} \right) dx \hspace{1pt} dy$$

Let us change variables from cartesian $(x, y)$ to polar $(r, \theta)$ using $x = r \cos \theta$ and $y = r \sin \theta$. Since $dx \hspace{1pt} dy = r \hspace{1pt} dr \hspace{1pt} d\theta$, and $\cos^{2} \theta + \sin^{2} \theta = 1$, we have

$$Z^{2} = \int_{0}^{2 \pi} \int_{0}^{\infty} r \exp \left( - \frac{r^{2}}{2\sigma^{2}} \right) dr \hspace{1pt} d\theta$$

Evaluate this integral and hence show $Z = \sigma \sqrt{2 \pi}$. We suggest to separate the integral into a product of two terms, the first of which (involving $d\theta$) is constant, so is easy. Another simplification is possibly by realizing that if $u = e^{-r^{2}/2\sigma^{2}}$ $du/dr = - \frac{1}{\sigma^{2}} r \hspace{1pt} e^{-r^{2} / 2\sigma^{2}}$, so the second integral reduces down to $\int u'(r)dr = u(r)$.

#### Kullback-Leibler Divergence

Evaluate the Kullback-Leibler divergence, expressing the relative entropy of two probability distributions,

$$\begin{align}
  KL(p || q) &= - \int p(\bf{x}) \ln q(\bf{x}) d\bf{x} - \left( - \int p(\bf{x}) \ln p(\bf{x}) d\bf{x} \right) \\
             &= - \int p(\bf{x}) \ln \left\{ \frac{q(\bf{x})}{p(\bf{x})} \right\} d\bf{x}.
\end{align}$$

between two Gaussians $p(\bf{x}) = \mathcal{N}(\bf{x} | \bf{\mu}, \bf{\Sigma})$, and $q(x) = \mathcal{N}(\bf{x} | \bf{m}, \bf{L})$.
