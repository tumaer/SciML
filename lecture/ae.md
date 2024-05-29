# Encoder-Decoder Models

A great many models in machine learning build on the Encoder-Decoder paradigm, but what is the intuition behind said paradigm? The core intuition is that while our original data may be high-dimensional, a good prediction only depends on a few key sub-dimensions of the original data, and as such we are able to dimensionally reduce our original data through an _Encoder_ before predicting or reconstructing our data onto the original data-plane with the _Decoder_.

```{figure} ../imgs/encoder/ae_architecture.png
---
width: 600px
align: center
name: ae_architecture
---
Autoencoder architecture (Adapted from [lilianweng.github.io](https://lilianweng.github.io/posts/2018-08-12-vae/)). 
```

The precursor to this paradigm is the realization which parameters/variables of our data are actually relevant. For this purpose we look at its roots in computational linear algebra beginning with the _Singular Value Decomposition_ to build towards the _Principal Component Analysis_, which subsequently feeds into _Autoencoder_ which are epitome of the Encoder-Decoder paradigm and build on the key insights from the Principal Component Analysis.

## Singular Value Decomposition

Generalizing the eigendecomposition of small matrices, the singular value decomposition allows for the factorization of real or complex matrices. For any matrix this decomposition can be computed, but only square matrices with a full set of linearly independent eigen-vectors can be diagonalized. Any $A \in \mathbb{R}^{m \times n}$ can be factored as

$$
A = U \Sigma V^{\top}
$$ (svd)

where $U \in \mathbb{R}^{m \times m}$ is a unitary matrix, where the columns are eigenvectors of $A A^{\top}$, $\Sigma$ is a diagonal matrix $\Sigma \in \mathbb{R}^{m \times n}$, and $V \in \mathbb{R}^{n \times n}$ is a unitary matrix whose columns are eigenvectors of $A^{\top}A$. I.e.

$$
\Sigma = \left[
    \begin{matrix}
        \sigma_{1} & 0 & \ldots & 0 & 0 \\
        0 & \ddots & & & \vdots \\
        \vdots & \ldots & \sigma_{r} & \ldots & 0 \\
         & & & \ddots & \\
         0 & & \ldots & & \sigma_{n}
    \end{matrix}
\right]
$$ (svd_sigma)

where $\sigma_{n}$ can also be 0. The singular values $\sigma_{1} \geq \ldots \sigma_{r} \geq 0$ are square roots of the eigenvalues of $A^{\top}A$, and $r = \text{rank}(A)$.

$$
A^{\top}A = V D V^{\top} \in \mathbb{R}^{n \times n}
$$ (svd_ata)

if symmetric, and has $n$ real eigenvalues. Here $V \in \mathbb{R}^{n \times n}$, columns are eigenvectors of $A^{\top}A$, and $D \in \mathbb{R}^{n \times n}$ is the diagonal matrix of real eigenvalues.

$$
A A^{\top} = U D' U^{\top} \in \mathbb{R}^{m \times m}
$$ (svd_aat)

if symmetric, and real has $m$ real eigenvalues. In this case $U \in \mathbb{R}^{m \times m}$ of which the columns are eigenvectors of $AA^{\top}$, and $D' \in \mathbb{R}^{m \times m}$ is a diagonal matrix of real eigenvalues. The geometric analogy of the singular value decomposition is then

* $A$ is a matrix of $r = \text{rank}(A)$
* $U$ is the rotation/reflection
* $\Sigma$ is the stretching
* $V^{\top}$ is the rotation/reflection

Broken down into its individual pieces, we get the following commutative diagram:

```{figure} ../imgs/encoder/ae_svd.svg
---
width: 400px
align: center
name: ae_svd_svg
---
Decomposing the effect of a matrix transforming a vector into its SVD components (Adapted from [Wikipedia](https://en.wikipedia.org/wiki/Singular_value_decomposition)). 
```

The singular value decomposition of $A$ can then be expressed in terms of a basis of $\text{range}(A)$ and a basis of $\text{range}(A^{\top})$.

$$
A = \sigma_{1} u_{1} v_{1}^{\top} + \ldots + \sigma_{r} u_{r} v_{r}^{\top}
$$ (svd_1)

$$
\begin{align}
    \Longrightarrow A^{\top} A &= V \Sigma^{\top} U^{\top} U \Sigma V^{\top} \\
    &= V \left[\begin{matrix}
        \sigma_{1}^{2} & & \ldots & & 0 \\
         & \ddots & & & \\
         \vdots & & \sigma_{r}^{2} & & \vdots \\
         & & & \ddots & \\
         0 & & \ldots & & 0
    \end{matrix}\right] V^{\top},
\end{align}
$$ (svd_2)

where $V$ is the eigenvector matrix (columns) of $A^{\top}A$, and the large matrix $\Sigma$ is a diagonal matrix of eigenvalues of $A^{\top}A$. Also,

$$
\begin{align}
    A A^{\top} &= U \Sigma V^{\top} V \Sigma^{\top} U^{\top} \\
    &= U \left[\begin{matrix}
        \sigma_{1}^{2} & & \ldots & & 0 \\
        & \ddots & & & \\
        \vdots & & \sigma_{r}^{2} & & \vdots \\
        & & & \ddots & \\
        0 & & \ldots & & 0
    \end{matrix}\right] U^{\top},
\end{align}
$$ (svd_3)

where $U$ is the eigenvector matrix (columns) of $A A^{\top}$, and the large diagonal matrix consists of the eigenvalues of $A A^{\top}$. Graphically, this decomposition looks like that:

```{figure} ../imgs/encoder/ae_svd.png
---
width: 600px
align: center
name: ae_svd
---
Singular value decompositions of a matrix $A=U\Sigma V^{\top}$. From left to right: $A$, $U$, $\Sigma$, $V^{\top}$. 
```

### Moore-Penrose pseudo-inverse

One application of singular value decomposition is to e.g. compute the pseudo-inverse, also called the _Moore-Penrose Inverse_ of a matrix $A \in \mathbb{R}^{m \times n}$.

Given $A \in \mathbb{R}$, $x \in \mathbb{R}^{n}$, and $y \in \mathbb{R}^{m}$, we consider the task of solving the following linear equation system for $x$:

$$
Ax = y,
$$ (ae_linear_eq)

where $y \notin \text{col}(A)$, i.e. not in the column space of A, is allowed.

```{figure} ../imgs/encoder/ae_moore_penrose.png
---
width: 400px
align: center
name: ae_moore_penrose
---
Geometric interpretation of Moore-Penrose inverse. 
```

Here, $y'$ is the orthogonal projection of $y$ onto the column space of $A$. With this transformation we are transforming the problem to solve to

$$
A \bar{x} + \bar{y} = y \Longrightarrow A \bar{x} = y'
$$ (ae_mp_system)

which is exactly what we require the pseudo-inverse for as it assures for the error $|| \bar{y}|| = \min$ to be minimal. We then employ the singular value decomposition of $A$ to solve $Ax=y$ in a least-squares sense:

$$
\begin{align}
    A &= U \Sigma V^{\top} \\
    A^{+} &= V \Sigma^{+} U^{\top}
\end{align}
$$ (ae_mp1)

where $A^{+}$ is the pseudo-inverse with

$$
\Sigma^{+}_{ij} = \begin{cases}
    0, \quad i \neq j \\
    \frac{1}{\sigma_{i}}, \quad 1 \leq i \leq r \\
    0, \quad r+1 \leq i \leq n
\end{cases}
$$ (ae_mp2)

As the approximation of $x$, we then compute

$$
\bar{x} = A^{+}y.
$$ (ae_mp3)

* If $A$ is invertible $\Rightarrow A^{+} = A^{-1} \Rightarrow \bar{x} = x$, and we obtain the exact solution.
* If A is singular, and $y \in \text{col}(A)$, then $\bar{x}$ is the solution which minimizes $||\bar{x}||_{2}$ as $( \bar{x} + \text{null}(A))$, and all are possible solutions.
* If A is singular, and $y \notin \text{col}(A)$, then $\bar{x}$ is the least-squares solution which minimizes $|| A \bar{x} - y ||_{2}$

Some useful properties of the pseudo-inverse are:

$$
\begin{align}
    A^{+}u_{i} &= \begin{cases}
        \frac{1}{\sigma_{i}} v_{i}, \quad i \leq r \\
        0, \quad i > r
    \end{cases} \\
    (A^{+})^{\top} v_{i} &= \begin{cases}
        \frac{1}{\sigma_{i}} u_{i}, \quad i \leq r \\
        0, \quad i > r
    \end{cases}
\end{align}
$$ (ae_mp4)

From which follows that $A \in \mathbb{R}^{m \times n}$ with n linearly independent columns $A^{\top}A$ is invertible. The following relations apply to $A^{+}$:

$$
A^{+} = \begin{cases}
    (A^{\top} A)^{-1} A^{\top}, \quad \text{if } \text{rank}(A) = n \\
    A^{\top}(A A^{\top})^{-1}, \quad \text{if } \text{rank}(A) = m
\end{cases}
$$ (ae_mp5)

## Principal Component Analysis

To further discern information in our data which is not required for our defined task, be it classification or regression, we can apply principal component analysis to reduce the dimensionality. Principal component analysis finds the optimal linear map from a $D$-dimensional input, to the design matrix on a $K$-dimensional space. Expressing this relation in probabilistic terms we speak of a linear map $D \rightarrow K$ with maximum variance $\Longrightarrow$ "autocorrelation" of the input data. This linear map can be written the following way in matrix notation:

```{figure} ../imgs/encoder/ae_svd_dataset.png
---
width: 600px
align: center
name: ae_svd_dataset
---
SVD applied to dataset $X$ of $D$-dimensional vectors $x_n$.
```

$$
X = U S V^{\top}
$$ (pca_svd)

Here, $U$ are the left singular vectors, $S$ are the singular values, and $V^{\top}$ are the right singular values. **This dimensionality reduction is achieved through the singular value decomposition (SVD).** What we seek to achieve is to compress

$$
\underset{D \times N}{x} \rightarrow \underset{D \times N}{\hat{x}},
$$ (pca_compression)

where the rank of $\hat{x}$ is $K$. To derive the relation between SVD and PCA we can use the following Lemma:

$$
||x - \hat{x}||_{F}^{2} \geq || x - \tilde{U} \tilde{U}^{\top}X||_{F}^{2} = \sum_{i \geq K + 1} s_{i}^{2}.
$$ (pca_lemma)

Here, the $s_{i}$ are the ordered singular values $s_{1} \geq s_{2} \geq \ldots \geq 0$, and $\tilde{U}$ is a $D \times K$ matrix

```{figure} ../imgs/encoder/ae_utilde.png
---
width: 200px
align: center
name: ae_utilde
---
$\tilde{U}$ matrix. 
```

We then have

$$
\begin{align}
    \tilde{U} \tilde{U}^{\top} X &= \tilde{U} \tilde{U}^{\top} U S V^{\top} \\
        &= \tilde{U} \tilde{S} V^{\top} \\
        &= U \tilde{S} V^{\top}
\end{align}
$$ (pca1)

as we have

$$
\begin{align}
    \tilde{U}^{\top} U &= \left[ I_{K \times K} | 0 \right] \in \mathbb{R}^{K \times D} \\
    \tilde{U}U S &= [I_{K \times K} | 0] S = \tilde{S} \\
    \tilde{U}S &= [\tilde{U}|0] \tilde{S} = U \tilde{S}
\end{align}
$$ (pca2)

From a probabilistic view we are creating N independent identically distributed (i.i.d) $D$-dimensional vectors $x_{n}$, see left side of {numref}`ae_svd_dataset`. We then have a sample mean, or sometimes called empirical mean, of

$$
\bar{x} = \frac{1}{N} \sum_{n=1}^{N}x_{n}
$$ (pca_mean)

and a sample co-variance (biased), or sometimes called empirical covariance, of

$$
\bar{\Sigma} = \frac{1}{N} \sum_{n=1}^{N} (x_{n} - \bar{x})(x_{u} - \bar{x})^{\top}
$$ (pca_cov)

for $N \rightarrow \infty$ the two then converge to

$$
\begin{align}
    \bar{x} &\rightarrow \mathbb{E}[x_{n}] \\
    \bar{\Sigma} &\rightarrow \mathbb{E}[(x_{n} - \bar{x})^{2}]
\end{align}
$$ (pca3)

if we assume for simplicity that $\bar{x} = 0$, then

$$
\Longrightarrow \begin{align*}
    N \bar{\Sigma} &= \sum_{n=1}^{N} x_{n} x_{n}^{\top} = X X^{\top} = U S V^{\top}V S^{\top} U^{\top} \\
    &= U S S^{\top}U^{\top} = U S^{(2)}U^{\top}
\end{align*}
$$ (pca4)

where $S^{(2)}$ is a newly constructed singular value matrix. Comparing this with the compressed matrix

$$
\hat{X} = \tilde{U} \tilde{U}^{\top} X
$$ (pca5)

from which follows

$$
\begin{align}
    N \hat{\bar{\Sigma}} &= \hat{X} \hat{X}^{\top} \\
        &= \tilde{U} \tilde{U}^{\top} X X \tilde{U}^{\top} \tilde{U} \\
        &= \tilde{I} S^{(2)} \tilde{I} \\
        &= \tilde{S}^{(2)}.
\end{align}
$$ (pca6)

The data vectors $\hat{x}_{n}$ in $\hat{x}$ are hence uncorrelated and $\hat{x}_{1}$ has the highest variance by ordering of the singular values. If you were to consider the case of classification, then the lesson of PCA is that data with higher variance is always better to classify, as it is the more important information contained in the dataset.

> The number of principal components we consider is effectively a hyperparameter we have to carefully evaluate before making a choice. See the reconstruction error on MNIST vs the number of latent dimensions used by the PCA as an example of this.

```{figure} ../imgs/encoder/ae_pca_components.png
---
width: 600px
align: center
name: ae_pca_components
---
Recontruction error vs number of latent dimensions used by PCA (Souce: {cite}`murphy2022`).
```

With this all being a bit abstract, here are a few examples of applications of PCA to increasingly more practical applications.

```{figure} ../imgs/encoder/ae_pca_2d1d.png
---
width: 300px
align: center
name: ae_pca_2d1d
---
PCA projection of 2d data onto 1d (Souce: {cite}`murphy2022`).
```

```{figure} ../imgs/encoder/ae_olivetti.png
---
width: 600px
align: center
name: ae_olivetti
---
(a) randomly chosen images from the Olivetti face database. (b) mean and first three PCA components (Souce: {cite}`murphy2022`).
```

### Computational Issues of Principal Components Analysis

So far we have only considered the eigendecomposition of the covariance matrix, but computationally more advantageous is the correlation matrix instead. This avoids the potential of a "confused" PCA due to a mismatch between length-scales in the dataset.

## Autoencoder

These dimensionality reduction techniques originating from computational linear algebra led to the construction of _Autoencoders_ in machine learning, an encoder-decoder network focussed solely on the reconstruction error of a network-induced dimensionality reduction in the classical case. Considering the linear case we have the following network:

$$
\begin{align}
    z &= W_{1} x \\
    \hat{x} &= W_{2} z
\end{align}
$$ (ae_linear)

with $W_{1} \in \mathbb{R}^{L \times D}$, $W_{2} \in \mathbb{R}^{D \times L}$, and $L < D$. This network can hence be simplified as $\hat{x} = W_{2}W_{1}x = Wx$ as the output of the model. The autoencoder is trained for the reconstruction error

$$
\mathcal{L}(W) = \sum_{n=1}^{N} ||x_{n} - Wx_{n}||_{2}^{2}
$$ (ae_loss)

which results in $\hat{W}$ being an orthogonal projection onto the first L eigenvectors of the empirical covariance of the data. Hence being equivalent to performing PCA. This is where the real power of the autoencoder begins as it can be extended in many directions:

* Addition of nonlinearities
* More, and deeper layers
* Usage of specialized architectures like convolutions for more efficient computation

There are no set limits to the addition of further quirks here.

## Further References

* {cite}`murphy2022`, 20.1 PCA and 20.3 Autoencoders
* [Principal component analysis in scikit-learn](https://scikit-learn.org/stable/modules/decomposition.html#principal-component-analysis-pca)
