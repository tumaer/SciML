# Support Vector Machines

Support Vector Machines are one of the most popular classic supervised learning algorithms. In this lecture, we will first discuss the mathematical formalism of solving constrained optimization problems by means of Lagrange multipliers, then we will look at linear binary classification using the maximum margin and soft margin classifiers, and in the end we will present the kernel trick and how it extends the previous two approaches to non-linear boundaries within the more general Support Vector Machine.

## The Constrained Optimization Problem

We define a constraint optimization problem as

$$\underset{\omega}{\min} f(\omega) \quad \text{s.t.} \hspace{5pt} h_{i}(\omega)=0, \hspace{5pt} i=1, \ldots, l$$ (constr_opt)

Here, the $\underset{\omega}{\min}$ seeks to find the minimum subject to the constraint(s) $h_{i}(\omega)$. One possible way to solve this is by using Lagrange multipliers, for which we define the Lagrangian $\mathcal{L}$ which takes the constraints into account.

$$\mathcal{L}(\omega, \beta) = f(\omega) + \sum_{i=1}^{l} \beta_{i} h_{i}(\omega).$$ (constr_opt_lagr)

The $\beta_{i}$ are the *Lagrangian multipliers*, which need to be identified to find the constraint-satisfying $\omega$. The necessary conditions to solve this problem for the optimum are to solve

$$\frac{\partial \mathcal{L}}{\partial \omega_{i}} = 0; \quad \frac{\partial \mathcal{L}}{\partial \beta_{i}} = 0,
$$ (constr_opt_lagr_optimim)

for $\omega$ and $\beta$. In classification problems, we do not only have *equality constraints* as above but can also encounter *inequality constraints*. We formulate the general **primal optimization problem** as:

$$\underset{\omega}{\min} f(\omega) \text{ s.t.}
\begin{cases}
&g_{i}(\omega) \leq 0, \quad i=1, \ldots, k, \\
&h_{j}(\omega) = 0, \quad j=1, \ldots, l.
\end{cases}$$ (primal_problem)

To solve it we define the **generalized Lagrangian**

$$\mathcal{L}(\omega, \alpha, \beta) = f(\omega) + \sum_{i=1}^{k} \alpha_{i} g_{i}(\omega) + \sum_{j=1}^{l} \beta_{i} h_{j}(\omega),$$ (primal_lagr)

with the further Lagrange multipliers $\alpha_i$ and the corresponding optimization problem becomes

$$\theta_{p}(\omega) = \underset{\alpha_{i} \geq 0, \beta_{j}}{\max} \mathcal{L}(\omega, \alpha, \beta).$$ (theta_primal)

Now we can verify that this optimization problem satisfies

$$
\theta_{p}(\omega) =
\begin{cases}
&f(\omega) \quad \text{if } \omega \text{ satisfies the primal constraints,} \\
&\infty \qquad \text{otherwise}.
\end{cases}
$$ (theta_primal_cases)

Where does this case-by-case breakdown come from?

1. In the first case, the constraints are inactive and contribute nil to the sum.
2. In the second case, the sums increase linearly with $\alpha_{i}$, $\beta_{i}$ beyond all bounds.

$$\Longrightarrow p^{\star} = \underset{\omega}{\min} \hspace{2pt} \theta_{p}(\omega) = \underset{\omega}{\min} \hspace{2pt} \underset{\alpha_{i} \geq 0, \beta_{j}}{\max} \mathcal{L}(\omega, \alpha, \beta).$$ (primal_solution)

I.e. we recover the original primal problem with $p^{\star}$ being the *optimal value of the primal problem*. With this we can now formulate the *dual optimization problem*:

$$d^{\star} = \underset{\alpha_{i} \geq 0, \beta_{j}}{\max} \theta_{D}(\alpha, \beta) = \underset{\alpha_{i} \geq 0, \beta_{j}}{\max} \hspace{2pt} \underset{\omega}{\min} \hspace{2pt} \mathcal{L}(\omega, \alpha, \beta),$$ (dual_solution)

with $\theta_{D}(\alpha, \beta) = \underset{\omega}{\min} \hspace{2pt} \mathcal{L}(\omega, \alpha, \beta)$ and $d^{\star}$ the optimal value of the dual problem. Please note that the primal and dual problems are equivalent up to exchanging the order of minimization and maximization. To show how these problems are related, we first derive a relation between min-max and max-min using $\mu(y) = \underset{x}{\inf} \hspace{2pt} \kappa(x, y)$:

$$\begin{aligned}
    &\Longrightarrow \mu(y) \leq \kappa(x, y) \\
    &\Longrightarrow \underset{y}{\sup} \hspace{2pt} \mu(y) \leq \underset{y}{\sup} \hspace{2pt} \kappa(x, y) \\
    &\Longrightarrow \underset{y}{\sup} \hspace{2pt} \mu(y) \leq \underset{x}{\inf} \hspace{2pt} \underset{y}{\sup} \hspace{2pt} \kappa(x, y) \\
    &\Longrightarrow \underset{y}{\sup} \hspace{2pt} \underset{x}{\inf} \hspace{2pt} \kappa(x, y) \leq \underset{x}{\inf} \hspace{2pt} \underset{y}{\sup} \hspace{2pt} \kappa(x, y) \\
\end{aligned}$$ (min_max_and_max_min)

From the last line we can immediately imply the relation between the primal and the dual problems

$$d^{\star} = \underset{\alpha_{i} \geq 0, \beta_{j}}{\max} \hspace{2pt} \underset{\omega}{\min} \hspace{2pt} \mathcal{L}(\omega, \alpha, \beta) \leq \underset{\omega}{\min} \hspace{2pt} \underset{\alpha_{i} \geq 0, \beta_{j}}{\max} \hspace{2pt} \mathcal{L}(\omega, \alpha, \beta) = p^{\star},$$ (dual_smaller_primal)

where the inequality can, under certain conditions, turn into equality. The conditions for this to turn into equality are the following (going back to the Lagrangian form from above):

- $f$ and $g_{i}$ are convex, i.e. their Hessians are positive semi-definite.
- The $h_{i}$ are affine, i.e. they can be expressed as linear functions of their arguments.

Under the above conditions, the following holds:

1. The optimal solution $\omega^{\star}$ to the primal optimization problem exists,
2. The optimal solution $\alpha^{\star}$, $\beta^{\star}$ to the dual optimization problem exists,
3. $p^{\star}=d^{\star}$,
4. $\omega^{\star}$, $\alpha^{\star}$, and $\beta^{\star}$ satisfy the Karush-Kuhn-Tucker (KKT) conditions.

The KKT conditions are expressed as the following conditions:

$$\begin{aligned}
\left. \frac{\partial \mathcal{L}(\omega, \alpha, \beta)}{\partial \omega_{i}} \right|_{\omega^{\star}, \alpha^{\star}, \beta^{\star}} &= 0, \quad i=1, \ldots, n \qquad \text{(KKT1)}\\
\left. \frac{\partial \mathcal{L}(\omega, \alpha, \beta)}{\partial \beta_{i}} \right|_{\omega^{\star}, \alpha^{\star}, \beta^{\star}} &= 0, \quad i=1, \ldots, l \qquad \text{(KKT2)} \\
% The KKT complementarity condition then amounts to: (next three)
\alpha_{i}^{\star} g_{i}(\omega^{\star}) &= 0, \quad i=1, \ldots, k \qquad \text{(KKT3)}\\
g_{i}(\omega^{\star}) &\leq 0, \quad i=1, \ldots, k \qquad \text{(KKT4)}\\
\alpha_{i}^{\star} &\geq 0, \quad i=1, \ldots, k \qquad \text{(KKT5)}
\end{aligned}$$ (kkt)

Moreover, if a set $\omega^{\star}$, $\alpha^{\star}$, and $\beta^{\star}$ satisfies the KKT conditions, then it is a solution to the primal/dual problem. The KKT conditions are sufficient and necessary here. The **dual complementarity condition** (KKT3) indicates whether the $g_{i}(\omega) \leq 0$ constraint is active:

$$\alpha_{i}^{\star} > 0 \Longrightarrow g_{i}(\omega^{\star}) = 0,$$ (kkt_dual_complementarity)

i.e. if $\alpha_{i}^{\star} > 0$, then $\omega^{\star}$ is "on the constraint boundary".s

## Maximum Margin Classifier (MMC)

Alternative names for Maximum Margin Classifier (MMC) are Hard Margin Classifier and Large Margin Classifier. This classifier assumes that the classes are linearly separable.

Now, we can (re-)introduce the *linear discriminator*. Logistic regression $p(y=1| x; \vartheta)$ is then modeled by $h(x) = g(\vartheta^{\top} x) = \text{sigmoid}(\vartheta^{\top} x)$:

```{figure} ../imgs/svm/sigmoid_svm.png
---
width: 400px
align: center
name: sigmoid_svm
---
Sigmoid function.
```

If $g(\vartheta^{\top} x)$ is then close to one, then we have large confidence that $x$ belongs to class $\mathcal{C}_{1}$ with $y=1$, whereas if it is close to $\frac{1}{2}$, we have much less confidence:

```{figure} ../imgs/svm/log_reg_confidence.png
---
width: 500px
align: center
name: log_reg_confidence
---
Confidence of logistic regression classifier.
```

> The intuition here is that we seek to find the model parameters $\vartheta$ such that $g(\vartheta^{\top}x)$ maximizes the distance from the decision boundary $g(\vartheta^{\top}x) = \frac{1}{2}$ for all data points.

For consistency with the standard notation we slightly reformulate this problem setting:

$$\begin{aligned}
y &\in \{ -1, 1 \} \text{ as binary class labels} \\
h(x) &= g(\omega^{\top}x + b) \text{ as classifier with} \\
g(z) &= \begin{cases}
    1, \quad z \geq 0 \\
    -1, \quad z\ < 0,
\end{cases}
\end{aligned}$$ (svm_notation)

where $\omega^{\top} x + b$ defines a hyperplane for our linear classifier. With $b$ we now make the bias explicit, as it was previously implicit in our expressions.

### Functional Margin

The *functional margin* of $(\omega, b)$ w.r.t. a single training sample is defined as

$$\hat{\gamma}^{(i)} = y^{(i)}(\omega^{\top} x^{(i)} + b).$$ (funct_margin)

For a confident prediction we would then like to have a maximum gap between the classes for a good classifier.

```{figure} ../imgs/svm/good_vs_bad_classifier_svm.png
---
width: 400px
align: center
name: good_vs_bad_classifier_svm
---
Good vs bad linear decision boundry.
```

For correctly classified samples we always have

$$y^{(i)}(\omega^{\top} x^{(i)} + b) > 0$$ (funct_margin_inequality)

as $g(\omega^{\top}x + b) = y = \pm 1 $. Note that the induced functional margin is invariant to scaling:

$$g(\omega^{\top}x + b) = g(2\omega^{\top}x + 2b) = \ldots$$ (funct_margin_scalability)

At times this may not be desirable as the classifier does not reflect that the margin itself is **not** invariant. For the entire set of training samples, we can also define the functional margin as

$$\hat{\gamma} = \underset{i}{\min} \hspace{2pt} \hat{\gamma}^{(i)}.$$ (funct_margin_set)

### Geometric Margin

Now we can define the *geometric margin* with respect to a single sample $\gamma^{(i)}$ as follows.

```{figure} ../imgs/svm/geometric_margin.png
---
width: 300px
align: center
name: geometric_margin
---
The geometric margin.
```

The distance $\gamma^{(i)}$ of $x^{(i)}$ from the decision boundary, i.e. from point P, is given by

$$\omega^{\top}\left(x^{(i)} - \gamma^{(i)} \frac{\omega}{||\omega||}\right) + b = 0,$$ (point_to_line_distance)

where $x^{(i)} - \gamma^{(i)} \frac{\omega}{||\omega||}$ gives the location of the point P, and $\frac{\omega}{||\omega||}$ is the unit normal. As P is a member of the decision boundary, no matter where it is placed on the boundary, the equality always has to be zero by definition.

$$\Longrightarrow \gamma^{(i)} = \left( \frac{\omega}{|| \omega ||} \right)^{\top} x^{(i)} + \frac{b}{||\omega||}.$$ (geom_margin_positive)

As this was for an example on the $+$ side we can generalize said expression to obtain

$$\gamma^{(i)} = y^{(i)} \left( \left(\frac{\omega}{|| \omega ||}\right)^{\top} x^{(i)} + \frac{b}{|| \omega ||} \right).$$ (geom_margin)

Please note that the geometric margin indeed is **scale invariant**. Also, note that for $|| \omega ||=1$ the functional and geometric margin are the same. For the entire set of samples, we can then define the geometric margin as:

$$\gamma = \underset{i}{\min} \gamma^{(i)}.$$ (geom_margin_set)

### Maximum Margin Classifier

With these mathematical tools we are now ready to derive the **Support Vector Machine (SVM)** for linearly separable sets (a.k.a. Maximum Margin Classifier) by maximizing the previously derived geometric margin:

$$\underset{\omega, b}{\max} \hspace{2pt} \gamma \quad \text{ s.t. }
\begin{cases}
    &y^{(i)} (\omega^{\top} x^{(i)} + b) \geq \gamma, \quad i=1, \ldots, m \\
    &||\omega|| = 1 \Longrightarrow \hat{\gamma} = \gamma.
\end{cases}
$$ (mmc_primal_geom_margin)

We then seek to reformulate to get rid of the non-convex $||\omega|| = 1$ constraints:

$$\underset{\omega, b}{\max} \frac{\hat{\gamma}}{||\omega||} \quad \text{s.t. } y^{(i)}(\omega^{\top} x^{(i)} + b) \geq \gamma = \frac{\hat{\gamma}}{||\omega||}, \quad i=1, \ldots, m,$$ (mmc_primal_func_margin)

where we applied the definition of $\gamma = \frac{\hat{\gamma}}{||\omega||}$, but in the process suffered a setback as we now have a non-convex objective function. As the geometric margin $\gamma$ is scale-invariant we can now simply scale $||\omega||$ such that $\hat{\gamma} = \gamma ||\omega|| = 1$. Given $\hat{\gamma} = 1$ it is clear that

$$\underset{\omega, b}{\max} \frac{\hat{\gamma}}{||\omega||} = \underset{\omega, b}{\max} \frac{1}{||\omega||} = \underset{\omega, b}{\min} ||\omega||^{2},$$ (mmc_primal_objective)

s.t. the constraints are satisfied. Which is now a convex objective.

> The Support Vector Machine is generated by the primal optimization problem.

$$\underset{\omega, b}{\min} \frac{1}{2} ||\omega||^{2} \quad \text{s.t. } y^{(i)}\left( \omega^{\top} x^{(i)} + b \right) \geq 1, \quad i=1, \ldots, m,$$ (mmc_primal_final)

or alternatively

$$\underset{\omega, b}{\min} \frac{1}{2} ||\omega||^{2} \quad \text{s.t. } g_{i}(\omega) = 1 - y^{(i)}\left( \omega^{\top} x^{(i)} + b \right) \leq 0, \quad i=1, \ldots, m.$$ (mmc_primal_final_with_g)

Upon checking with the KKT dual complementarity condition, we see that $\alpha_{i} > 0$ only for samples with $\gamma_{i}=1$, i.e.

```{figure} ../imgs/svm/maximum_margin_classifier.png
---
width: 400px
align: center
name: maximum_margin_classifier
---
Maximum Margin Classifier.
```

The 3 samples "-", "-", and "+" in the sketch are the only ones for which the KKT constraint is active.

> These are called the **support vectors**.

From the sketch, we can already ascertain that the number of support vectors may be significantly smaller than the number of samples, i.e. also the number of active constraints that we have to take into account. Next, we construct the Lagrangian of the optimization problem:

$$\mathcal{L}(\omega, b, \alpha) = \frac{1}{2} ||\omega||^{2} - \sum_{i=1}^{m} \alpha_{i} \left[ y^{(i)} \left( \omega^{\top} x^{(i)} + b \right) -1 \right]$$ (mmc_lagrangian)

> Note that in this case, our Lagrangian only has inequality constraints!

We can then formulate the dual problem $\theta_{D}(\alpha) = \underset{\omega, b}{\min} \hspace{2pt} \mathcal{L}(\omega, b, \alpha)$ and solve for $\omega$ and $b$ by setting the derivatives to zero.

$$\begin{aligned}
    \nabla_{\omega} \mathcal{L}(\omega, b, \alpha) &= \omega - \sum_{i=1}^{m} \alpha_{i} y^{(i)} x^{(i)} = 0 \\
    \Longrightarrow \omega &= \sum_{i=1}^{m} \alpha_{i} y^{(i)} x^{(i)} \\
    \Longrightarrow \frac{\partial \mathcal{L}}{\partial b} &= \sum_{i=1}^{m} \alpha_{i} y^{(i)} = 0.
\end{aligned}$$ (mmc_derivative_lagrangian)

The $\omega$ we can then resubstitute into the original Lagrangian (Eq. {eq}`mmc_lagrangian`) to get the term

$$\sum_{i=1}^{m} \alpha_{i} y^{(i)} \omega^{\top} x^{(i)} = \omega^{\top} \sum_{i=1}^{m} \alpha_{i} y^{(i)}x^{(i)} = \omega^{\top} \omega = ||\omega||^{2}.$$ (mmc_lagrangian_omega_term)

If we plug that into the Lagrangian (Eq. {eq}`mmc_lagrangian`), we get the dual

$$\begin{aligned}
   \theta_{D}(\alpha)&= - \frac{1}{2} ||\omega||^{2} + \sum_{i=1}^{m} \alpha_{i} - b \sum_{i=1}^{m} \alpha_{i} y^{(i)} \\
    &= - \frac{1}{2} ||\omega||^{2} + \sum_{i=1}^{m} \alpha_{i} \quad \text{(by the derivative w.r.t. $b$)}\\
    &= \sum_{i=1}^{m} \alpha_{i} - \frac{1}{2} \sum_{i, j=1}^{m} y^{(i)}y^{(j)} \alpha_{i} \alpha_{j} x^{(i)^{\top}} x^{(j)},
\end{aligned}$$ (mmc_dual)

which we can then optimize as an optimization problem

$$\underset{\alpha}{\max} \hspace{2pt} \theta_{D}(\alpha) \quad \text{s.t. }
\begin{cases}
    &\alpha_{i} \geq 0, \quad i=1, \ldots, m \\
    &\sum_{i=1}^{m} \alpha_{i} y^{(i)} = 0.
\end{cases}$$ (mmc_dual_optimization)

The first constraint in this optimization problem singles out the support vectors, whereas the second constraint derives itself from our derivation of the dual of the Lagrangian (see above). The KKT conditions are then also satisfied

- The first KKT condition is satisfied as of our first step in the conversion to the Lagrangian dual.
- The second KKT condition is not relevant.
- The third KKT condition is satisfied with $\alpha_{i} > 0 \Leftrightarrow y^{(i)}(\omega^{\top}x^{(i)} + b) = 0$, i.e. the support vectors, and $\alpha_{i}=0 \Leftrightarrow y^{(i)} (\omega^{\top} x^{(i)} + b) < 1$ for the others.
- The fourth KKT condition $y^{(i)} (\omega^{\top} x^{(i)} + b) \leq 1$ is satisfied by our construction of the Lagrangian.
- The fifth KKT condition $\alpha_{i} \geq 0$ is satisfied as of our dual optimization problem formulation.

$$\Longrightarrow d^{\star} = p^{\star}, \text{ i.e. the dual problem solves the primal problem.}$$

The two then play together in the following fashion:

- The dual problem gives $\alpha^{\star}$
- The primal problem gives $\omega^{\star}$, $b^{\star}$, with $b^{\star}$ given by

$$b^{\star} = - \frac{\underset{i \in \mathcal{C}_{2}}{\max} \hspace{2pt} \omega^{\star^{\top}}x^{(i)} + \underset{j \in \mathcal{C}_{1}}{\min} \hspace{2pt} \omega^{\star^{\top}} x^{(j)}}{2}$$ (mmc_bstar)

```{figure} ../imgs/svm/maximum_margin_classifier_solution.png
---
width: 500px
align: center
name: maximum_margin_classifier_solution
---
Maximum Margin Classifier solution.
```

To derive $b^{\star}$, we start from $x^{(i)}$ on the negative margin, one then gets to the decision boundary by $x^{(i)} + \omega^{\star}$, and from $x^{(j)}$ on the positive margin by $x^{(j)} - \omega^{\star}$, i.e.

$$\begin{aligned}
\Longrightarrow \underset{i \in \mathcal{C}_{2}}{\max} \hspace{2pt} \omega^{\star^{\top}} x^{(i)} + \omega^{\star^{\top}} \omega^{\star} + b^{\star} &= 0 \\
\underset{j \in \mathcal{C}_{1}}{\min} \hspace{2pt} \omega^{\star^\top} x^{(j)} - \omega^{\star^\top} \omega^{\star} + b^{\star} &= 0
\end{aligned}$$ (mmc_bstar_derivation)

As $\omega^{\star^\top} \omega^{\star} = 1$, we can then solve for $b^{\star}$ and obtain the result in Eq. {eq}`mmc_bstar`. Now we can check how the SVM predicts $y$ given $x$:

$$\begin{aligned}
\omega^{\star^{\top}}x + b^{\star} &= \left( \sum_{i=1}^{m} \alpha_{i}^{\star} y^{(i)} x^{(i)} \right)^{\top} x + b^{\star} \\
&= \sum_{i=1}^{m} \alpha_{i}^{\star} y^{(i)} \langle x^{(i)}, x \rangle + b^{\star},
\end{aligned}$$ (mmc_model)

where $\alpha_{i}$ is only non-zero for support vectors and calls the inner product $\langle x^{(i)}, x \rangle$ which is hence a highly efficient computation. As such we have derived the support vector machine for the linear classification of sets. The formulation of the optimal set-boundary/decision boundary was formulated as the search for a margin optimization, then transformed into a convex constrained optimization problem, before restricting the contributions of the computation to contributions coming from the _support vectors_, i.e., vectors on the actual decision boundary estimate hence leading to a strong reduction of the problem dimensionality.

## Advanced Topics: Soft Margin Classifier (SMC)

What if the data is not linearly separable?

```{figure} ../imgs/svm/svm_nonlinearly_separable.png
---
width: 500px
align: center
name: svm_nonlinearly_separable
---
Non-linearly separable sets.
```

*Data may not be exactly linealy seperable or some data outliers may undesirably deform the exact decision boundary.*

### Outlier problem

```{figure} ../imgs/svm/svm_sets_with_outlier.png
---
width: 500px
align: center
name: svm_sets_with_outlier
---
Sensitivity of MMC to outlier.
```

### Original SVM optimization problem

$$\min _{\omega, b} \frac{1}{2}\|\omega\|^{2}  \quad \text {s.t. } y^{(i)}\left(\omega^{\top} x^{(i)}+b\right) \ge 1, i=1, \ldots, m$$ (mmc_primal_problem)

To make the algorithm work for non-linearly separable data, we introduce $l_1$-regularization, i.e. a penalty term proportional to the magnitude of a certain quantity

$\Rightarrow l_1$-regularised primal optimization problem

$$\min _{\omega, b} \frac{1}{2}\|\omega\|^{2}+C \sum_{i=1}^{m} \xi_{i}$$
$$\text{s.t.}\left\{\begin{array}{l}y^{(i)}\left(\omega^{\top} x^{(i)}+b\right) \geq 1-\xi_{i}, \quad i=1, \ldots, m \\ \xi_{i} \ge 0, \quad i=1, \ldots, \mathrm{m}\end{array}\right.$$ (smc_primal_problem)

Here, $\xi_i$ is called a "slack" variable. We relax the previous requirement of a unit functional margin $\hat{\gamma}=1$ by allowing some violation, which is penalized in the objective function.

- margin $\hat{\gamma}^{(i)}=1 - \xi_{i}, \quad \xi_{i}>0$
- penalization $C\xi_{i}$
- parameter $C$ controls the weight of the penalization

Then, the Lagrangian of the penalized optimization problem becomes

$$\mathcal{L}(\omega, b, \xi, \alpha, \mu)=\frac{1}{2} \omega^{T} \omega+C \sum_{i=1}^{m} \xi_{i} -\sum_{i=1}^{m} \alpha_{i}\left[y^{(i)}\left(\omega^{\top} x^{(i)}+b\right)-1+\xi_{i}\right]-\sum_{i=1}^{m} \mu_{i} \xi_{i}.$$ (smc_lagrangina)

In the above equation, the second term ($C \sum_{i=1}^{m} \xi_{i}$) represents the soft penalization of strict margin violation, whereas the third and fourth terms are the inequality constraints with Lagrangian multipliers $\alpha_{i}$ and $\mu_{i}$. The derivation of the dual problem follows from the analogous steps of the non-regularised SVM problem.

$$\begin{aligned}
&\frac{\partial \mathcal{L}}{\partial \omega} \stackrel{!}{=} 0  \quad \Rightarrow \quad w=\sum_{i=1}^{m} \alpha_{i} y^{(i)} x^{(i)}\\
&\frac{\partial \mathcal{L}}{\partial b} \stackrel{!}{=} 0  \quad \Rightarrow \quad \sum_{i=1}^{m} \alpha_{i} y^{(i)}=0\\
&\frac{\partial \mathcal{L}}{\partial \xi} \stackrel{!}{=} 0 \quad \Rightarrow \quad \alpha_{i}=C-\mu_i, \quad i=1, \ldots,m \qquad (\star)
\end{aligned}$$ (smc_derivative_lagrangian)

The last equation arises from the additional condition due to slack variables. Upon inserting these conditions into $\mathcal{L}(\omega,b,\xi,\alpha,\mu)$ we obtain the dual-problem Lagrangian:

$$\begin{aligned}
\max_{\alpha} \theta_{D}(\alpha)&=\max_{\alpha} \left( \sum_{i=1}^{m} \alpha_{i}-\frac{1}{2} \sum_{i,j=1}^{m} y^{(i)} y^{(j)} \alpha_{i} \alpha_{j}\left\langle x^{(i)}, x^{(j)}\right\rangle \right) \\
\text { s.t. }&\left\{\begin{array}{l}
0 \leq \alpha_{i} \leq C, \quad i=1, \ldots, m \\
\sum_{i=1}^{m} \alpha_{i} y^{(i)}=0
\end{array}\right.
\end{aligned}$$ (smc_dual_problem)

The "box constraints" ($0 \leq \alpha_{i} \leq C$) follow from the positivity requirement on Lagrange multipliers: $\alpha_{i}\ge 0,\mu_{i}\ge 0$. Evaluating the KKT3-complementarity condition leads to

$$\begin{aligned}
&\alpha_{i}^{*} \left[y^{(i)}\left(\omega^{\top} x^{(i)}+b\right)-1+\xi_{i}\right]=0 \Leftrightarrow \begin{cases}\alpha_{i}^{*}>0, & y^{(i)}\left(\omega^{\top} x^{(i)}+b\right)=1-\xi_{i} \\
\alpha_{i}^{*}=0, & y^{(i)}\left(\omega^{\top} x^{(i)}+b\right) \ge 1-\xi_{i} \end{cases}\\
&\mu_{i}^{*} \xi_{i}=0 \Leftrightarrow \begin{cases}\mu_{i}^{*}>0, & \xi_{i}=0 \\
\mu_{i}^{*}=0, & \xi_{i}>0\end{cases}
\end{aligned}$$ (smc_kkt3)

The resulting dual complementarity conditions for determining the support vectors become:

- Support vectors on slack margin ($\alpha_i=0$, i.e. data point is ignored)

$$\alpha_{i}^{*}=0 \Rightarrow y^{(i)}\left(\omega^{\top} x^{(i)}+b\right) \ge 1$$ (smc_kkt3_alpha_0)

- Support vectors inside or outside margin ($\alpha^*=C$, i.e. data point violates the MMC)

$$\begin{aligned}
C=\alpha_{i}^{*} \quad & (\star) \Rightarrow \mu_{i}^{*}=0,\left\{\begin{array}{rr}
0 < \xi_{i} \leq 1 & \text{(correctly classified)} \\
 \xi_{i}>1 & \text{(misclassified)}\end{array}\right.\\
& \Rightarrow \quad y^{(i)}\left(\omega^{\top} x^{(i)}+b\right) < 1
\end{aligned}$$ (smc_kkt3_alpha_c)

- Support vectors on margin ($\alpha_i^*>0$ & $\xi_i>0$, i.e. data point on the margin)

$$\begin{aligned}
0 < \alpha_{i}^{*}<C \quad & (\star) \Rightarrow \mu_{i}^{*}>0 \quad \Rightarrow \xi_{i}=0 \\
&\Rightarrow \quad y^{(i)}\left(\omega^{\top} x^{(i)}+b\right)=1
\end{aligned}$$ (smc_kkt3_alpha_else)

The optimal $b^*$ is obtained from averaging over all support vectors: condition to be satisfied by $b^*$ is given by SVs on the margin:

$$0<\alpha_{i}^{*}<C \Rightarrow y^{(i)}\left(\omega^{\top} x^{(i)}+b\right)=1$$ (smc_bstar_condition)

$\Rightarrow$ for $\omega^*$ we obtain the same result as for the linearly separable problem

$$\omega^*=\sum_{i=1}^{m_{s}} \alpha_{i}^* y^{(i)} x^{(i)}, \quad m_{s} \text{ support vectors.}$$ (smc_omegastar)

For $b^{\star}$ we then get the condition

$$\Rightarrow y^{(i)}(\omega^* x^{(i)} + b^*) = 1.$$ (smc_bstar_condition2)

A numerically stable option is to average over $m_{\Sigma}$, i.e. all SV on the margin satisfying $0<\alpha_{i}^{*}<C$:

$$b^* = \frac{1}{m_{\Sigma}} \sum_{j=1}^{m_{\Sigma}}\left(y^{(j)}-\sum_{i=1}^{m_{\Sigma}} \alpha_{i}^{*} y^{(i)}\left\langle x^{(i)}, x^{(j)}\right\rangle\right)$$ (smc_bstar)

> Recall: only data with $\alpha_{i}^*\ne 0$, i.e. support vectors, will contribute to the SVM prediction (last eq. of Maximum Margin Classifier).

In conclusion, we illustrate the functionality of the slack variables.

```{figure} ../imgs/svm/soft_margin_classifier.png
---
width: 500px
align: center
name: soft_margin_classifier
---
Soft Margin Classifier.
```

## Advanced Topics: Sequential Minimal Optimization (SMO)

- Efficient algorithm for solving the SVM dual problem
- Based on *Coordinate Ascent* algorithm.

### Coordinate Ascent

- Task : find $\max _{x} f\left(x_{1}, \ldots, x_{m}\right)$
- Perform a component-wise search on $x$

_Algorithm_

**do until converged** <br>
&emsp;**for** $i=1, \ldots, m$ <br>
$\qquad x_{i}^{(k+1)}=\underset{\tilde{x}_{i}}{\operatorname{argmax} } f\left(x_{1}^{(k)}, \ldots, \tilde{x}_{i}, \ldots, x_{m}^{(k)}\right) $ <br>
&emsp;**end for** <br>
**end do**

_Sketch of algorithm_

```{figure} ../imgs/svm/sequential_minimal_optimization.png
---
width: 500px
align: center
name: sequential_minimal_optimization
---
Sequential Minimal Optimization.
```

Coordinate ascent converges for convex continuous functions but may not converge to the dual optimum!

### Outline of SMO for SVM

Task: solve SVM dual optimization problem

$$\begin{aligned}
\max_{\alpha} \theta_{D}(\alpha)&=\max_{\alpha} \left( \sum_{i=1}^{m} \alpha_{i}-\frac{1}{2} \sum_{i,j=1}^{m} y^{(i)} y^{(j)} \alpha_{i} \alpha_{j}\left\langle x^{(i)}, x^{(j)}\right\rangle \right) \\
\text { s.t. }&\left\{\begin{array}{l}
0 \leq \alpha_{i} \leq C, \quad i=1, \ldots, m \\
\sum_{i=1}^{m} \alpha_{i} y^{(i)}=0
\end{array}\right.
\end{aligned}$$ (smc_dual_problem_duplicate)

Consider now an iterative update for finding the optimum:
- Iteration step delivers a constraint satisfying set of $\alpha_{i}$.
- Can we just change some $\alpha_{j} \epsilon$ {$\alpha_{1}$,....,$\alpha_{m}$} according to coordinate ascent for finding the next iteration update?
    - No, because $\sum_{i=1}^{m} \alpha_{i}y^{(i)}=0$ constrains the sum of all $\alpha_{i}$, and varying only a single $\alpha_{p}$ may lead to constraint violation
- Fix it by changing a pair of $\alpha_{p},\alpha_{q}$ , $p\ne q$ , simultaneously.

$\Rightarrow$ the SMO algorithm.

_Algorithm_

**do until convergence criterion satisfied**

1. Given $\alpha^{(k)}$ constraint satisfying
2. Select $\alpha_{p}^{(k+1)}=\alpha_{p}^{(k)}, \; \alpha_{q}^{(k+1)}=\alpha_{q}^{(k)}$ for $p \ne q$ following some estimate which $p,q$ will give fastest ascent
3. Find $\left(\alpha_{p}^{(k+1)}, \alpha_{q}^{(k+1)}\right)=\underset{\alpha_{p},\alpha_{q}}{\operatorname{argmax} } \theta_{D}\left(x_{1}^{(k)}, \ldots, \alpha_{p},  \ldots,\alpha_{q}, \ldots, x_{m}^{(k)}\right)$
// except for $\alpha_{p},\alpha_{q}$ all others are kept fixed.

**end do**

Convergence criterion for SMO: check whether KKT conditions are satisfied up to a chosen tolerance.
  
**Discussion of SMO**

- assume $\alpha^{(k)}$ given with $\sum_{i=1}^{m} y^{(i)}  \alpha^{(k)}_{i}=0$
- pick $\alpha_{1}=\alpha^{(k)}_1$ and  $\alpha_2=\alpha^{(k)}_2$ for optimization

$$\Rightarrow \alpha_{1} y^{(1)}+\alpha_{2} y^{(2)}=-\sum_{i=3}^{m} \alpha_{i} y^{(i)}=\rho \quad$$ (smo_algorithm_example1)

Note that the r.h.s. is constant during the current iteration step.

```{figure} ../imgs/svm/sequential_minimal_optimization_truncation.png
---
width: 500px
align: center
name: sequential_minimal_optimization_truncation
---
Truncation during Sequential Minimal Optimization.
```

The box constraints imply $L \le \alpha_{2} \le H$. Note that depending on the slope of the line, L or H may be clipped by the box constraint.

$\alpha_{1} y^{(1)}+\alpha_{2} y^{(2)}=\rho$

$\Rightarrow \alpha_{1}=\left(\rho-\alpha_{2} y^{(2)}\right) y^{(1)}$

$\rightarrow$ Do you see what happened here?

Answer:

$\alpha_{1} y^{(1)}+\alpha_{2} y^{(2)}=\rho \quad / \cdot y^{(1)}$

$\Rightarrow \alpha_{1} y^{(1)^2}=\left(\rho-\alpha_{2} y^{(2)}\right) y^{(1)}$

$y^{(1)^2}=1,$ as $y \in\{-1,1\}$

$$\Rightarrow \theta_{D}(\alpha)=\sum_{i=1}^{m} \alpha_{i}-\frac{1}{2} \sum_{i,j=1}^{m} y^{(i)} y^{(j)} \alpha_{i} \alpha_{j}\left\langle x^{(i)}, x^{(j)}\right\rangle$$ (smo_algorithm_example2)

*with $\alpha_{1}=\left(\rho-\alpha_{2} y^{(2)}\right) y^{(1)}$ thus becomes a quadratic function of $\alpha_2$ , as all other $\alpha_{j\ne 1,2}$ are fixed.*

- $\theta_D(\alpha_2)=A\alpha^2_2+B\alpha_2+const.$
can be solved for $arg\max_{\alpha_2}\theta_{D}(\alpha_2)=\alpha_2^{'}$

- $\alpha_2^{'}\rightarrow$ box constraints $\rightarrow \alpha_2^{''}$

$$\alpha_2^{''} = \left\{\begin{array}{l}
H , \quad \alpha_{2}^{'}>H \\
\alpha_{2}^{'} , \quad L \leq \alpha_{2}^{'} \leq H\\
L, \quad \alpha_{2}^{'}<L
\end{array}\right.$$ (smo_algorithm_example3)

- set $\alpha_2^{(k+1)} = \alpha_2^{''}$

$$\alpha_{1}^{(k+1)}=\left(\rho-\alpha_{2}^{(k+1)} y^{(2)}\right) y^{(1)}$$ (smo_algorithm_example4)

- next iteration update

## Kernel Methods

Consider binary classification in the non-linearly-separable case, assuming that there is a feature map $\varphi(x) \in\mathbb{R}^n$ transforming the input $x \in\mathbb{R}^d$ to a space which is linearly separable.

```{figure} ../imgs/svm/kernel_feature_map.png
---
width: 600px
align: center
name: kernel_feature_map
---
Feature map transformation.
```

The feature map is essentially a change of basis as:

$$x\rightarrow\varphi(x)$$ (feature_map)

In general, $x$ and $\varphi$ are vectors where $\varphi$ has the entire $x$ as argument. The resulting modified classifier becomes

$$h(x)= g(\omega^T \varphi(x)+b).$$ (classifier_with_feature_map)

**Example: XNOR**

The following classification problem is non-linear as there is no linear decision boundary.

```{figure} ../imgs/svm/xnor_example.png
---
width: 300px
align: center
name: xnor_example
---
XNOR example.
```

Upon defining a feature map $\varphi (x_1,x_2) = x_1 x_2$ (*maps 2D $\rightarrow$ 1D*), we get

```{figure} ../imgs/svm/xnor_example_embedded.png
---
width: 600px
align: center
name: xnor_example_embedded
---
XNOR after feature mapping.
```

**Example: Circular Region**

Given is a set of points $x \in \mathbb{R}^{2}$ with two possible labels: purple ($-1$) and orange ($1$), as can be seen in the left figure below. The task is to find a feature map such that a linear classifier can perfectly separate the two sets.

```{figure} ../imgs/svm/kernel_trick_idea.svg
---
width: 600px
align: center
name: kernel_trick_idea
---
Binary classification of circular region (Source: [Wikipedia](https://en.wikipedia.org/wiki/Kernel_method)).
```

Here, it is again obvious that if we embed the inputs in a 3D space by adding their squares, i.e. $\varphi((x_1, x_2)) = (x_1, x_2, x_1^2+x_2^2)$, we will be able to draw a hyperplane separating the subsets.

But of course, these examples are constructed, as here we could immediately guess $\varphi(x_1,x_2)$. In general, this is not possible.

> Recall : the dual problem of SVM involves a scalar product $x^{(i)\top}x^{(j)}$ of feature vectors.
$\Rightarrow$ motivates the general notation of a dual problem with feature maps.

### Dual representations

Motivated by Least Mean Squares (LMS) regression, we consider the following **regularized** cost function:

$$J(\omega)=\sum_{i=1}^{m}\left(w^{T} \varphi(x^{(i)})-y^{(i)}\right)^{2}+\frac{\lambda}{2} \omega^{T} \omega,$$ (ridge_reg_loss)

with penalty parameter $\lambda \geqslant 0$.

Regularization helps to suppress the overfitting problem (see [Tricks of Optimization](tricks.md)). The squared L2 regularization above is also called **Tikhonov regularization**.  In machine learning, linear regression in combination with Tikhonov regularization is often dubbed **ridge regression**.

Setting the gradient of that loss to zero $\nabla_{\omega}J=0$ we obtain the solution

$$\omega = -\frac{1}{\lambda} \sum_{i=1}^{m}\left(w^{T} \varphi(x^{(i)})-y^{(i)}\right)\varphi(x^{(i)})=\Phi^Ta,$$ (ridge_reg_solution)

with *design matrix* $\Phi$ using the feature map $\varphi$ defined as

$$\Phi=\left[\begin{array}{c}\vdots \\ \varphi\left(x^{(i)}\right) \\ \vdots\end{array}\right] \in \mathbb{R}^{m \times n}$$ (ridge_reg_design_matrix)

and

$$a = -\frac{1}{\lambda} \left[\begin{array}{c}\vdots \\ w^{T} \varphi\left(x^{(i)}\right)-y^{(i)} \\ \vdots\end{array}\right] \in \mathbb{R}^m.$$ (ridge_reg_a)

Substituting the necessary condition  $\omega = \Phi^Ta$ into $J(\omega)$ we obtain the dual problem:

$$\begin{aligned}
J_{D}(a)=& \frac{1}{2} a^T \Phi \Phi^T \Phi \Phi^Ta -a^T \Phi \Phi^T y+\frac{1}{2} y^T y+\frac{\lambda}{2} a^T \Phi \Phi^Ta \\=& \frac{1}{2} a^T K K a-a^T K y+\frac{1}{2} y^T y+\frac{\lambda}{2} a^T Ka
\end{aligned}$$ (ridge_reg_dual)

Here, $K=\Phi\Phi^T$ is a **Grammatrix** generated by a vector $\varphi$ according to $K_{ij} = \langle \varphi_i , \varphi_j \rangle$ where $\langle \cdot , \cdot \rangle$ is an inner product - here $\langle \varphi(x^{(i)}),\varphi(x^{(j)}) \rangle$.

$$K_{ij} = \varphi^T(x^{(i)})\varphi(x^{(j)}) =: K(x^{(i)},x^{(j)}),$$ (kernel_def)

where $K(x^{(i)},x^{(j)})$ is the **kernel function**.

Now we find that we can express the LMS prediction in terms of the kernel function:

$$J_{D}(a)=\frac{1}{2} a^{T} K K a-a^T K y+\frac{1}{2} y^{T} y+\frac{\lambda}{2} a^T Ka.$$ (ridge_reg_dual_with_kernel)

In order to find $\max_{a} J_D (a)$ we set $\nabla_a J_D (a) = 0$.

$$\begin{aligned}
&\Rightarrow Ka - y + \lambda Ia = 0 \\
&\Rightarrow a = (K+\lambda I)^{-1} y
\end{aligned}$$ (ridge_reg_dual_sol_a)

Upon inserting this result into a linear regression model with feature mapping

$$\begin{aligned}
h(x) &= \omega^T \varphi (x) + b = a^T \Phi \varphi (x) = \Phi^T \varphi (x) a \\
&=k^T (K+\lambda I)^{-1} y,
\end{aligned}$$ (ridge_reg_dual_sol)

where $k_i = K(x^{(i)},x)$ are the components of K.

> The *kernel trick* refers to this formulation of the learning problem, which relies on computing the kernel similarities between pairs of input points $x$ and $x'$, instead of computing $\varphi(x)$ explicitly.

> The term *Support Vector Machine* typically refers to a margin classifier using the kernel formulation.

Now, let's do some bookkeeping using the number of data points $M$ and dimension of the input data $N$.

$$\underbrace{a}_{M}=\underbrace{(K-\lambda I )^{-1}}_{M \times M} \underbrace{y}_{M}.$$ (ridge_reg_dual_sol_a_dimensions)

> Recall from the lecture on linear models that $\underbrace{\vartheta}_{N}=\underbrace{(X^{T}X)^{-1}}_{N \times N} \underbrace{X^T}_{N \times M} \underbrace{y}_{M}$

As typically $M>>N$ we see that solving the dual problem for LMS requires us to invert a $M\times M$ matrix, whereas the primal problem tends only to a $N\times N$ matrix.

The benefit of the dual problem with the kernel $K(x,x')$ is that now we can work with the kernel directly
$\Rightarrow$ dimensionality of $\varphi(x)$ matters no longer.
$\Rightarrow$ we can consider even an infinite-dimensional feature vector $N \rightarrow \infty$, i.e. a continuous $\varphi (x)$

### Construction of suitable kernels

- construction from feature map

    $$ K(x,x') = \varphi^T (x) \varphi (x') = \sum_{i=1}^{N} \varphi_i (x) \varphi_i (x')$$ (kernel_from_feature_map)

- direct construction with constraint that a *valid kernel* is obtained, i.e. it needs actually to correspond to a possible feature map scalar product.

A **necessary and sufficient condition** for a valid kernel is that **$K$ is positive semidefinite for all $x$**.

**Example: Kernel 1**

Given is $x, x' \in \mathbb{R}^N$ and a scalar kernel

$$\begin{aligned}
K(x,x') & = (x^Tx')^2 \\
& = \sum_{i=1}^N x_i x_i' \sum_{j=1}^N x_jx_j' \\
& = \sum_i \sum_j x_ix_jx_i'x_j'.
\end{aligned}$$ (kernel_polynomial_2)

The corresponding feature map for $K(x,x') = \varphi^T(x)\varphi(x')$ and with $N=3$ is

$$\varphi (x) = \left[\begin{array}{l} x_1 x_1 \\ x_1 x_2 \\ x_1 x_3 \\ x_2 x_1 \\ \ldots \\ x_3x_3 \end{array}\right]$$ (kernel_polynomial_2_n_3_feature_map)

**Example: Kernel 2**

Alternative kernel with parameter $c$:

$$K(x,x') = (x^Tx'+c)^2 = \sum_{i,j} x_ix_jx_i'x_j' + \sum_i \sqrt{2c} x_i \sqrt{2c} x_j + c^2$$ (kernel_polynomial_2_with_bias)

belongs to

$$\varphi(x)= \left[\begin{array}{l} x_1 x_1 \\ x_1 x_2 \\ \ldots \\ x_3x_3 \\ \sqrt{2c}x_1 \\ \sqrt{2c}x_2 \\ \sqrt{2c}x_3 \\ c  \end{array}\right]$$ (kernel_polynomial_2_with_bias_n_3_feature_map)

Considering that $\varphi(x)$ and $\varphi(x')$ are vectors, the scalar product $K(x,x')=\varphi^T(x)\varphi(x')$ expresses the projection of $\varphi(x')$ onto $\varphi(x)$.

$\Rightarrow$ the larger the kernel value, the more parallel the vectors are. Conversely, the smaller, the more orthogonal they are.

$\Rightarrow$ intuitively $K(x,x')$ is a measure of "how close" $\varphi(x)$ and $\varphi(x')$ are.

**Example: Gaussian kernel**

$$K(x,x')= \exp \left\{- \frac{(x-x')^T (x-x')}{2 \sigma^2} \right\} \\
\left\{\begin{array}{l} \approx 1 , \quad x \text{ and } x' \text{ close}  \\ \approx 0 , \quad x \text{ and } x' \text{ far apart} \end{array}\right.$$ (gaussian_kernel)

Now we show for illustration that a valid kernel is positive semidefinite, which is the above-mentioned necessary condition.

#### Proof:

$$K_{ij} = \varphi^T(x^{(i)})\varphi(x^{(j)}) = \varphi^T(x^{(j)})\varphi(x^{(i)}) = K_{ji}$$ (kernel_symmetry)

$$\begin{aligned}
\Rightarrow (x')^TKx' &= x'_i K_{ij}x'_j \\
& = x'_i \varphi^T(x^{(i)}) \varphi(x^{(j)}) x'_j \\
& = x'_i \varphi_k^{(i)} \varphi_k^{(j)} x'_j \\
& = \sum_k (x_i^{(i)} \varphi_k^{(i)})^2 \geqslant 0
\end{aligned}$$ (kernel_positivity)

The necessary and sufficient condition is due to **[Mercer's theorem](https://en.wikipedia.org/wiki/Mercer%27s_theorem)**:

A given $K: \mathbb{R}^N \times \mathbb{R}^N \rightarrow \mathbb{R}$ is a valid kernel if for any {$x^{(1)},...,x^{(m)}$}, $m<\infty$ the resulting $K$ is positive semidefinite (which implies that it also must be symmetric). For non-separable sets we still can apply MMC or slack-variable SMC: We simply replace the feature-scalar product within the SVM prediction.

$$h(x) = \omega^{*T}x+b^* = \sum_{i=1}^m \alpha_i^* y^{(i)} \langle x^{(i)} , x \rangle + b^*$$ (smc_with_dot_product)

We replace $\langle x^{(i)} , x \rangle$ by $k(x^{(i)},x)$.

$$h(x) = \sum_{i=1}^m \alpha_i^* y^{(i)} k(x^{(i)},x)+ b^*$$ (smc_with_kernel)

This pulls through into the dual problem for determining $\alpha_i^*$ and $b^*$ such that by picking a suitable kernel function we may get close to a linearly separable transformation function without actually performing the feature mapping.

## Recapitulation of SVMs

- Problem of non-linearly separable sets
  - Introduction of controlled violations of the strict margin constraints
  - Introduction of feature maps
- Controlled violation:
 slack variables $\rightarrow l_1$-regularized opt. problem $\rightarrow$ modified dual problem $\rightarrow$ solution by SMO
- Feature map:
  kernel function (tensorial product of feature maps) $\rightarrow$ reformulated dual opt. problem $\rightarrow$ prediction depends only on the kernel function
- Kernel construction:
  a lot of freedom but $\rightarrow$ valid kernel
- Important kernel: Gaussian
- Solution of non-separable problems:
  - Pick a suitable kernel function
  - Run SVM with a kernel function

## Further References

- {cite}`cs229notes`, Chapters 5 und 6
- {cite}`murphy2022`, Section 17.3
- {cite}`bishop2006`, Eppendix E: Lagrange Multipliers
