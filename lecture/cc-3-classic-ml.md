# Core Content 3: Classic ML

## Support Vector Machines

### Linearly Separable Sets - Maximum Margin Classifier (MMC)

Alternative names for Maximum Margin Classifier (MMC) are Hard Margin Classifier and Large Margin Classifier.

First, we need some mathematical formalism to solve constrained optimization problems. If we define our optimization problem as:

$$\underset{\omega}{\min} f(\omega) \quad \text{s.t.} \hspace{2pt} h_{i}(\omega)=0, \hspace{2pt} i=1, \ldots, l$$

Where the $\underset{\omega}{\min}$ seeks to find the minimum subject to the constraint(s) $h_{i}(\omega)$. To solve this we have to define a Lagrangian to take into account the constraints.

$$\mathcal{L}(\omega, \beta) = f(\omega) + \sum_{i=1}^{l} \beta_{i} h_{i}(\omega)$$

The $\beta_{i}$ are the *Lagrangian multipliers*, which need to be identified to find the constraint-satisfying $\omega$. The necessary condition to solve this problem for the optimum is to solve

$$\begin{aligned}
    \frac{\partial \mathcal{L}}{\partial \omega_{i}} &= 0 \\
    \frac{\partial \mathcal{L}}{\partial \beta_{i}} &= 0
\end{aligned}$$

for $\omega$, and $\beta$. In classification problems, we do not only have *equality constraints* as above but can also encounter *inequality constraints*. We formulate the primal optimization problem as:

$$\underset{\omega}{\min} f(\omega) \text{ s.t.} \begin{cases} &g_{i}(\omega) \leq 0, \quad i=1, \ldots, k \\
                    &h_{j}(\omega) = 0, \quad j=1, \ldots, l
       \end{cases}$$

To then define the generalized Lagrangian

$$\mathcal{L}(\omega, \alpha, \beta) = f(\omega) + \sum_{i=1}^{k} \alpha_{i} g_{i}(\omega) + \sum_{j=1}^{l} \beta_{i} h_{j}(\omega)$$

and its corresponding optimization problem

$$
\theta_{p}(\omega) = \underset{\alpha_{i} \geq 0, \beta_{j}}{\max} \mathcal{L}(\omega, \alpha, \beta).
$$

Now we can verify that this optimization problem satisfies

$$
\theta_{p}(\omega) = \begin{cases} &f(\omega), \quad \text{if } \omega \text{ satisfies the primal constraints} \\
&\infty, \quad \text{otherwise}
\end{cases}
$$

Where does this case-by-case breakdown come from?

1. In the first case, the constraints are inactive and contribute nil to the sum.
2. In the second case, the sums increase linearly with $\alpha_{i}$, $\beta_{i}$ beyond all bounds.

$$
\Longrightarrow p^{\star} = \underset{\omega}{\min} \hspace{2pt} \theta_{p}(\omega) = \underset{\omega}{\min} \hspace{2pt} \underset{\alpha_{i} \geq 0, \beta_{j}}{\max} \mathcal{L}(\omega, \alpha, \beta)
$$

i.e. we recover the original primal problem with $p^{\star}$ being the *value of the primal problem*. With this we can now formulate the *dual optimization problem*:

$$
d^{\star} = \underset{\alpha_{i} \geq 0, \beta_{j}}{\max} \theta_{D}(\alpha, \beta) = \underset{\alpha_{i} \geq 0, \beta_{j}}{\max} \hspace{2pt} \underset{\omega}{\min} \hspace{2pt} \mathcal{L}(\omega, \alpha, \beta)
$$

with $\theta_{D}(\alpha, \beta) = \underset{\omega}{\min} \hspace{2pt} \mathcal{L}(\omega, \alpha, \beta)$ and $d^{\star}$ the value of the dual problem. Please note that the primal and dual problems correspond to each other, but exchange the order of minimization and maximization. To show this relation between min-max, and max-min we let $\mu(y) = \underset{x}{\inf} \hspace{2pt} \kappa(x, y)$:

$$
\begin{aligned}
    &\Longrightarrow \mu(y) \leq \kappa(x, y) \\
    &\Longrightarrow \underset{y}{\sup} \hspace{2pt} \mu(y) \leq \underset{y}{\sup} \hspace{2pt} \kappa(x, y) \\
    &\Longrightarrow \underset{y}{\sup} \hspace{2pt} \mu(y) \leq \underset{x}{\inf} \hspace{2pt} \underset{y}{\sup} \hspace{2pt} \kappa(x, y) \\
    &\Longrightarrow \underset{y}{\sup} \hspace{2pt} \underset{x}{\inf} \hspace{2pt} \kappa(x, y) \leq \underset{x}{\inf} \hspace{2pt} \underset{y}{\sup} \hspace{2pt} \kappa(x, y) \\
\end{aligned}
$$

From the last line we can immediately imply the relation between the primal, and the dual problems:

$$
d^{\star} = \underset{\alpha_{i} \geq 0, \beta_{j}}{\max} \hspace{2pt} \underset{\omega}{\min} \hspace{2pt} \mathcal{L}(\omega, \alpha, \beta) \leq \underset{\omega}{\min} \hspace{2pt} \underset{\alpha_{i} \geq 0, \beta_{j}}{\max} \hspace{2pt} \mathcal{L}(\omega, \alpha, \beta) = p^{\star}
$$

where the inequality can, under certain conditions, turn into equality. The conditions for this to turn into equality are the following (going back to the Lagrangian form from above):

- $f$, and $g_{i}$ are convex, i.e. their Hessians are positive semi-definite. The exact definitions of the Hessians will be presented in 3 lectures.
- The $h_{i}$ are affine, i.e. they can be expressed as linear functions of their arguments.

We hence have three conditions fulfilled

1. The optimum solution $\omega^{\star}$ to the primal optimization problem exists
2. The optimum solution $\alpha^{\star}$, $\beta^{\star}$ to the dual optimization problem exists
3. $\omega^{\star}$, $\alpha^{\star}$, and $\beta^{\star}$ satisfy the Karush-Kuhn-Tucker (KKT) conditions.

The KKT conditions are expressed as the following conditions

$$
\begin{aligned}
    \left. \frac{\partial \mathcal{L}(\omega, \alpha, \beta)}{\partial \omega_{i}} \right|_{\omega^{\star}, \alpha^{\star}, \beta^{\star}} &= 0, \quad i=1, \ldots, n \qquad \text{(KKT1)}\\
    \left. \frac{\partial \mathcal{L}(\omega, \alpha, \beta)}{\partial \beta_{i}} \right|_{\omega^{\star}, \alpha^{\star}, \beta^{\star}} &= 0, \quad i=1, \ldots, l \qquad \text{(KKT2)}
\end{aligned}
$$

The KKT complementarity condition then amounts to

$$
\begin{aligned}
    \alpha_{i}^{\star} g_{i}(\omega^{\star}) &= 0, \quad i=1, \ldots, k \qquad \text{(KKT3)}\\
    g_{i}(\omega^{\star}) &\leq 0, \quad i=1, \ldots, k \qquad \text{(KKT4)}\\
    \alpha_{i}^{\star} &\geq 0, \quad i=1, \ldots, k \qquad \text{(KKT5)}
\end{aligned}
$$

Moreover, if a set $\omega^{\star}$, $\alpha^{\star}$, and $\beta^{\star}$ satisfies the KKT conditions it is a solution to the primal/dual problem. The KKT conditions are sufficient and necessary here. The dual complementarity condition indicates whether the $g_{i}(\omega) \leq 0$ constraint is active.

$$
\alpha_{i}^{\star} > 0 \Longrightarrow g_{i}(\omega^{\star}) = 0
$$

i.e. $\omega^{\star}$ is "on the constraint boundary". Now we can (re-)introduce the *linear discriminator*. Logistic regression $p(y=1| x; \theta)$ is then modeled by $h(x) = g(\theta^{\top} x) = \text{sigm}(\theta^{\top} x)$:

<div style="text-align:center">
    <img src="https://i.imgur.com/7VgPzdM.png" alt="drawing" width="400"/>
</div>

If $g(\theta^{\top} x)$ is then close to one, then we have large confidence that $x$ belongs to class $\mathcal{C}_{1}$ with $y=1$, whereas if it is close to $\frac{1}{2}$, we have much less confidence:

<div style="text-align:center">
    <img src="https://i.imgur.com/4uZ08Td.png" alt="drawing" width="500"/>
</div>

> The intuition here is that we seek to find the model parameters $\theta$ such that $g(\theta^{\top}x)$ maximizes the distance from the decision boundary $g(\theta^{\top}x) = \frac{1}{2}$ for all data points.

For consistency with the standard notation we slightly reformulate this problem setting:

$$
\begin{align*}
    y &\in \{ -1, 1 \} \text{ as binary class labels} \\
    h(x) &= g(\omega^{\top}x + b) \text{ as classifier with} \\
    g(z) &= \begin{cases}
        1, \quad z \geq 0 \\
        -1, \quad \ < 0
    \end{cases}
\end{align*}
$$

where $\omega^{\top} x + b$ defines a hyperplane for our linear classifier. With $b$ we now make the bias explicit, where it was previously implicit in our expressions. The functional margin of $(\omega, b)$ w.r.t.  a single training sample then is:

$$
g^{(i)} = y^{(i)}(\omega^{\top} x^{(i)} + b)
$$

For a confident prediction we would then like to have a maximum gap between the classes for a good classifier:

<div style="text-align:center">
    <img src="https://i.imgur.com/eTkx5uU.png" alt="drawing" width="400"/>
</div>

For correctly classified samples we always have

$$
y^{(i)}(\omega^{\top} x^{(i)} + b) > 0
$$

as $g(\omega^{\top}x + b) = y = \pm 1 \leftarrow \omega^{\top}x^{(i)} + b \geq 0$. Note that the induced functional margin is invariant to scaling:

$$
g(\omega^{\top}x + b) = g(2\omega^{\top}x + 2b) = \ldots
$$

at times this may not be desirable as the classifier does not reflect that the margin itself is **not** invariant. For the entire set of training samples, we can also define the functional margin as

$$
\hat{\gamma} = \underset{i}{\min} \hspace{2pt} \hat{\gamma}^{(i)}
$$

Now we can define a geometric margin with respect to a single sample as

<div style="text-align:center">
    <img src="https://i.imgur.com/UrEUiGA.png" alt="drawing" width="300"/>
</div>

The distance $y^{(i)}$ of $x^{(i)}$ from the decision boundary, i.e. from point P is given by 

$$
\omega^{\top}\left(x^{(i)} - \gamma^{(i)} \frac{\omega}{||\omega||}\right) + b = 0
$$

where $x^{(i)} - \gamma^{(i)} \frac{\omega}{||\omega||}$ gives the location of the point P, and $\frac{\omega}{||\omega||}$ is the unit normal. As P is a member of the decision boundary, no matter where it is placed on the boundary, the equality always has to be $0$ by definition.

$$
\Longrightarrow \gamma^{(i)} = \left( \frac{\omega}{|| \omega ||} \right)^{\top} x^{(i)} + \frac{b}{||\omega||}.
$$

As this was for an example on the $+$ side we can generalize said expression to obtain:

$$
\gamma^{(i)} = y^{(i)} \left( \left(\frac{\omega}{|| \omega ||}\right)^{\top} x^{(i)} + \frac{b}{|| \omega ||} \right)
$$

Please note that the geometric margin is indeed **scale invariant**. For the entire set of samples, we can then define the geometric margin as:

$$
\gamma = \underset{i}{\min} \gamma^{(i)}
$$

With these mathematical tools we are now ready to derive the **Support Vector Machine (SVM)** for linearly separable sets by maximizing the previously derived geometric margin:

$$
\underset{\omega, b}{\max} \hspace{2pt} \gamma \quad \text{ s.t. } \begin{cases}
    &y^{(i)} (\omega^{\top} x^{(i)} + b) \geq \gamma, \quad i=1, \ldots, m \\
    &||\omega|| = 1 \Longrightarrow \hat{\gamma} = \gamma
\end{cases}
$$

We then seek to reformulate to get rid of the non-convex $||\omega|| = 1$ constraints:

$$
\underset{\omega, b}{\max} \frac{\hat{\gamma}}{||\omega||} \quad \text{s.t. } y^{(i)}(\omega^{\top} x^{(i)} + b) \geq \gamma = \frac{\hat{\gamma}}{||\omega||}, \quad i=1, \ldots, m
$$

where we applied the definition of $\gamma = \frac{\hat{\gamma}}{||\omega||}$, but in the process suffered a setback as we now have a non-convex objective function. As the geometric margin $\gamma$ is scale-invariant we can now simply scale $||\omega||$ such that $\hat{\gamma} = \gamma ||\omega|| = 1$. Given $\hat{\gamma} = 1$ it is clear that

$$
\underset{\omega, b}{\max} \frac{\hat{\gamma}}{||\omega||} = \underset{\omega, b}{\max} \frac{1}{||\omega||},
$$

s.t. the constraints are satisfied, is the same as

$$
\underset{\omega, b}{\min} ||\omega||^{2} 
$$

s.t. the constraints are satisfied. Which is now a convex objective.

> The Support Vector Machine is generated by the primal optimization problem.

$$
\underset{\omega, b}{\min} \frac{1}{2} ||\omega||^{2} \quad \text{s.t. } y^{(i)}\left( \omega^{\top} x^{(i)} + b \right) \geq 1, \quad i=1, \ldots, m
$$

or alternatively

$$
\underset{\omega, b}{\min} \frac{1}{2} ||\omega||^{2} \quad \text{s.t. } g_{i}(\omega) = 1 - y^{(i)}\left( \omega^{\top} x^{(i)} + b \right) \leq 0, \quad i=1, \ldots, m
$$

Upon checking with the KKT dual complementarity condition, we see that $\alpha_{i} > 0$ only for samples with $\gamma_{i}=1$, i.e.

<div style="text-align:center">
    <img src="https://i.imgur.com/T7dwj4d.png" alt="drawing" width="400"/>
</div>

The 3 samples "-", "-", and "+" in the sketch are the only ones for which the KKT constraint is active.

> These are called the **support vectors**.

From the sketch, we can already ascertain that the number of support vectors may be significantly smaller than the number of samples, i.e. also the number of active constraints that we have to take into account. Next, we construct the Lagrangian of the optimization problem:

$$
\mathcal{L}(\omega, b, \alpha) = \frac{1}{2} ||\omega||^{2} - \sum_{i=1}^{m} \alpha_{i} \left[ y^{(i)} \left( \omega^{\top} x^{(i)} + b \right) -1 \right]
$$

> Note that in this case, our Lagrangian only has inequality constraints!

We can then formulate the dual problem:

1. $\theta_{D}(\alpha) = \underset{\omega, b}{\min} \hspace{2pt} \mathcal{L}(\omega, b, \alpha)$

$$
\begin{aligned}
    \nabla_{\omega} \mathcal{L}(\omega, b, \alpha) &= \omega - \sum_{i=1}^{m} \alpha_{i} y^{(i)} x^{(i)} = 0 \\
    \Longrightarrow \omega &= \sum_{i=1}^{m} \alpha_{i} y^{(i)} x^{(i)} \\
    \Longrightarrow \frac{\partial \mathcal{L}}{\partial b} &= \sum_{i=1}^{m} \alpha_{i} y^{(i)} = 0
\end{aligned}
$$

Which we can then resubstitute into the original Lagrangian

$$
\mathcal{L}(\omega, \alpha, \beta) = \frac{1}{2} ||\omega||^{2} - \sum_{i=1}^{m} \alpha_{i} \left[ y^{(i)} \left( \omega^{\top} x^{(i)} + b \right) - 1 \right]
$$

we now use

$$
\sum_{i=1}^{m} \alpha_{i} y^{(i)} \omega^{\top} x^{(i)} = \omega^{\top} \sum_{i=1}^{m} \alpha_{i} y^{(i)}x^{(i)} = \omega^{\top} \omega = ||\omega||^{2}
$$

in the Lagrangian to arrive at

$$
\begin{aligned}
    &= - \frac{1}{2} ||\omega||^{2} + \sum_{i=1}^{m} \alpha_{i} - b \sum_{i=1}^{m} \alpha_{i} y^{(i)} \\
    &= \sum_{i=1}^{m} \alpha_{i} - \frac{1}{2} \sum_{i, j=1}^{m} y^{(i)}y^{(j)} \alpha_{i} \alpha_{j} x^{(i)^{\top}} x^{(j)} - b \sum_{i=1}^{m}\alpha_{i} y^{(i)}
\end{aligned}
$$

The dual is then

$$
\Longrightarrow \theta_{D}(\alpha) = \sum_{i=1}^{m} \alpha_{i} - \frac{1}{2} \sum_{i,j=1}^{m} y^{(i)} y^{(j)} \alpha_{i} \alpha_{j} x^{(i)^{\top}} x^{(j)}
$$

which we can then optimize as an optimization problem

$$
\underset{\alpha}{\max} \hspace{2pt} \theta_{D}(\alpha) \quad \text{s.t. } \begin{cases}
    &\alpha_{i} \geq 0, \quad i=1, \ldots, m \\
    &\sum_{i=1}^{m} \alpha_{i} y^{(i)} = 0
\end{cases}
$$

The first constraint in this optimization problem singles out the support vectors, whereas the second constraint derives itself from our derivation of the dual of the Lagrangian (see above). The KKT conditions are then also satisfied

- The first KKT condition is satisfied as of our first step in the conversion to the Lagrangian dual.
- The second KKT condition is not relevant.
- The third KKT condition is satisfied with $\alpha_{i} > 0 \Leftrightarrow y^{(i)}(\omega^{\top}x^{(i)} + b) = 0$, i.e. the support vectors, and $\alpha_{i}=0 \Leftrightarrow y^{(i)} (\omega^{\top} x^{(i)} + b) < 1$ for the others.
- The fourth KKT condition $y^{(i)} (\omega^{\top} x^{(i)} + b) \leq 1$ is satisfied by our construction of the Lagrangian.
- The fifth KKT condition $\alpha_{i} \geq 0$ is satisfied as of our dual optimization problem formulation.

$$
\Longrightarrow d^{\star} = p^{\star}, \text{ i.e. the dual problem solves the primal problem.}
$$

The two then play together in the following fashion:

- The dual problem gives $\alpha^{\star}$
- The primal problem gives $\omega^{\star}$, $b^{\star}$, with $b^{\star}$ given by

$$
b^{\star} = - \frac{\underset{i \in \mathcal{C}_{2}}{\max} \hspace{2pt} \omega^{\star^{\top}}x^{(i)} + \underset{j \in \mathcal{C}_{1}}{\min} \hspace{2pt} \omega^{\star^{\top}} x^{(j)}}{2}
$$

<div style="text-align:center">
    <img src="https://i.imgur.com/hvW7yvx.png" alt="drawing" width="500"/>
</div>

From the $x^{(i)}$ on the negative margin one then gets to the decision boundary by $x^{(i)} + \omega^{\star}$, and from $x^{(j)}$ on the positive margin by $x^{(j)} - \omega^{\star}$, i.e.

$$
\begin{aligned}
    \Longrightarrow \underset{i \in \mathcal{C}_{2}}{\max} \hspace{2pt} \omega^{\star^{\top}} x^{(i)} + \omega^{\star^{\top}} \omega^{\star} + b^{\star} &= 0 \\
    \underset{j \in \mathcal{C}_{1}}{\min} \hspace{2pt} \omega^{\star^\top} x^{(j)} - \omega^{\star^\top} \omega^{\star} + b^{\star} &= 0
\end{aligned}
$$

As $\omega^{\star^\top} \omega^{\star} = 1$, we can then solve for $b^{\star}$

$$
\Longrightarrow b^{\star} = - \frac{1}{2} \left( \underset{i \in \mathcal{C}_{2}}{\max} \hspace{2pt} \omega^{\star^{\top}} x^{(i)} + \underset{j \in \mathcal{C}_{1}}{\min} \hspace{2pt} \omega^{\star^\top} x^{(j)} \right)
$$

Now we can check how the SVM predicts $y$ given $x$:

$$
\begin{aligned}
    \omega^{\star^{\top}}x + b^{\star} &= \left( \sum_{i=1}^{m} \alpha_{i}^{\star} y^{(i)} x^{(i)} \right)^{\top} x + b \\
    &= \sum_{i=1}^{m} \alpha_{i}^{\star} y^{(i)} \langle x^{(i)}, x \rangle + b
\end{aligned}
$$

where $\alpha_{i}$ is only non-zero for support vectors and calls the inner product $\langle x^{(i)}, x \rangle$ which is hence a highly efficient computation. As such we have derived the support vector machine for the linear classification of sets. The formulation of the optimum set-boundary/decision boundary was formulated as the search for a margin optimization, then transformed into a convex constrained optimization problem, before restricting the contributions of the computation to contributions coming from the *support vectors*, i.e., vectors on the actual decision boundary estimate hence leading to a strong reduction of the problem dimensionality.

### Non-Separable Sets - Soft Margin Classifier (SMC)

<div style="text-align:center">
    <img src="https://i.imgur.com/PMZV7it.png" alt="drawing" width="500"/>
</div>

*Data may not be exactly linealy seperable or some data outliers may undesirably deform the exact decision boundary.*

**Outlier problem**

<div style="text-align:center">
    <img src="https://i.imgur.com/8s61hRD.png" alt="drawing" width="500"/>
</div>

**Original SVM optimization problem**

$$
\min _{\omega, b} \frac{1}{2}\|\omega\|^{2}  \quad \text {s.t. } y^{(i)}\left(\omega^{\top} x^{(i)}+b\right) \ge 1, i=1, \ldots, m
$$

To make the algorithm work for non-linearly separable data, we introduce $l_1$-regularization, i.e. a penalty term proportional to the magnitude of a certain quantity

$\Rightarrow l_1$-regularised primal optimization problem


$$
\min _{\omega, b} \frac{1}{2}\|\omega\|^{2}+C \sum_{i=1}^{m} \xi_{i}$$
$$\text{s.t.}\left\{\begin{array}{l}y^{(i)}\left(\omega^{\top} x^{(i)}+b\right) \geq 1-\xi_{i}, \quad i=1, \ldots, m \\ \xi_{i} \ge 0, \quad i=1, \ldots, \mathrm{m}\end{array}\right.$$

Here, $\xi_i$ is called a "slack" variable. We relax the previous requirement of a unit functional margin $\hat{\gamma}=1$ by allowing some violation, which is penalized in the objective function.

* margin $\hat{\gamma}^{(i)}=1 - \xi_{i}, \quad \xi_{i}>0$
* penalization $C\xi_{i}$
* parameter $C$ controls the weight of the penalization

Then, the Lagrangian of the penalized optimization problem becomes

$\mathcal{L}(\omega, b, \xi, \alpha, \mu)=\frac{1}{2} \omega^{T} \omega+C \sum_{i=1}^{m} \xi_{i} -\sum_{i=1}^{m} \alpha_{i}\left[y^{(i)}\left(\omega^{\top} x^{(i)}+b\right)-1+\xi_{i}\right]-\sum_{i=1}^{m} \mu_{i} \xi_{i}.$

In the above equation, the second term ($C \sum_{i=1}^{m} \xi_{i}$) represents the soft penalization of strict margin violation, whereas the third and fourth terms are the inequality constraints with Lagrangian multipliers $\alpha_{i}$ and $\mu_{i}$. The derivation of the dual problem follows from the analogous steps of the non-regularised SVM problem.

$$
\begin{aligned}
&\frac{\partial \mathcal{L}}{\partial \omega} \stackrel{!}{=} 0  \quad \Rightarrow \quad w=\sum_{i=1}^{m} \alpha_{i} y^{(i)} x^{(i)}\\
&\frac{\partial \mathcal{L}}{\partial b} \stackrel{!}{=} 0  \quad \Rightarrow \quad \sum_{i=1}^{m} \alpha_{i} y^{(i)}=0\\
&\frac{\partial \mathcal{L}}{\partial \xi} \stackrel{!}{=} 0 \quad \Rightarrow \quad \alpha_{i}=C-\mu_i, \quad i=1, \ldots,m \qquad (\star)
\end{aligned}
$$

The last equation arises from the additional condition due to slack variables. Upon inserting these conditions into $\mathcal{L}(\omega,b,\xi,\alpha,\mu)$ we obtain the dual-problem Lagrangian:

$$
\max_{\alpha} \theta_{D}(\alpha)=\sum_{i=1}^{m} \alpha_{i}-\frac{1}{2} \sum_{i,j=1}^{m} y^{(i)} y^{(j)} \alpha_{i} \alpha_{j}\left\langle x^{(i)}, x^{(j)}\right\rangle
$$

$$
\text { s.t. }\left\{\begin{array}{l}
0 \leq \alpha_{i} \leq C, \quad i=1, \ldots, m \\
\sum_{i=1}^{m} \alpha_{i} y^{(i)}=0
\end{array}\right.
$$

The "box constraints" ($0 \leq \alpha_{i} \leq C$) follow from the positivity requirement on Lagrange multipliers: $\alpha_{i}\ge 0,\mu_{i}\ge 0$.

Evaluating the KKT3-complementarity condition leads to

$$
\begin{aligned}
&\alpha_{i}^{*} \left[y^{(i)}\left(\omega^{\top} x^{(i)}+b\right)-1+\xi_{i}\right]=0 \Leftrightarrow \begin{cases}\alpha_{i}^{*}>0, & y^{(i)}\left(\omega^{\top} x^{(i)}+b\right)=1-\xi_{i} \\ 
\alpha_{i}^{*}=0, & y^{(i)}\left(\omega^{\top} x^{(i)}+b\right) > 1-\xi_{i} \end{cases}\\
&\mu_{i}^{*} \xi_{i}=0 \Leftrightarrow \begin{cases}\mu_{i}^{*}>0, & \xi_{i}=0 \\ 
\mu_{i}^{*}=0, & \xi_{i}>0\end{cases}
\end{aligned}
$$

The resulting dual complementarity conditions for determining the support vectors become: 

- Support vectors on slack margin ($\alpha_i=0$, i.e. data point is ignored)

$$
\alpha_{i}^{*}=0 \Rightarrow y^{(i)}\left(\omega^{\top} x^{(i)}+b\right)=1-\xi_{i}
$$

- Support vectors inside or outside margin ($\alpha^*=C$, i.e. data point violates the MMC)

$$\begin{aligned}
C=\alpha_{i}^{*} \quad & (\star) \Rightarrow \mu_{i}^{*}=0,\left\{\begin{array}{rr}
0 < \xi_{i} \leq 1 & \text{(correctly classified)} \\ 
 \xi_{i}>1 & \text{(misclassified)}\end{array}\right.\\
& \Rightarrow \quad y^{(i)}\left(\omega^{\top} x^{(i)}+b\right) < 1
\end{aligned}$$

- Support vectors on margin ($\alpha_i^*>0$ & $\xi_i>0$, i.e. data point on the margin)

$$\begin{aligned}
0 < \alpha_{i}^{*}<C \quad & (\star) \Rightarrow \mu_{i}^{*}>0 \quad \Rightarrow \xi_{i}=0 \\
&\Rightarrow \quad y^{(i)}\left(\omega^{\top} x^{(i)}+b\right)=1
\end{aligned}$$

The optimum $b^*$ is obtained from averaging over all support vectors: condition to be satisfied by $b^*$ is given by SVs on the margin:

$$0<\alpha_{i}^{*}<C \Rightarrow y^{(i)}\left(\omega^{\top} x^{(i)}+b\right)=1$$

$\Rightarrow$ for $b^*$ we obtain the same result as for the linearly separable problem 

$$\omega^*=\sum_{i=1}^{m_{s}} \alpha_{i}^* y^{(i)} x^{(i)}, \quad m_{s} \text{ support vectors.}$$

$$\Rightarrow y^{(i)}(\omega^* x^{(i)} + b^*) = 1.$$

A numerically stable option is to average over $m_{\Sigma}$, i.e. all SV on the margin satisfying $0<\alpha_{i}^{*}<C$:


$$b^* = \frac{1}{m_{\Sigma}} \sum_{j=1}^{m_{\Sigma}}\left(y^{(j)}-\sum_{i=1}^{m_{s}} \alpha_{i}^{*} y^{(i)}\left\langle x^{(i)}, x^{(j)}\right\rangle\right)
$$

> Recall: only data with $\alpha_{i}^*\ne 0$, i.e. support vectors, will contribute to the SVM prediction (last eq. of Linearly Separable Sets).

In conclusion, we illustrate the functionality of the slack variables.

<div style="text-align:center">
    <img src="https://i.imgur.com/luacvXq.png" alt="drawing" width="500"/>
</div>

### Sequential Minimal Optimization (SMO)

- Efficient algorithm for solving the SVM dual problem
- Based on *Coordinate Ascent* algorithm.

#### Coordinate Ascent

- Task : find $\max _{x} f\left(x_{1}, \ldots, x_{m}\right)$
- Perform a component-wise search on $x$

_Algorithm_

**do until converged** <br>
&emsp;**for** $i=1, \ldots, m$ <br>
$\qquad x_{i}^{(k+1)}=\underset{\tilde{x}_{i}}{\operatorname{argmax} } f\left(x_{1}^{(k)}, \ldots, \tilde{x}_{i}, \ldots, x_{m}^{(k)}\right) $ <br>
&emsp;**end for** <br>
**end do**


_Sketch of algorithm_

<div style="text-align:center">
    <img src="https://i.imgur.com/2mNIzO1.png" alt="drawing" width="500"/>
</div>

Coordinate ascent converges for convex continuous functions but may not converge to the dual optimum!

#### Outline of SMO for SVM

Task: solve SVM dual optimization problem

$$
\max_{\alpha} \theta_{D}(\alpha)=\sum_{i=1}^{m} \alpha_{i}-\frac{1}{2} \sum_{i,j=1}^{m} y^{(i)} y^{(j)} \alpha_{i} \alpha_{j}\left\langle x^{(i)}, x^{(j)}\right\rangle
$$

$$
\text { s.t. }\left\{\begin{array}{l}
0 \leq \alpha_{i} \leq C, \quad i=1, \ldots, m \\
\sum_{i=1}^{m} \alpha_{i} y^{(i)}=0
\end{array}\right.
$$

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

* assume $\alpha^{(k)}$ given with $\sum_{i=1}^{m} y^{(i)}  \alpha^{(k)}_{i}=0$
* pick $\alpha_{1}=\alpha^{(k)}_1$ and  $\alpha_2=\alpha^{(k)}_2$ for optimization

$$
\Rightarrow \alpha_{1} y^{(1)}+\alpha_{2} y^{(2)}=-\sum_{i=3}^{m} \alpha_{i} y^{(i)}=\rho \quad
$$

Note that the r.h.s. is constant during the current iteration step.

<div style="text-align:center">
    <img src="https://i.imgur.com/Pk0o5BN.png" alt="drawing" width="500"/>
</div>

The box constraints imply $L \le \alpha_{2} \le H$. Note that depending on the slope of the line, L or H may be clipped by the box constraint.

$$
\begin{aligned}
&\alpha_{1} y^{(1)}+\alpha_{2} y^{(2)}=\rho \\
&\Rightarrow \alpha_{1}=\left(\rho-\alpha_{2} y^{(2)}\right) y^{(1)}
\end{aligned}
$$

$\rightarrow$ Do you see what happened here?

Answer:

$\alpha_{1} y^{(1)}+\alpha_{2} y^{(2)}=\rho \quad / \cdot y^{(1)}$

$\Rightarrow \alpha_{1} y^{(1)^2}=\left(\rho-\alpha_{2} y^{2}\right) y^{(1)}$

$y^{(1)^2}=1,$ as $y \in\{-1,1\}$

$$\Rightarrow \theta_{D}(\alpha)=\sum_{i=1}^{m} \alpha_{i}-\frac{1}{2} \sum_{i,j=1}^{m} y^{(i)} y^{(j)} \alpha_{i} \alpha_{j}\left\langle x^{(i)}, x^{(j)}\right\rangle$$

*with $\alpha_{1}=\left(\rho-\alpha_{2} y^{(2)}\right) y^{(1)}$ thus becomes a quadratic function of $\alpha_2$ , as all other $\alpha_{j\ne 1,2}$ are fixed.*

* $\theta_D(\alpha_2)=A\alpha^2_2+B\alpha_2+const.$ 
can be solved for $arg\max_{\alpha_2}\theta_{D}(\alpha_2)=\alpha_2^{'}$

* $\alpha_2^{'}\rightarrow$ box constraints $\rightarrow \alpha_2^{''}$

$$\alpha_2^{''} = \left\{\begin{array}{l}
H , \quad \alpha_{2}^{'}>H \\
\alpha_{2}^{'} , \quad L \leq \alpha_{2}^{'} \leq H\\
L, \quad \alpha_{2}^{'}<L
\end{array}\right.$$

* set $\alpha_2^{(k+1)} = \alpha_2^{''}$

$$
\alpha_{1}^{(k+1)}=\left(\rho-\alpha_{2}^{(k+1)} y^{(1)}\right) y^{(1)}
$$

* next iteration update

### Kernel Methods


Consider binary classification in the non-linearly-separable case: 

<div style="text-align:center">
    <img src="https://i.imgur.com/uWMk3qq.png" alt="drawing" width="600"/>
</div>

<br>

*Kernel trick* or kernel substitution : 

$$x\rightarrow\varphi(x)$$

- *in general $x$ and $\varphi$ are vectors where $\varphi$ has the entire $x$ as argument*

$\Rightarrow$ modified classifyer 

$$h(x)= g(\omega^T \varphi(x)+b)$$

*e.g. such that the above problem becomes linearly separable*  

**Example**

<div style="text-align:center">
    <img src="https://i.imgur.com/qo0P4Yz.png" alt="drawing" width="300"/>
</div>

This classification problem is non-linear as there is no linear decision boundary.

Define feature map $\varphi (x_1,x_2) = x_1 x_2$  (*maps 2D $\rightarrow$ 1D*)

<div style="text-align:center">
    <img src="https://i.imgur.com/y55rIlC.png" alt="drawing" width="600"/>
</div>

But of course, this is constructed, as then we could immediately guess $\varphi(x_1,x_2)$. In general, this is not possible.

> Recall : the dual problem of SVM involves a scalar product $x^{(i)T}x^{(j)}$ of feature vectors.
$\Rightarrow$ motivates the general notation of a dual problem with feature maps.

#### Dual representations

Motivated by Least Mean Squares (LMS) regression, we consider the following **regularized** cost function: 

$$
J(\omega)=\sum_{i=1}^{m}\left(w^{T} \varphi\left(x^{(i)}\right)-y^{(i)}\right)^{2}+\frac{\lambda}{2} \omega^{T} \omega
$$

with penalty parameter $\lambda \geqslant 0$. 

> Notation alert: in Core Content 1 we used $\vartheta$ for the parameters instead of the $\omega$ used here.

Regularization helps to suppress the overfitting problem (see Core Content 2). The squared L2 regularization above is also called **Tikhonov regularization**.  In machine learning, linear regression in combination with Tikhonov regularization is often dubbed **ridge regression**.

$$\nabla_{\omega}J=0$$

$$\Rightarrow \omega = -\frac{1}{\lambda} \sum_{i=1}^{m}\left(w^{T} \varphi\left(x^{(i)}\right)-y^{(i)}\right)\varphi(x^{(j)})=\Phi^Ta$$

with design matrix $\Phi$ with feature map $\varphi$ defined as

$$\Phi=\left[\begin{array}{l}\ldots \\ \varphi\left(x^{(j)}\right) \\ \cdots\end{array}\right] \in \mathbb{R}^{m \times n}$$

and 

$$a_i = -\frac{1}{\lambda}\left(w^{T} \varphi\left(x^{(i)}\right)-y^{(i)}\right) \in \mathbb{R}^m$$

Substituting the necessary condition  $\omega = \Phi^Ta$ into $J(\omega)$ we obtain the dual problem:

$$\begin{aligned} J_{D}(a)=& \frac{1}{2} a^T \Phi \Phi^T \Phi \Phi^Ta -a^T \Phi \Phi^T y+\frac{1}{2} y^T y+\frac{\lambda}{2} a^T \Phi \Phi^Ta \\=& \frac{1}{2} a^T K K a-a^T K y+\frac{1}{2} y^T y+\frac{\lambda}{2} a^T Ka \end{aligned}$$

Here, $K=\Phi\Phi^T$ is a **Grammatrix** generated by a vector $\varphi$ according to $K_{ij} = \langle \varphi_i , \varphi_j \rangle$ where $\langle \cdot , \cdot \rangle$ is an inner product - here $\langle \varphi(x^{(i)}),\varphi(x^{(j)}) \rangle$.

$$K_{ij} = \varphi^T(x^{(i)})\varphi(x^{(j)}) =: K(x^{(i)},x^{(j)})$$

where $K(x^{(i)},x^{(j)})$ is the **kernel function**.

Now we find that we can express the LMS prediction in terms of the kernel function:

$$J_{D}(a)=\frac{1}{2} a^{T} K K a-a^T K y-\frac{1}{2} y^{T} y+\frac{\lambda}{2} a^T K a$$

In order to find $\max_{a} J_D (a)$ we set $\nabla_a J_D (a) = 0$ 

$$\Rightarrow Ka - y + \lambda Ia = 0$$

$$\Rightarrow a = (K+\lambda I)^{-1} y$$

Upon inserting this result into a linear regression model with feature mapping 

$$\begin{aligned}
h(x) &= \omega^T \varphi (x) + b = a^T \Phi \varphi (x) = \Phi^T \varphi (x) a \\
&=K^T (K+\lambda I)^{-1} y
\end{aligned}$$

where $K_i = K(x^{(i)},x)$ are the components of K.

Now, let's do some bookkeeping 

$$\underbrace{a}_{M}=\underbrace{(K-\lambda I )^{-1}}_{M \times M} \underbrace{y}_{M}, \quad M = \text{number of data points}$$


> Recall from Class 2
$$\underbrace{\vartheta}_{N}=\underbrace{(X^{T}X)^{-1}}_{N \times N} \underbrace{X^T}_{N \times M} \underbrace{y}_{M}, \quad N = \text{dim of input data vector}$$


As typically $M>>N$ we see that solving the dual problem for LMS requires us to invest a $MxM$ matrix, whereas the primal problem tends only to a $NxN$ matrix.

The benefit of the dual problem with the kernel $K(x,x')$ is that now we can work with the kernel directly
$\Rightarrow$ dimensionality of $\varphi(x)$ matters no longer.
$\Rightarrow$ we can consider even an infinite-dimensional feature vector $N \rightarrow \infty$, i.e. a continuous $\varphi (x)$


#### Construction of suitable kernels

* construction from feature map 
$$ K(x,x') = \varphi^T (x) \varphi (x') = \sum_{i=1}^{N} \varphi_i (x) \varphi_i (x')$$

* direct construction with constraint that a *valid kernel* is obtained, i.e. it needs actually to correspond to a possible feature-map scalar product.

A **necessary and sufficient condition** for a valid kernel is that **$K$ is positive semidefinite for all $x$**.

**Examples (1)** 

Given is $x, x' \in \mathbb{R}^N$ and a scalar kernel 

$$\begin{aligned}
K(x,x') & = (x^Tx')^2 \\
& = \sum_{i=1}^N x_i x_i' \sum_{j=1}^N x_jx_j' \\
& = \sum_i \sum_j x_ix_jx_i'x_j'.
\end{aligned}$$

The corresponding feature map for 

$$K(x,x') = \varphi^T(x)\varphi(x')$$ 

for $N=3$ is

$$\varphi (x) = \left[\begin{array}{l} x_1 x_1 \\ x_1 x_2 \\ x_1 x_3 \\ x_2 x_1 \\ \ldots \\ x_3x_3 \end{array}\right] $$

**Examples (2)** 

Alternative kernel with parameter $c$:

$$K(x,x') = (x^Tx'+c)^2 = \sum_{i,j} x_ix_jx_i'x_j' + \sum_i \sqrt{2c} x_i \sqrt{2c} x_j + c^2$$

belongs to 

$$\varphi(x)= \left[\begin{array}{l} x_1 x_1 \\ x_1 x_2 \\ \ldots \\ x_3x_3 \\ \sqrt{2c}x_1 \\ \sqrt{2c}x_2 \\ \sqrt{2c}x_3 \\ c  \end{array}\right]$$

<br>

Considering that $\varphi(x)$ and $\varphi(x')$ are vectors, the scalar product $K(x,x')=\varphi^T(x)\varphi(x')$ expresses the projection of $\varphi(x')$ onto $\varphi(x)$.

$\Rightarrow$ the larger the kernel value, the more parallel the vectors are. Conversely, the smaller, the more orthogonal they are.

$\Rightarrow$ intuitively $K(x,x')$ is a measure of "how close" $\varphi(x)$ and $\varphi(x')$ are.

**Example: Gaussian kernel**

$$K(x,x')= \exp \left\{- \frac{(x-x')^T (x-x')}{2 \sigma^2} \right\} \\
\left\{\begin{array}{l} \approx 1 , \quad x \text{ and } x' \text{ close}  \\ \approx 0 , \quad x \text{ and } x' \text{ far apart} \end{array}\right.$$

Now we show for illustration that a valid kernel is positive semidefinite, this is the above-mentioned necessary condition.

**Proof:**

$$K_{ij} = \varphi^T(x^{(i)})\varphi(x^{(j)}) = \varphi^T(x^{(j)})\varphi(x^{(i)}) = K_{ji}$$


$$\begin{aligned} \Rightarrow (x')^TKx' &= x'_i K_{ij}x'_j \\ 
& = x'_i \varphi^T(x^{(i)}) \varphi(x^{(j)}) x'_j \\
& = x'_i \varphi_k^{(i)} \varphi_k^{(j)} x'_j \\
& = \sum_k (x_i^{(i)} \varphi_k^{(i)})^2 \geqslant 0 \end{aligned}$$


The necessary and sufficient condition is due to **[Mercer's theorem](https://en.wikipedia.org/wiki/Mercer%27s_theorem)**: 

A given $K: \mathbb{R}^N \times \mathbb{R}^N \rightarrow \mathbb{R}$ is a valid kernel if for any {$x^{(1)},...,x^{(m)}$}, $m<\infty$ the resulting $K$ is positive semidefinite (which implies that it also must be symmetric).

<br>

For non-separable sets we still can apply MMC or slack-variable SMC: We simply replace the feature-scalar product within the SVM prediction.

$$h(x) = \omega^{*T}x+b^* = \sum_{i=1}^m \alpha_i^* y^{(i)} \langle x^{(i)} , x \rangle + b^*$$

We replace $\langle x^{(i)} , x \rangle$ by $K(x^{(i)},x)$.

$$h(x) = \sum_{i=1}^m \alpha_i^* y^{(i)} K(x^{(i)},x)+ b^*$$

This pulls through into the dual problem for determining $\alpha_i^*$ and $b^*$ such that by picking a suitable kernel function we may get close to a linearly separable transformation function without actually performing the feature mapping.

### Recapitulation of SVMs

* Problem of non-linearly separable sets
  * introduction of controlled violations of the strict margin constraints
  * introduction of feature maps

* Controlled violation: 
 slack variables $\rightarrow l_1$-regularized opt. problem $\rightarrow$ modified dual problem $\rightarrow$ solution by SMO

* Feature map: 
  kernel function (tensorial product of feature maps) $\rightarrow$ reformulated dual opt. problem $\rightarrow$ prediction depends only on the kernel function
  

* Kernel construction: 
  a lot of freedom but $\rightarrow$ valid kernel
  
* Important kernel: Gaussian

* Solution of non-separable problems: 
  * pick a suitable kernel function
  * run SVM with a kernel function


## Gaussian Processes

As the main mathematical construct behind Gaussian Processes, we first introduce the Multivariate Gaussian distribution. We will analyze this distribution in some more detail to provide reference results. For a more detailed derivation of the results, refer to [Bishop, 2006](https://link.springer.com/book/9780387310732), Section 2.3.

### Multivariate Gaussian Distribution

The **univariate** (for a scalar random variable) Gaussian distribution has the form

$$mathcal{N}(x; \underbrace{\mu, \sigma^2}_{\text{parameters}}) = \frac{1}{\sqrt{2 \pi \sigma^2}} \exp\left\{ - \frac{(x-\mu)^2}{2 \sigma^2}\right\}.$$

The two parameters are the mean $\mu$ and variance $\sigma^2$.

The multivariate Gaussian distribution then has the following form


$$\mathcal{N}(x; \mu, \Sigma)= \frac{1}{(2\pi)^{d/2}\sqrt{\det (\Sigma)}} \exp \left(-\frac{1}{2}(x-\mu)^{\top}\Sigma^{-1}(x-\mu)\right),$$

with 

- $x \in \mathbb{R}^d \quad $ - feature / sample / random vector
- $\mu \in \mathbb{R}^d \quad $ - mean vector
- $\Sigma \in \mathbb{R}^{d \times d} \quad$ - covariance matrix

Properties:
- $\Delta^2=(x-\mu)^{\top}\Sigma^{-1}(x-\mu)$ is a quadratic form.
- $\Delta$ is the Mahalanobis distance from $\mu$ to $x$. It collapses to the Euclidean distance for $\Sigma = I$.
- $\Sigma$ is symmetric positive semi-definite and its diagonal elements contain the variance, i.e. covariance with itself.
- $\Sigma$ can be diagonalized with real eigenvalues $\lambda_i$

$$\sum u_i = \lambda_i u_i \quad i=1,...,d$$

and eigenvectors $u_i$ forming a unitary matrix

$$U= \left[\begin{array}{l} u_1^{\top} \\ ... \\ u_d^{\top}\\ \end{array}\right], \quad U^{\top}U=I$$

$$U\Sigma U^{\top} = \lambda, \quad \lambda \text{ diagonal matrix of eigenvalues}$$

If we apply the variable transformation $y=U(x-\mu)$, we can transform the Gaussian PDF to the $y$ coordinates according to the change of variables rule (see preliminaries)

$$p_X(x) = p_Y(y) \underbrace{\left| \frac{\partial y}{\partial x} \right|}_{det|U_{ij}|},$$

which leads to 

$$p(y) = \Pi_{i=1}^d \frac{1}{\sqrt{2 \pi \lambda_i}} \exp \left\{ - \frac{y_i^2}{2\lambda_i} \right\}$$

Note that the diagonalization of $\Delta$ leads to a factorization of the PDF into $d$ 1D PDFs.
#### 1st and 2nd Moment

$$E[x] = \mu$$

$$E[xx^{\top}] = \mu \mu^{\top} + \Sigma$$

$$Cov(x) = \Sigma$$

#### Conditional Gaussian PDF

Consider the case $x \sim \mathcal{N}(x; \mu, \Sigma)$ being a $d$-dimensional Gaussian random vector. We can partition $x$ into two disjoint subsets $x_a$ and $x_b$. 


$$
x=\left(\begin{array}{c}
x_a \\
x_b
\end{array}\right) .
$$

The corresponding partitions of the mean vector $\mu$ and covariance matrix $\Sigma$ become

$$
\mu=\left(\begin{array}{l}
\mu_a \\
\mu_b
\end{array}\right)
$$

$$
\Sigma=\left(\begin{array}{ll}
\Sigma_{a a} & \Sigma_{a b} \\
\Sigma_{b a} & \Sigma_{b b}
\end{array}\right) .
$$

Note that $\Sigma^T=\Sigma$ implies that $\Sigma_{a a}$ and $\Sigma_{b b}$ are symmetric, while $\Sigma_{b a}=\Sigma_{a b}^{\mathrm{T}}$.

We also define the precision matrix $\Lambda = \Sigma^{-1}$, being the inverse of the covariance matrix, where

$$
\Lambda=\left(\begin{array}{ll}
\Lambda_{a a} & \Lambda_{a b} \\
\Lambda_{b a} & \Lambda_{b b}
\end{array}\right)
$$

and $\Lambda_{a a}$ and $\Lambda_{b b}$ are symmetric, while $\Lambda_{a b}^{\mathrm{T}}=\Lambda_{b a}$. Note, $\Lambda_{a a}$ is not simply the inverse of $\Sigma_{a a}$.

Now, we want to evaluate $\mathcal{N}(x_a| x_b; \mu, \Sigma)$ and use $p(x_a, x_b) = p(x_a|x_b)p(x_b)$. We expand all terms of the pdf given the split, and consider all terms that do not involve $x_a$ as constant and then we compare with the generic form of a Gaussian for $p(x_a| x_b)$. We can decompose the equation into quadratic, linear and constant terms in $x_a$ and find an expression for $p(x_a|x_b)$.

> For all intermediate steps refer to [Bishop, 2006](https://link.springer.com/book/9780387310732).

$$
\begin{aligned}
\mu_{a \mid b} & =\mu_a+\Sigma_{a b} \Sigma_{b b}^{-1}\left(x_b-\mu_b\right) \\
\Sigma_{a \mid b} & =\Sigma_{a a}-\Sigma_{a b} \Sigma_{b b}^{-1} \Sigma_{b a}
\end{aligned}
$$

$$p(x_a| x_b) = \mathcal{N(x; \mu_{a|b}, \Sigma_{a|b})}$$

#### Marginal Gaussian PDF

For the marginal PDF we integrate out the dependence on $x_b$ of the joint PDF:

$$p(x_a) = \int p(x_a, x_b) dx_b.$$

We can follow similar steps as above for separating terms that involve $x_a$ and $x_b$. After integrating out the Guassian with a quadratic term depending on $x_b$ we are left with a lengthy term involving $x_a$ only. By comparison with a Gaussian PDF and re-using the block relation between $\Lambda$ and $\Sigma$ as above we obtain for the marginal

$$p(x_a) = \mathcal{N}(x_a; \mu_a, \Sigma_{a a}).$$

#### Bayes Theorem for Gaussian Variables

Generative learning addresses the problem of finding a posterior PDF from a likelihood and prior. The basis is Bayes rule for conditional probabilities 

$$p(x|y) = \frac{p(y|x)p(x)}{p(y)}$$

We want to find the posterior $p(x|y)$ and the evidence $p(y)$ under the assumption that the likelihood $p(y|x)$ and the prior are **linear Gaussian models**.

- $p(y|x)$ is Gaussian and has a mean that depends at most linearly on $x$ and a variance that is independent of $x$.
- $p(x)$ is Gaussian.

These requirements correspond to the following structure of $p(x)$ and $p(y|x)$

$$
\begin{aligned}
p(x) & =\mathcal{N}\left(x \mid \mu, \Lambda^{-1}\right) \\
p(x \mid x) & =\mathcal{N}\left(x \mid A x+b, L^{-1}\right).
\end{aligned}
$$

From that we can derive an analytical evidence (marginal) and posterior (conditional) distributions (for more details see [Bishop, 2006](https://link.springer.com/book/9780387310732)):

$$
\begin{aligned}
p(x) & =\mathcal{N}\left(x \mid A \mu+b, L^{-1}+A \Lambda^{-1} A^{\top}\right) \\
p(x \mid x) & =\mathcal{N}\left(x \mid \Sigma\left\{A^{\top} L(x-b)+\Lambda \mu\right\}, \Sigma\right),
\end{aligned}
$$

where

$$
\Sigma=\left(\Lambda+A^{\top} L A\right)^{-1} .
$$

#### Maximum Likelihood for Gaussians

In generative learning, we need to infer PDFs from data. Given a dataset $X=(x_1, ..., x_N)$, where $x_i$ are i.i.d. random variables drawn from a multivariate Gaussian, we can estimate $\mu$ and $\Sigma$ from the maximum likelihood (ML) (for more details see [Bishop, 2006](https://link.springer.com/book/9780387310732)):

$$\mu_{ML} = \frac{1}{N} \sum_{n=1}^N x_n$$

$$\Sigma_{ML} = \frac{1}{N} \sum_{n=1}^N (x-\mu)^{\top}(x-\mu)$$


$\mu_{ML}$ and $\Sigma_{ML}$ correspond to the so-called sample or empirical estimates. However, $\Sigma_{ML}$ does not deliver an unbiased estimate of the covariance. The difference decreases with $N \to \infty$, but for a certain $N$ we have $\Sigma_{ML}\neq \Sigma$. The practical reason used in the above derivation is that the $\mu_{ML}$ estimate may occur as $\bar{x}$ within the sampling of $\Sigma$ $\Rightarrow$ miscount by one. An unbiased sample variance can be defined as

$$\widetilde{\Sigma} = \frac{1}{N-1} \sum_{n=1}^N (x_n-\mu_{ML})^{\top}(x_n-\mu_{ML})$$


## Further References

**Support Vector Machines**

- [CS229 Lecture Notes](https://cs229.stanford.edu/lectures-spring2022/main_notes.pdf), Chapters 5 and 6; Andrew Ng; 2022
- [Probabilistic Machine Learning: An Introduction](https://probml.github.io/pml-book/book1.html), Section 17.3; Kevin Murphy; 2022

**Gaussian Processes**

- [Pattern Recognition and Machine Learning](https://link.springer.com/book/9780387310732), Section 2.3; Christopher Bishop; 2006