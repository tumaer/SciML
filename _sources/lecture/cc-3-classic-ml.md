# Core Content 3: Classic ML

## Support Vector Machines

### Linearly Separable Sets

First, we need some mathematical formalism to solve constrained optimization problems. If we define our optimization problem as:

$$\underset{\omega}{\min} f(\omega) \quad s.t. \hspace{2pt} h_{i}(\omega)=0, \hspace{2pt} i=1, \ldots, l$$

Where the $\underset{\omega}{\min}$ seek to find the minimum subject to the constraint(s) $h_{i}(\omega)$. To solve this we have to define a Lagrangian to take into account the constraints.

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

where the inequality can under certain conditions turn into an equality. The conditions for this to turn into an equality are the following (going back to the Lagrangian form from above):

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

If $g(\theta^{\top} x)$ is then close to one, then we have a large confidence that $x$ belongs to class $\mathcal{C}_{1}$ with $y=1$, whereas if it is close to $\frac{1}{2}$, we have much less confidence:

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

where $\alpha_{i}$ is only non-zero for support vectors, and calls the inner product $\langle x^{(i)}, x \rangle$ which is hence a highly efficient computation. As such we have derived the support vector machine for the linear classification of sets. The formulation of the optimum set-boundary/decision boundary was formulated as the search for a margin optimization, then transformed into a convex constrained optimization problem, before restricting the contributions of the computation to contributions coming from the *support vectors*, i.e., vectors on the actual decision boundary estimate hence leading to a strong reduction of the problem dimensionality.

### Non-Separable Sets

<div style="text-align:center">
    <img src="https://i.imgur.com/PMZV7it.png" alt="drawing" width="500"/>
</div>

*Data may not be exactly linealy seperable or some data outliers may undesirably deform the exact decision boundary.*

**Outlier problem**

<div style="text-align:center">
    <img src="https://i.imgur.com/8s61hRD.png" alt="drawing" width="500"/>
</div>

**Original SVM optimisation problem**

$$
\min _{\omega, b} \frac{1}{2}\|\omega\|^{2}  \quad \text {s.t. } y^{(i)}\left(\omega^{\top} x^{(i)}+b\right) \ge 1, i=1, \ldots, m
$$

To make the algorithm work for non-linearly separable data, we introduce $l_1$-regularization, i.e. a penalty-term proportional to the magnitude of a certain quantity

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
&\alpha_{i}^{*} \xi_{i}\left(\omega^{*}\right)=0 \Leftrightarrow \begin{cases}\alpha_{i}^{*}>0, & y^{(i)}\left(\omega^{\top} x^{(i)}+b\right)=0 \\ \alpha_{i}^{*}=0, & y^{(i)}\left(\omega^{\top} x^{(i)}+b\right)<0\end{cases}\\
&\mu_{i}^{*} \xi_{i}=0 \Leftrightarrow \begin{cases}\mu_{i}^{*}>0, & \xi_{i}=0 \\ \mu_{i}^{*}=0, & \xi_{i}>0\end{cases}
\end{aligned}
$$

The resulting dual complementarity conditions for determining the support vectors become: 

- Support vectors on slack margin

$$
\alpha_{i}^{*}=0 \Rightarrow y^{(i)}\left(\omega^{\top} x^{(i)}+b\right)=1-\xi_{i}
$$

- Support vectors inside or outside margin

$$\begin{aligned}
C=\alpha_{i}^{*} & \xRightarrow{(\star)} \mu_{i}^{*}=0,\left\{\begin{array}{rr}
0 \leq \xi_{i} \leq 1 & \text{(correctly classified)} \\ 
 \xi_{i}>1 & \text{(misclassified)}\end{array}\right.\\
& \Rightarrow \quad y^{(i)}\left(\omega^{\top} x^{(i)}+b\right) \leq 1
\end{aligned}$$

- Support vectors on margin

$$\begin{aligned}
0 < \alpha_{i}^{*}<C & \xRightarrow{(\star)} \mu_{i}^{*}>0 \quad \Rightarrow \xi_{i}=0 \\
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

> Recall : only data with $\alpha_{i}^*\ne 0$ , i.e. support vectors, will contribute to the SVM prediction (last eq. of Linearly Separable Sets)

As conclusion, we illustrate the functionality of the slack variables.

<div style="text-align:center">
    <img src="https://i.imgur.com/luacvXq.png" alt="drawing" width="500"/>
</div>

### Sequential Minimal Optimization (SMO)

- Efficient algorithm for solving the SVM dual problem
- Based on *Coordination Ascent* algorithm.

#### Coordinate Ascent

- Task : find $\max _{x} f\left(x_{1}, \ldots, x_{m}\right)$
- Perform a component-wise search on $x$

_Algorithm_

**do until converged**
    &emsp;**for** $i=1, \ldots, m$
        $\qquad x_{i}^{(k+1)}=\underset{\tilde{x}_{i}}{\operatorname{argmax} } f\left(x_{1}^{(k)}, \ldots, \tilde{x}_{i}, \ldots, x_{m}^{(k)}\right) $
    &emsp;**end for**
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

Consider now an iterative update for finding the optimum : 
- Iteration step delivers constraint satisfying set of $\alpha_{i}$.
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

## Kernel Methods

## Gaussian Processes
