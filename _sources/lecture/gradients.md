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
