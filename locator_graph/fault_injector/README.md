# Fault Injector

Based on DeepCrime we develop Fault Injector to systematically explore bug-space of Sequence-Based Models.

We modify originally collected models to work with Fault Injector by adding placeholder variables for each operator.

Models are trained multiple times to check the statistical significance of the results and metrics.

For classification based models we use accuracy for thresholding while for others we use loss.