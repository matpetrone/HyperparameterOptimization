# Bayesian Hyperparameter Optimization
A Project with the purpose to optimize hyperparameters in a Neural Network using Bayesian optimization

## Getting Started

This program needs to install ["pytorch"](https://pytorch.org/get-started/locally/), ["BayesianOptimization"](https://github.com/fmfn/BayesianOptimization) and ["tensorboardX"](https://pypi.org/project/tensorboard/).
Then is sufficient to download the project and run "evaluation.py"

_Please note_: When you run "evaluation.py" the program starts downloading the [STL 10](https://cs.stanford.edu/~acoates/stl10/) dataset.

## How read the output
When the program ends, the output represents a table indicating each value of the hyperparameters (learning rate and weight decay) for each iteration and 4 vectors:
- vector of min train loss for every iteration
- vector of max train accuracy for every iteration
- vector of min validation loss for every iteration
- vector of max validation accuracy for every iteration

