This package was developed as part of my [master's thesis](https://github.com/antcc/tfm).

------------

# rk-bfr

A Bayesian framework for functional linear and logistic regression models, built on the theory of RKHS's. An overview of the model is available on Chapter 3 [here](https://github.com/antcc/tfm/releases/download/v1.0/masters-thesis.pdf).

## Code structure

- The folder `rkbfr` contains the inference and prediction pipeline implemented, using the [emcee](https://emcee.readthedocs.io/) MCMC sampler and following the style of the [scikit-learn](https://scikit-learn.org/) and [scikit-fda](https://fda.readthedocs.io/) libraries.
- The folder `reference_methods` contains the implementation of some functional algorithms used for comparison.
- The folder `utils` contains several utility files for experimentation and visualization.

## Usage

There are some experiments (with both simulated and real data) available to test the performance of the models against other usual alternatives, functional or otherwise. The script `results_cv.py` runs the experiments with a cross-validation loop for our Bayesian models, while the script `results_all.py` runs the experiments for all hyperparameters without a cross-validation loop. A typical execution can be seen in the `launch.sh` file, and there are two Jupyter notebooks that demonstrate the usage of the code.

*Code developed for Python 3.9.*
