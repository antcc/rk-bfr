This package was developed as part of my [master's thesis](https://github.com/antcc/tfm), which has since been rewritten and expanded as [an article](https://github.com/antcc/rk-bfr-preprint).

------------

# rk-bfr

A Bayesian framework for functional linear and logistic regression models, built on the theory of RKHS's. An overview of the models is available on Chapter 3 [here](https://github.com/antcc/tfm/releases/download/v1.2/masters-thesis.pdf) or Section 2 [here](https://arxiv.org/pdf/2312.14086)

## Code structure

- The folder `rkbfr` contains the inference and prediction pipeline implemented, using the [emcee](https://emcee.readthedocs.io/) MCMC sampler and following the style of the [scikit-learn](https://scikit-learn.org/) and [scikit-fda](https://fda.readthedocs.io/) libraries.
- The folder `reference_methods` contains the implementation of some functional algorithms used for comparison.
- The folder `utils` contains several utility files for experimentation and visualization.
- The `experiments` folder contains plain text files with numerical experimental results, as well as `.csv` and `.npz` files that facilitate working with them directly in Python.

## Usage

There are some experiments (with both simulated and real data) available to test the performance of the models against other usual alternatives, functional or otherwise. The script `results_cv.py` runs the experiments with a cross-validation loop for our Bayesian models, while the script `results_all.py` runs the experiments for all hyperparameters without a cross-validation loop. 

A typical execution can be seen in the `launch.sh` file, and additionally there are two Jupyter notebooks that demonstrate the usage of the code.

*Code developed for Python 3.9.*
