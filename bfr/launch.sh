#!/bin/bash

python -Wignore cv_lin.py \
	emcee \
	rkhs \
	--kernel fbm \
	--p-prior 0.1 0.5 0.4 \
	--seed 2022 \
	--n-cores 4 \
	--n-reps 1 \
	--n-folds 2 \
	--n-samples 50 \
	--n-grid 50 \
	--smoothing \
	--train-size 0.7 \
	--n-walkers 20 \
	--n-iter 100 \
	--eta-range -1 -1 \
	--g 5 \
	--frac-random 0.3 \
	--moves de \
	--step metropolis \
	--target-accept 0.8
