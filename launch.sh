#!/bin/bash

KIND=$1  # 'emcee' or 'pymc'
METHOD=$2  # 'linear' or 'logistic'
DATA=$3  # 'rkhs', 'l2', 'gbm', 'mixture', 'real'
KERNEL=$4  # 'bm', 'fbm', 'ou', 'sqexp', 'gbm', or 'homo/heteroscedastic'
SEED=$5
SMOOTHING=${6-"none"}  # 'none', 'nw' (Nadaraya-Watson) or 'basis'
MOVES=${7-"sw"}  # 'sw', 'de' or 's'

python -Wignore results_cv.py \
  ${KIND} \
  ${METHOD} \
	${DATA} \
	--kernel ${KERNEL} \   # For real data use --data-name [NAME]
	--p-range 1 10 \
	--seed ${SEED} \
	--n-cores 4 \
	--n-reps 10 \
	--n-folds 5 \
	--n-samples 250 \
	--n-grid 100 \
	--smoothing ${SMOOTHING} \
	--train-size 0.6 \
	--n-walkers 64 \
	--n-iters 900 \
	--n-tune 100 \
  --n-burn 400 \
  --n-reps-mle 4 \
	--eta-range -4 2 \
	--g 5 \
	--frac-random 0.3 \
	--moves ${MOVES} \
	--step metropolis \
	--target-accept 0.8
