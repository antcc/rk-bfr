#!/bin/bash

METHOD=$1
KERNEL=$2
SEED=$3
SMOOTHING=${5-"nw"}
MOVES=${4-"sw"}

python -Wignore cv_lin.py \
	${METHOD} \
	rkhs \
	--kernel ${KERNEL} \
	--p-range 2 3 \
	--seed ${SEED} \
	--n-cores 4 \
	--n-reps 2 \
	--n-folds 2 \
	--n-samples 200 \
	--n-grid 100 \
	--smoothing ${SMOOTHING}\
	--train-size 0.7 \
	--n-walkers 64 \
	--n-iters 1000 \
	--n-tune 100 \
	--eta-range -2 -1 \
	--g 5 \
	--frac-random 0.3 \
	--moves ${MOVES} \
	--step metropolis \
	--target-accept 0.8
