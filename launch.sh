#!/bin/bash

KIND=$1
METHOD=$2
DATA=$3
KERNEL=$4
SEED=$5
SMOOTHING=${6-"none"}
MOVES=${7-"sw"}

python -Wignore results_cv.py \
    ${KIND} \
	${METHOD} \
	${DATA} \
	--kernel ${KERNEL} \
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
