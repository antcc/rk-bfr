#!/bin/bash
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=antonio.coin@estudiante.uam.es
#SBATCH --output=slurm/%A_%a.out
#SBATCH --array=1-6

METHOD=$1
SEED=$2
MOVES=${3-"sw"}

case ${SLURM_ARRAY_TASK_ID} in
  1)
    KERNEL="fbm"
    SMOOTHING="none"
  ;;
  2)
    KERNEL="fbm"
    SMOOTHING="nw"
  ;;
  3)
    KERNEL="ou"
    SMOOTHING="none"
  ;;
  4)
    KERNEL="ou"
    SMOOTHING="nw"
  ;;
  5)
    KERNEL="sqexp"
    SMOOTHING="none"
  ;;
  6)
    KERNEL="sqexp"
    SMOOTHING="nw"
  ;;
esac

srun python -Wignore cv_lin.py \
	${METHOD} \
	rkhs \
	--kernel ${KERNEL} \
	--p-range 2 3 \
	--seed ${SEED} \
	--n-cores ${SLURM_CPUS_PER_TASK} \
	--n-reps 1 \
	--n-folds 2 \
	--n-samples 200 \
	--n-grid 100 \
	--smoothing ${SMOOTHING}\
	--train-size 0.7 \
	--n-walkers 64 \
	--n-iters 1000 \
	--n-tune 100 \
	--eta-range -3 -3 \
	--g 5 \
	--frac-random 0.3 \
	--moves ${MOVES} \
	--step metropolis \
	--target-accept 0.8
