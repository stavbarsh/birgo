#!/bin/bash

# Fixed budget of 100,000 x d function evaluations, identical for every algorithm
evals_per_dim=100000

for seed in 3407 1 696 52 789 152 817 3028 2691 3780 5491 8754 16591 4246 23467 82954 98505 251807 928260 240666 752050 354216 287017 6801059 9866364 6148455 6865903 51160425 42006642 23301468 50545420
do
  for func in Ackley BentCigar DixonPrice DropWave Griewank Rastrigin Rosenbrock Salomon Schwefel StyblinskiTang Trid
	do
    for dim in 2 50 100
    do
      # |S| = d and |S| = 10d
      for dim_sol in 1 10
      do
        # BiRGO-I, Basic-MS, Multi-PGD, EGD and Random Search, all with Adam
        for res_met in init none pgd egd random
        do
          python main.py configs/exp.json --force_fn=$func --force_seed=$seed --force_opt=adam --force_lr=0.01 --force_rest=$res_met --force_dim=$dim --force_sol=$dim_sol --force_max_eval=$((evals_per_dim * dim))
        done

        # BiRGO-II, Basic-MS and Threshold-MS, all with L-BFGS-B, which ignores the lr.
        # Threshold-MS searches one solution at a time, so both |S| give the same run
        for res_met in init none ms_thresh
        do
          python main.py configs/exp.json --force_fn=$func --force_seed=$seed --force_opt=lbfgsb --force_rest=$res_met --force_dim=$dim --force_sol=$dim_sol --force_max_eval=$((evals_per_dim * dim))
        done
      done
    done
  done
done
