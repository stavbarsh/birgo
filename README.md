# Bi-criteria Restart Heuristic for Gradient-based Global Optimization

This repository contains the code and experiments accompanying the paper titled **"Introducing a Bi-criteria Restart Heuristic for Gradient-based Global Optimization"** by Stav Bar-Sheshet, Adham Salih, and Amiram Moshaiov. The paper introduces a novel bi-criteria restart heuristic for gradient-based optimization methods in global optimization settings.

## Overview
Gradient-based optimization methods often converge to local optima, especially in multimodal functions. Multi-start strategies help explore the search space better, but their performance depends on effective restart heuristics. This paper proposes a novel bi-criteria heuristic that considers both the function value and the gradient norm to decide which solutions to restart. Solutions are compared based on Pareto dominance in the bi-criteria space, balancing exploration and exploitation.

## Key Features
- Novel bi-criteria restart heuristic using both function values and gradient norms.
- Adaptive domination-based restart decision.
- Implemented algorithm called **BiRGO (Bi-criteria Restart for Global Optimization)**.
- Evaluation against state-of-the-art gradient-based multi-start algorithms.
- Performance evaluation using a set of differentiable multimodal and unimodal test functions.

## Requirements
- Python 3.x
- PyTorch
- SciPy
- NumPy

## Usage
To run the experiments with the BiRGO algorithm or other compared methods, execute:
```bash
bash loop.sh
```
### Example Parameters:
- `--force_fn`: Objective function (e.g., Ackley, Rastrigin, Rosenbrock, etc.)
- `--force_seed`: Random seed for reproducibility
- `--force_opt`: Optimizer (e.g., Adam)
- `--force_lr`: Learning rate
- `--force_rest`: Restart method (e.g., init, none, pgd, egd, random)
- `--force_dim`: Dimensionality of the optimization problem
- `--force_sol`: Solution set size per dimension
- `--force_max_eval`: Budget of objective function evaluations

All the compared algorithms are stopped by the same budget of objective function
evaluations, which is 100,000 x d in the reported experiments. For the local search
methods this includes the points evaluated inside the L-BFGS-B line search.

Basic-MS and BiRGO-II bound the line search of a single gradient step by the
solutions set size. Threshold-MS is bounded solely by the remaining budget, so a
local search runs until the L-BFGS-B thresholds stop it. Being sequential, an early
search can take a large part of the budget and leave less for the ones after it,
and over a whole run it may perform more gradient steps than the parallel methods
for the same number of evaluations.

## Experiments
The experiments described in the paper can be reproduced using the provided configuration files in the `configs/` directory. The test functions include both multimodal and unimodal functions such as:
- Ackley
- BentCigar
- DixonPrice
- DropWave
- Griewank
- Rastrigin
- Rosenbrock
- Salomon
- Schwefel
- StyblinskiTang
- Trid

The compared algorithms include:
- **BiRGO (Proposed)**
- **MS-Basic** (Simple Multi-Start without restart)
- **MS-Threshold** (Sequential Multi-Start with function value and gradient thresholds)
- **MS-PGD** (Perturbed Gradient Descent)
- **EGD** (Evolutionary Gradient Descent)
- **Random Search**

## Results
The results demonstrate that BiRGO consistently outperforms the benchmark algorithms, especially on multimodal functions. Statistical significance was verified using the Mann-Whitney U test.

## Citation
If you use this code in your research, please cite our paper:
```
@article{barsheshet2026birgo,
  title={Introducing a Bi-criteria Restart Heuristic for Gradient-based Global Optimization},
  author={Stav Bar-Sheshet and Adham Salih and Amiram Moshaiov},
  journal={In Review},
  year={2026}
}
```

## License
This project is licensed under the MIT License.

## Contact
For questions, comments, or collaborations, please contact:
- Stav Bar-Sheshet: stavb1@mail.tau.ac.il
- Adham Salih: adhamsalih@braude.ac.il
- Amiram Moshaiov: moshaiov@tauex.tau.ac.il

Repository: [GitHub - BiRGO](https://github.com/stavbarsh/birgo)