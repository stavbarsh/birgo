import os
import pickle
import argparse
import json
from types import SimpleNamespace
import logging
import torch
from torch.optim import Optimizer
import numpy as np
import scipy.optimize


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Config file that will be used')
    parser.add_argument('--force_fn', type=str, help='Override test function', default=None)
    parser.add_argument('--force_seed', type=int, help='Override seed', default=None)
    parser.add_argument('--force_max_eval', type=float, help='Override the budget of function evaluations', default=None)
    parser.add_argument('--force_dim', type=int, help='Override function dimension', default=None)
    parser.add_argument('--force_rest', type=str, help='Override restart method', default=None)
    parser.add_argument('--force_sol', type=int, help='Override solutions set size per dimension', default=None)
    parser.add_argument('--force_lr', type=float, help='Override lr', default=None)
    parser.add_argument('--force_opt', type=str, help='Override optimizer', default=None)
    return parser.parse_args()


def load_config(cfg_parser):
    """:returns: a SimpleNamespace from a json file"""
    with open(cfg_parser.config) as file:
        cfg = json.load(file, object_hook=lambda d: SimpleNamespace(**d))
    cfg.seed = cfg_parser.force_seed if cfg_parser.force_seed is not None else cfg.seed
    cfg.func.name = cfg_parser.force_fn if cfg_parser.force_fn is not None else cfg.func.name
    cfg.training.max_eval = int(cfg_parser.force_max_eval) if cfg_parser.force_max_eval is not None else int(cfg.training.max_eval)
    cfg.func.dim = cfg_parser.force_dim if cfg_parser.force_dim is not None else cfg.func.dim
    cfg.training.sol_per_dim = int(cfg_parser.force_sol) if cfg_parser.force_sol is not None else int(
        cfg.training.sol_per_dim)
    cfg.training.restart_method = cfg_parser.force_rest if cfg_parser.force_rest is not None else cfg.training.restart_method
    cfg.training.learning_rate = cfg_parser.force_lr if cfg_parser.force_lr is not None else cfg.training.learning_rate
    cfg.training.optimizer_type = cfg_parser.force_opt if cfg_parser.force_opt is not None else cfg.training.optimizer_type
    return cfg


def save(variable, folder: str, file: str):
    if not os.path.exists(folder):
        os.makedirs(folder)
    with open(os.path.join(folder, file), 'wb') as f:
        pickle.dump(variable, f)


def load(filename: str):
    with open(filename, 'rb') as f:
        var = pickle.load(f)
    return var


def calc_results(path):
    # Lists to store the tracked values across iterations
    results = []
    log_path = path + '/' + 'results.log'
    with open(log_path, 'r') as file:
        for line in file:
            try:
                evals, selection_size, best_fitness, best_gradient_norm, avg_fitness, avg_gradient_norm, time = line.split(
                    ',')

                # Save the results
                results.append({
                    "evals": int(evals),
                    "selection_size": int(selection_size),
                    "mean_fitness": float(avg_fitness),
                    "best_fitness": float(best_fitness),
                    "mean_gradient_norm": float(avg_gradient_norm),
                    "best_gradient_norm": float(best_gradient_norm),
                    "time": float(time)
                })
            except ValueError:
                continue
    return results


def summarize(test_function, results: list, final_set: list, path: str):
    """
    Print a summary of the results from the evolution.
    """
    # Retrieve final iteration details from history
    final_iteration = results[-1]
    best_solution = final_set[0]['solution']

    # Print summary header
    print("\nOptimization Summary")
    print("=" * 50)

    print(f"Number of iterations: {len(results)}")
    print(f"Number of function evaluations: {final_iteration['evals']}")

    n_grads = 0
    for r in results:
        n_grads += r['selection_size']
    print(f"Number of gradient steps: {n_grads}")
    print(f"In comparison to maximum gradient steps: {len(final_set) * len(results)}")

    print(f"Best value: {final_iteration['best_fitness']:.4f} vs Optimal value: {test_function.fglob}")
    print(
        f"Best point: {[round(p, 4) for p in best_solution.tolist()]} vs Optimal Points: {test_function.global_optimum}")

    # Print the average fitness of the final solutions set
    print(f"Average fitness of the final set: {final_iteration['mean_fitness']:.4f}")

    # Print the average gradient norm of the final solutions set
    print(f"Average gradient norm of the final solutions set: {final_iteration['mean_gradient_norm']:.4f}")

    summary = {
        "iterations": len(results),
        "evals": final_iteration['evals'],
        "num_grads": n_grads,
        "best_point": best_solution.tolist(),
        "mean_fitness": final_iteration['mean_fitness'],
        "best_fitness": final_iteration['best_fitness'],
        "mean_gradient_norm": final_iteration['mean_gradient_norm'],
        "best_gradient_norm": final_iteration['best_gradient_norm']
    }

    # Write the summary to the JSON file
    with open(f"{path}/log.json", 'w') as file:
        json.dump(summary, file, indent=4)


def setup_logger(path, name='results.log', level=logging.INFO):
    """
    Function to set up a custom logger with a specified filename and level.
    """
    # Create a custom logger
    logger = logging.getLogger('logger')
    logger.setLevel(level)

    # Create handlers
    file_handler = logging.FileHandler(path + '/' + name, mode='w')

    # Create formatters and add it to handlers
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)

    return logger


class LBFGSB(Optimizer):
    """
    Wrapper around scipy.optimize.minimize(method='L-BFGS-B').

    Line Search: Uses the implementation from the original Fortran code
    (More & Thuente), which satisfies the Strong Wolfe conditions.

    max_iter bounds the line search iterations of a single step, while max_eval
    bounds its function evaluations and is left open when it is not given.
    eval_count reports what the last step spent, so that the caller can charge it
    to the total budget.
    """

    def __init__(self, params, lr=1, max_iter=20, max_eval=None, bounds=None):
        # lr is ignored by SciPy but kept for API compatibility
        super().__init__(params, dict(lr=lr, max_iter=max_iter, max_eval=max_eval, bounds=bounds))
        self.max_iter = max_iter
        self.max_eval = max_eval
        self.bounds = bounds
        self.eval_count = 0

    def step(self, closure=None):
        """The closure is the objective itself, SciPy decides which points to evaluate."""
        if closure is None:
            return None

        p = self.param_groups[0]['params'][0]
        device, dtype = p.device, p.dtype
        self.eval_count = 0

        def objective(x):
            # Called for every point SciPy tries, the inner points of the line search
            # included, so all of them are counted in the total evaluations
            self.eval_count += 1

            # Load Numpy -> Tensor
            with torch.no_grad():
                p.copy_(torch.from_numpy(x).to(device).to(dtype))

            # Forward + Backward
            self.zero_grad()
            with torch.enable_grad():
                loss = closure(p)
                loss.backward()

            # Return loss (float) and gradients (float64)
            return loss.item(), p.grad.cpu().numpy().astype(np.float64)

        options = {'maxiter': self.max_iter, 'ftol': 1e-6, 'gtol': 1e-6}
        if self.max_eval is not None:
            options['maxfun'] = self.max_eval

        res = scipy.optimize.minimize(
            fun=objective,
            x0=p.detach().cpu().numpy().astype(np.float64).flatten(),
            method='L-BFGS-B',
            jac=True,
            bounds=self.bounds,
            options=options
        )

        # Load result back to Tensor
        with torch.no_grad():
            p.copy_(torch.from_numpy(res.x).to(device).to(dtype))

        return res.fun
