import os
import time

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import torch

from test_functions import FUNCS
from algorithm import optimize
from utils import parse_args, load_config, save, calc_results, summarize, setup_logger

import warnings
warnings.filterwarnings("ignore")

torch.set_default_dtype(torch.float64)

def main():
    cfg_parser = parse_args()
    cfg = load_config(cfg_parser)

    test_function = FUNCS[cfg.func.name](cfg.func.dim)
    solutions_set_size = cfg.func.dim * cfg.training.sol_per_dim
    exp_name = f"{cfg.func.name}-{cfg.func.dim}-{cfg.training.restart_method}-{cfg.training.optimizer_type}-{solutions_set_size}"
    path = f"saved/{exp_name.replace('-', '/')}/{cfg.seed}"

    if not os.path.exists(path):
        os.makedirs(path)

    # Set logger
    logger = setup_logger(path)
    print(cfg)
    logger.info(cfg)

    # Set the seed for reproducibility
    torch.random.manual_seed(cfg.seed)
    print("Random Seed:", cfg.seed)
    logger.info(f"Random Seed:{cfg.seed}")

    # Run the evolutionary algorithm
    final_set = optimize(test_function, logger,
                         max_eval=cfg.training.max_eval,
                         solutions_set_size=solutions_set_size,
                         accuracy=cfg.training.accuracy,
                         restart_method=cfg.training.restart_method,
                         optimizer_type=cfg.training.optimizer_type,
                         learning_rate=cfg.training.learning_rate,
                         std=cfg.training.std,
                         verbose=cfg.verbose)

    # Calculate results
    results = calc_results(path)

    # Summarize the evolutionary process
    summarize(test_function, results, final_set, path)

    # Save the last history to continue later if needed and results
    save(results, path, "results.pkl")


if __name__ == '__main__':
    global_time = time.time()
    main()
    print(f'Done: {(time.time() - global_time) / 60:.2f} min')
