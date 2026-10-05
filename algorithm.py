import time
import torch
from utils import LBFGSB

def initialize_solutions_set(test_function, solutions_set_size: int, optimizer_type: str = "adam",
                          learning_rate: float = 0.01, max_iter: int = 20) -> list:
    """
    Initialize a random set of candidate solutions for a given test function.
    Each solution is represented as a tensor.
    Along with each solution, initialize an optimizer.
    """
    solutions_set = []
    dim = test_function.N
    bounds = test_function.bounds
    for _ in range(solutions_set_size):
        values = []
        for i in range(dim):
            # Uniformly sample every dimension across its entire range
            val = (bounds[i][1] - bounds[i][0]) * torch.rand(1).item() + bounds[i][0]
            values.append(val)
        solution = torch.tensor(values, requires_grad=True)
        optimizer = select_optim(optimizer_type, solution, learning_rate, bounds, max_iter)
        solutions_set.append({'solution': solution, 'optimizer': optimizer})
    return solutions_set


def evaluate_solutions_set(test_function, solutions_set: list) -> tuple[list, list]:
    """
    Evaluate the fitness and the gradient L2 norm of each solution in the solutions set.
    Whatever the optimizer is, this costs one evaluation per solution.
    """
    fitness_values = []
    gradient_norms = []

    for entry in solutions_set:
        solution = entry['solution']

        # Calculate the fitness of the solution and its gradient
        entry['optimizer'].zero_grad()
        fitness_value = test_function.fun(solution)
        fitness_value.backward()

        fitness_values.append(fitness_value.item())

        # Calculate the L2 norm of the gradient
        gradient_norms.append(torch.norm(solution.grad).item())

    return fitness_values, gradient_norms


def gradient_step(entry, test_function) -> int:
    """
    Advance a single solution with its own optimizer and clip it back into the bounds.
    L-BFGS-B gets the objective itself since it chooses the points of its line search.
    Returns the number of evaluations the step used, which is zero for Adam and SGD
    since they reuse the gradient of the last evaluation.
    """
    optimizer = entry['optimizer']
    if isinstance(optimizer, LBFGSB):
        optimizer.step(test_function.fun)
        evals = optimizer.eval_count
    else:
        optimizer.step()
        evals = 0

    with torch.no_grad():
        entry['solution'] = clip_bounds(entry['solution'], test_function.bounds)

    return evals


def dominates(i: int, j: int, fitness_values: list, gradient_norms: list) -> bool:
    """
    Check whether i dominates j
    """
    at_least_as_good = ((fitness_values[i] <= fitness_values[j]) and (gradient_norms[i] >= gradient_norms[j]))
    one_better = ((fitness_values[i] < fitness_values[j]) or (gradient_norms[i] > gradient_norms[j]))

    return (i != j) and at_least_as_good and one_better


def nondominated_front_indices(fitness_values: list, gradient_norms: list) -> list:
    """
    Compute the non-dominated front for the given fitness values and gradient norms using pure Python.
    """
    solutions_set_size = len(fitness_values)
    nondominated_indices = []

    for i in range(solutions_set_size):
        dominated = False
        for j in range(solutions_set_size):
            # min - max  problem
            if dominates(j, i, fitness_values, gradient_norms):
                dominated = True
                break
        if not dominated:
            nondominated_indices.append(i)

    # sort over fitness since eventually it is a single objective optimization
    nondominated_indices.sort(key=lambda idx: fitness_values[idx])

    return nondominated_indices


def clip_bounds(solution, bounds):
    """
    Clip a solution back into the search bounds, in place so that the optimizer
    keeps pointing at the same tensor.
    """
    lower = solution.new_tensor([b[0] for b in bounds])
    upper = solution.new_tensor([b[1] for b in bounds])
    return solution.clip_(lower, upper)


def gaussian_mutation(solution, bounds, mutation_prob=0.1, std=1):

    # Mutation mask: where mutations occur
    mutation_mask = torch.rand_like(solution) < mutation_prob

    # Normal random numbers for mutation calculations
    rand = torch.randn_like(solution) * std

    mutated_solution = solution + mutation_mask * rand
    mutated_solution = clip_bounds(mutated_solution, bounds)
    return mutated_solution


def select_optim(optimizer_type, params, learning_rate, bounds=None, max_iter=20):
    if optimizer_type == 'adam':
        return torch.optim.Adam([params], lr=learning_rate)
    elif optimizer_type == 'lbfgsb':
        # The line search of a single step is limited to the solutions set size
        return LBFGSB([params], lr=learning_rate, max_iter=max_iter, bounds=bounds)
    else:
        return torch.optim.SGD([params], lr=learning_rate)


def optimize(test_function, logger,
             max_eval: int = 20000,
             solutions_set_size: int = 20,
             accuracy: float = 1e-6,
             restart_method: str = "init",
             optimizer_type: str = "adam",
             learning_rate: float = 0.01,
             std: float = 1.0,
             verbose: bool = False) -> list:
    """
    Optimize an objective function according to the bi-criteria heuristic.

    Every method gets the same budget, given in function evaluations (max_eval).
    A single gradient based iteration over a set of n solutions therefore costs n,
    while a local search costs whatever its line search consumes. The evaluations
    are accumulated in evals, and itr counts the gradient steps, which PGD and EGD
    use to space their mutations.
    """

    best_acc = torch.inf
    evals = 0
    itr = 0
    t = time.time()

    # MS-Threshold parameters - sequential local searches, one solution at a time
    solutions_set_size = 1 if restart_method == "ms_thresh" else solutions_set_size
    optimizer_type = 'lbfgsb' if restart_method == "ms_thresh" else optimizer_type
    best_rand_fitness = torch.inf
    best_rand_solution = None

    # PGD, EGD parameters
    L = 100
    update = torch.ones(solutions_set_size, dtype=torch.bool)
    escape = torch.ones(solutions_set_size, dtype=torch.bool)
    radius = torch.linspace(1, 1.5, steps=solutions_set_size) * std if restart_method == "egd" else std
    itr_mutate = torch.zeros(solutions_set_size, dtype=torch.long) if restart_method == "egd" else 0

    logger.info(
        ','.join(
            ['evals', 'selection_size', 'best_fitness', 'best_gradient_norm', 'avg_fitness', 'avg_gradient_norm',
             'time']))

    solutions_set = initialize_solutions_set(test_function, solutions_set_size,
                                       optimizer_type=optimizer_type,
                                       learning_rate=learning_rate,
                                       max_iter=solutions_set_size)

    while best_acc > accuracy and evals < max_eval:

        # Calculate fitness, one evaluation per solution
        fitness_values, gradient_norms = evaluate_solutions_set(test_function, solutions_set)
        evals += len(solutions_set)

        # Perform non-dominated sorting and select solutions
        nondominated_indices = nondominated_front_indices(fitness_values, gradient_norms)
        selection_size = len(nondominated_indices)

        avg_fitness = sum(fitness_values) / len(fitness_values)
        avg_gradient_norm = sum(gradient_norms) / len(gradient_norms)
        best_fitness = fitness_values[nondominated_indices[0]]
        best_gradient_norm = gradient_norms[nondominated_indices[0]]
        best_acc = abs(best_fitness - test_function.fglob)

        # Random Search and MS-Threshold drop their solutions set, so keep the best point
        # while it is still there
        if restart_method in ("random", "ms_thresh") and best_fitness < best_rand_fitness:
            best_rand_fitness = best_fitness
            best_rand_solution = {'solution': solutions_set[nondominated_indices[0]]['solution'].detach().clone()}

        if verbose:
            # Print summary for this iteration
            print(f"\nEvaluations {evals}/{max_eval}")
            print("=" * 30)
            print(f"Number of solutions in non-dominated front: {selection_size}")
            print(f"Average fitness of the solutions_set: {avg_fitness:.4f}")
            print(f"Average gradient norm of the solutions_set: {avg_gradient_norm:.4f}")
            print(f"Loop time: {time.time() - t:.4f} seconds")

        # Log summary for this iteration
        logger.info(','.join(
            str(v) for v in
            [evals, selection_size, best_fitness, best_gradient_norm, avg_fitness,
             avg_gradient_norm, time.time() - t]))

        t = time.time()

        if restart_method == "none":
            # Optimize the whole solutions set using gradient descent - multi-start
            for entry in solutions_set:
                evals += gradient_step(entry, test_function)

        elif restart_method == "random":
            solutions_set = initialize_solutions_set(test_function, solutions_set_size,
                                               optimizer_type=optimizer_type,
                                               learning_rate=learning_rate,
                                               max_iter=solutions_set_size)

        elif restart_method == "ms_thresh":
            # A single local search, bounded only by the remaining budget, then a restart.
            # The searches are sequential, so an early one may take a large part of the
            # budget and leave less for the ones that follow it
            entry = solutions_set[0]
            entry['optimizer'].max_iter = max(max_eval - evals, 1)
            entry['optimizer'].max_eval = max(max_eval - evals, 1)

            evals += gradient_step(entry, test_function)
            new_entry = initialize_solutions_set(test_function, 1,
                                              optimizer_type=optimizer_type,
                                              learning_rate=learning_rate)
            solutions_set = new_entry + [entry]

        elif restart_method == "init":
            # Exploration - Create new solutions using initialization
            new_solutions_set = initialize_solutions_set(test_function, solutions_set_size - selection_size,
                                                         optimizer_type=optimizer_type,
                                                         learning_rate=learning_rate,
                                                         max_iter=solutions_set_size)

            # Exploitation - Perform gradient descent on selected partners using optimizer
            for idx in nondominated_indices:
                evals += gradient_step(solutions_set[idx], test_function)

            # Combine solutions for next iteration
            solutions_set = [solutions_set[idx] for idx in nondominated_indices] + new_solutions_set

        elif restart_method == "pgd":
            # Optimize using PGD algorithm - Perturbed GD
            for p in range(solutions_set_size):
                # Mutation if near saddle point (gradient norm is small) and respecting the i - i_mutate > L condition
                if gradient_norms[p] < 1e-3 and itr - itr_mutate > L:
                    itr_mutate = itr
                    parent = solutions_set[p]['solution'].detach().clone()
                    child = gaussian_mutation(parent, test_function.bounds, mutation_prob=1, std=radius)

                    child.requires_grad_()  # make it a leaf node
                    solutions_set[p]['solution'] = child
                    solutions_set[p]['optimizer'] = select_optim(optimizer_type, child, learning_rate,
                                                                 test_function.bounds, solutions_set_size)

                # Standard Gradient Descent Step
                else:
                    evals += gradient_step(solutions_set[p], test_function)

        elif restart_method == "egd":
            # Optimize using EGD algorithm
            for p in range(solutions_set_size):
                # Mutation if near saddle point (gradient norm is small) and respecting the i - i_mutate > L condition
                if gradient_norms[p] < 1e-3 and itr - itr_mutate[p] > L:
                    update[p] = False

                # Standard Gradient Descent Step
                else:
                    evals += gradient_step(solutions_set[p], test_function)

            if update.sum() == 0:
                # Mutation and selection
                for p in range(solutions_set_size):
                    itr_mutate[p] = itr
                    parent = solutions_set[p]['solution'].detach().clone()
                    child = gaussian_mutation(parent, test_function.bounds, mutation_prob=1, std=radius[p])

                    child.requires_grad_()  # make it a leaf node
                    optimizer = select_optim(optimizer_type, child, learning_rate, test_function.bounds,
                                             solutions_set_size)
                    for _ in range(L):
                        if evals >= max_eval:
                            break

                        # Standard Gradient Descent Step
                        optimizer.zero_grad()
                        fitness_value = test_function.fun(child)
                        evals += 1
                        fitness_value.backward()
                        optimizer.step()
                        with torch.no_grad():
                            child = clip_bounds(child, test_function.bounds)

                    # Check for improvement and update accordingly
                    evals += 2
                    if test_function.fun(child) + 1e-1 < test_function.fun(parent):
                        escape[p] = True
                        solutions_set[p]['solution'] = child
                        solutions_set[p]['optimizer'] = optimizer
                    else:
                        escape[p] = False

                fitnesses = torch.tensor([test_function.fun(entry['solution']) for entry in solutions_set])
                evals += len(solutions_set)
                mean_fitness = torch.mean(fitnesses)
                current_best_idx = torch.argmin(fitnesses)
                best_solution = solutions_set[current_best_idx]

                # The escape phase already took L gradient steps on every solution
                itr += L

                for p in range(solutions_set_size):
                    if escape[p]:
                        continue

                    evals += 1
                    if test_function.fun(solutions_set[p]['solution']) >= mean_fitness:
                        child = best_solution['solution'].detach().clone()
                        child.requires_grad_()  # make it a leaf node
                        solutions_set[p]['solution'] = child
                        solutions_set[p]['optimizer'] = select_optim(optimizer_type, child, learning_rate,
                                                                     test_function.bounds, solutions_set_size)

                update.fill_(True)

        itr += 1

        # These two methods keep nothing between iterations, so report the best point seen
        if restart_method in ("random", "ms_thresh") and evals >= max_eval:
            logger.info(','.join(
                str(v) for v in
                [evals, 0, best_rand_fitness, 0, 0, 0, 0]))
            solutions_set = [best_rand_solution]

    return solutions_set
