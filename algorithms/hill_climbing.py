import numpy as np
import time


def validate_inputs(population_points, weights, candidate_hospitals, lambd):

    population_points = np.array(population_points, dtype=float)
    weights = np.array(weights, dtype=float)
    candidate_hospitals = np.array(candidate_hospitals, dtype=float)

    if population_points.ndim != 2 or population_points.shape[1] != 2:
        raise ValueError("population points must be a 2D array with shape (n, 2).")

    if candidate_hospitals.ndim != 2 or candidate_hospitals.shape[1] != 2:
        raise ValueError("candidate hospitals must be a 2D array with shape (m, 2).")

    if len(weights) != len(population_points):
        raise ValueError("weights length must equal number of population points.")

    if len(candidate_hospitals) == 0:
        raise ValueError("candidate_hospitals cannot be empty.")

    if lambd <= 0:
        raise ValueError("lambda must be positive.")

    if np.any(weights < 0):
        raise ValueError("weights cannot be negative.")

    return population_points, weights, candidate_hospitals


def calculate_cost(population_points, weights, candidate_hospitals, solution, lambd):
    selected_indices = np.where(solution == 1)[0]

    if len(selected_indices) == 0:
        return float("inf")

    selected_hospitals = candidate_hospitals[selected_indices]

    total_distance_cost = 0

    for point, weight in zip(population_points, weights):
        distances = np.sqrt(np.sum((selected_hospitals - point) ** 2, axis=1))
        nearest_distance = np.min(distances)
        total_distance_cost += weight * nearest_distance

    hospital_building_cost = lambd * len(selected_indices)

    return total_distance_cost + hospital_building_cost


def generate_initial_solution(num_candidates, selection_rate=0.15 , seed=None):
    if num_candidates <= 0:
        raise ValueError("num_candidates must be greater than 0.")

    if selection_rate <= 0 or selection_rate > 1:
        raise ValueError("selection_rate must be between 0 and 1.")

    solution = np.zeros(num_candidates, dtype=int)

    num_selected = max(1, int(num_candidates * selection_rate))
    selected_indices = np.random.choice(num_candidates, num_selected, replace=False)

    solution[selected_indices] = 1

    return solution


def generate_successors(solution, max_swaps=2, seed=None):
    if solution is None:
        raise ValueError("solution cannot be None.")

    solution = np.array(solution, dtype=int)

    if solution.ndim != 1:
        raise ValueError("solution must be a 1D binary vector.")

    if not np.all((solution == 0) | (solution == 1)):
        raise ValueError("solution must contain only 0 and 1.")

    successors = []

    for i in range(len(solution)):
        new_solution = solution.copy()
        new_solution[i] = 1 - new_solution[i]
        successors.append(new_solution)

    # We remove one selected hospital and add one unselected hospital.
    # This keeps the number of hospitals the same but changes their locations.
    selected_indices = np.where(solution == 1)[0]
    unselected_indices = np.where(solution == 0)[0]


    if len(selected_indices) > 0 and len(unselected_indices) > 0:
        for _ in range(max_swaps):
            selected = np.random.choice(selected_indices)
            unselected = np.random.choice(unselected_indices)

            new_solution = solution.copy()
            new_solution[selected] = 0
            new_solution[unselected] = 1
            successors.append(new_solution)
            
    return successors

def random_restart_hill_climbing( population_points, weights, candidate_hospitals, lambd, restart_times=10, max_iterations=500, selection_rate=0.15 ,seed=None):
    population_points, weights, candidate_hospitals = validate_inputs( population_points, weights, candidate_hospitals, lambd)
    if seed is not None:
        np.random.seed(seed)
    if restart_times < 0:
        raise ValueError("restart times cannot be negative.")

    if max_iterations <= 0:
        raise ValueError("max iterations must be greater than 0.")
    start = time.time()
    best_solution = None
    best_cost = float("inf")

    total_expanded_states = 0
    total_viewed_states = 0
    cost_history = [] 

    for restart in range(restart_times + 1):

        current_solution = generate_initial_solution(len(candidate_hospitals),selection_rate)
        current_cost = calculate_cost(population_points, weights, candidate_hospitals, current_solution, lambd )

        for iteration in range(max_iterations):

            successors = generate_successors(current_solution)
            total_viewed_states += len(successors)

            best_successor = current_solution.copy()
            best_successor_cost = current_cost

            for successor in successors:
                successor_cost = calculate_cost(population_points,weights,candidate_hospitals,successor,lambd)
                if successor_cost < best_successor_cost:
                    best_successor = successor.copy()
                    best_successor_cost = successor_cost

            if best_successor_cost >= current_cost:
                break

            current_solution = best_successor.copy()
            current_cost = best_successor_cost
            total_expanded_states += 1
            cost_history.append(current_cost) # record cost at each improvement
        if current_cost < best_cost:
            best_solution = current_solution.copy()
            best_cost = current_cost
    runtime = round(time.time() - start, 4)
    return {"best_solution": best_solution,"total_cost": float(best_cost),"num_hospitals": int(np.sum(best_solution)),"hospital_penalty": float(lambd * int(np.sum(best_solution))),"travel_cost": float(best_cost - lambd * int(np.sum(best_solution))),"cost_history": cost_history,"runtime_seconds": runtime,"expanded_states": total_expanded_states,"viewed_states": total_viewed_states}
