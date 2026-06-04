import numpy as np
import random
import math

from utils.cost_function import calculate_cost, generate_candidate_hospitals
from utils.data_generator import generate_population_points, generate_population_weights
from algorithms.hill_climbing import generate_successors

# Check that inputs are valid before running the algorithm
def validate_inputs(population_points, weights, candidate_hospitals, lambd):
    population_points = np.array(population_points, dtype=float)
    weights = np.array(weights, dtype=float)
    candidate_hospitals = np.array(candidate_hospitals, dtype=float)

    if population_points.ndim != 2 or population_points.shape[1] != 2:
        raise ValueError("population points must have shape (n, 2).")

    if candidate_hospitals.ndim != 2 or candidate_hospitals.shape[1] != 2:
        raise ValueError("candidate hospitals must have shape (m, 2).")

    if len(weights) != len(population_points):
        raise ValueError("weights length must equal number of population points.")

    if len(candidate_hospitals) == 0:
        raise ValueError("candidate hospitals cannot be empty.")

    if lambd <= 0:
        raise ValueError("lambda must be positive.")

    if np.any(weights < 0):
        raise ValueError("weights cannot be negative.")

    return population_points, weights, candidate_hospitals


# Generate a random initial solution
def generate_initial_solution(num_candidates, selection_rate=0.15):
    if num_candidates <= 0:
        raise ValueError("num_candidates must be positive.")

    if selection_rate <= 0 or selection_rate > 1:
        raise ValueError("selection rate must be between 0 and 1.")

    solution = np.zeros(num_candidates, dtype=int)

    num_selected = max(1, int(num_candidates * selection_rate))

    selected_indices = random.sample(range(num_candidates),num_selected)
    solution[selected_indices] = 1
    return solution


# Compute the probability of accepting the new solution
def acceptance_probability(current_cost, new_cost, temperature):
    if temperature <= 0:
        raise ValueError("temperature must be positive.")

    # Better solutions are always accepted
    if new_cost < current_cost:
        return 1.0

    # Worse solutions may be accepted depending on temperature
    return math.exp((current_cost - new_cost) / temperature)


# Simulated Annealing algorithm
def simulated_annealing(population_points,weights,candidate_hospitals,lambd,max_iterations=500, initial_temperature=1000, alpha=0.95, selection_rate=0.15):

    population_points, weights, candidate_hospitals = validate_inputs(population_points, weights, candidate_hospitals, lambd)
    if max_iterations <= 0:
        raise ValueError("max iterations must be positive.")

    if initial_temperature <= 0:
        raise ValueError("initial temperature must be positive.")

    if alpha <= 0 or alpha >= 1:
        raise ValueError("alpha must be between 0 and 1.")

    num_candidates = len(candidate_hospitals)

    current_solution = generate_initial_solution(num_candidates, selection_rate)

    current_cost = calculate_cost(population_points,weights,candidate_hospitals,current_solution,lambd)
    best_solution = current_solution.copy()
    best_cost = current_cost
    temperature = initial_temperature

    # Store values 
    cost_curve = []
    temperature_curve = []

    for iteration in range(max_iterations):
        successors = generate_successors(current_solution) # Use the same successors used in Hill Climbing

        if len(successors) == 0:
            break
        neighbor_solution = random.choice(successors) # Choose one successor randomly

        neighbor_cost = calculate_cost( population_points, weights, candidate_hospitals, neighbor_solution, lambd)

        #acceptance probability
        probability = acceptance_probability( current_cost, neighbor_cost, temperature)

        # Decide whether to move to the neighbor
        if probability > random.random():
            current_solution = neighbor_solution.copy()
            current_cost = neighbor_cost

        # Keep track of the best solution found so far
        if current_cost < best_cost:
            best_solution = current_solution.copy()
            best_cost = current_cost

        cost_curve.append(best_cost)
        temperature_curve.append(temperature)

        # Decrease temperature gradually
        temperature = temperature * alpha

    return best_solution, best_cost, cost_curve, temperature_curve


if __name__ == "__main__":

    np.random.seed(42)
    random.seed(42)

    population_points = generate_population_points()
    weights = generate_population_weights()
    candidate_hospitals = generate_candidate_hospitals()

    lambd = 10

    best_solution, best_cost, cost_curve, temperature_curve = simulated_annealing(population_points,weights,candidate_hospitals,lambd)
    print("Best cost:", best_cost)
    print("Number of selected hospitals:", int(np.sum(best_solution)))
    print("Selected hospital indices:", np.where(best_solution == 1)[0])
    print("Expanded states:", len(cost_curve))
    print("Viewed states:", len(cost_curve))