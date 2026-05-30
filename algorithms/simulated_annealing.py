import numpy as np
import random
import math

from data_and_cost import calculate_cost


def generate_initial_solution(num_candidates, selection_ratio=0.15):

    solution = np.zeros(num_candidates, dtype=int)

    num_selected = int(num_candidates * selection_ratio)

    selected_indices = random.sample(
        range(num_candidates),
        num_selected
    )

    solution[selected_indices] = 1

    return solution


def generate_neighbor(solution):

    neighbor = solution.copy()

    index = random.randint(
        0,
        len(solution) - 1
    )

    neighbor[index] = 1 - neighbor[index]

    return neighbor


def acceptance_probability(current_cost,
                           new_cost,
                           temperature):

    if new_cost < current_cost:
        return 1.0

    return math.exp(
        (current_cost - new_cost) /
        temperature
    )


def simulated_annealing(population_points,
                        weights,
                        candidate_hospitals,
                        lambd,
                        max_iterations=500,
                        initial_temperature=1000,
                        alpha=0.95):

    num_candidates = len(candidate_hospitals)

    current_solution = generate_initial_solution(
        num_candidates
    )

    current_cost = calculate_cost(
        population_points,
        weights,
        candidate_hospitals,
        current_solution,
        lambd
    )

    best_solution = current_solution.copy()
    best_cost = current_cost

    temperature = initial_temperature

    cost_curve = []
    temperature_curve = []

    for iteration in range(max_iterations):

        neighbor_solution = generate_neighbor(
            current_solution
        )

        neighbor_cost = calculate_cost(
            population_points,
            weights,
            candidate_hospitals,
            neighbor_solution,
            lambd
        )

        probability = acceptance_probability(
            current_cost,
            neighbor_cost,
            temperature
        )

        if probability > random.random():

            current_solution = neighbor_solution

            current_cost = neighbor_cost

        if current_cost < best_cost:

            best_solution = current_solution.copy()

            best_cost = current_cost

        cost_curve.append(best_cost)

        temperature_curve.append(
            temperature
        )

        temperature = temperature * alpha

    return (
        best_solution,
        best_cost,
        cost_curve,
        temperature_curve
    )


if __name__ == "__main__":

    np.random.seed(42)
    random.seed(42)

    population_points = np.random.uniform(
        0,
        100,
        size=(100, 2)
    )

    weights = np.random.randint(
        1,
        11,
        size=100
    )

    candidate_hospitals = np.random.uniform(
        0,
        100,
        size=(100, 2)
    )

    lambd = 10

    (
        best_solution,
        best_cost,
        cost_curve,
        temperature_curve

    ) = simulated_annealing(
        population_points,
        weights,
        candidate_hospitals,
        lambd
    )

    print("Best cost:", best_cost)

    print(
        "Selected hospitals:",
        np.sum(best_solution)
    )

    print(
        "First 10 cost values:",
        cost_curve[:10]
    )

    print(
        "First 10 temperatures:",
        temperature_curve[:10]
    )