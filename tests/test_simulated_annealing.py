import numpy as np

from algorithms.simulated_annealing import (generate_initial_solution,acceptance_probability,simulated_annealing)

def test_generate_initial_solution():
    # Check that we generate a solution with the correct number of hospitals
    solution = generate_initial_solution(100)
    assert len(solution) == 100


def test_solution_is_binary():
    # Check that our solution contains only 0s and 1s
    solution = generate_initial_solution(100)
    assert np.all((solution == 0) | (solution == 1))


def test_acceptance_probability_better_solution():
    # Check that we always accept a better solution
    probability = acceptance_probability(current_cost=100,new_cost=80,temperature=1000)
    assert probability == 1.0

def test_acceptance_probability_worse_solution():
    # Check that a worse solution 
    probability = acceptance_probability(current_cost=100,new_cost=120,temperature=1000)
    assert 0 < probability < 1


def test_simulated_annealing_runs():
    # Check that our simulated annealing algorithm runs without errors
    population_points = np.random.uniform(0, 100, size=(20, 2))
    weights = np.random.randint(1, 11, size=20)
    candidate_hospitals = np.random.uniform(0, 100, size=(20, 2))

    result = simulated_annealing(population_points, weights, candidate_hospitals,lambd=10, max_iterations=50, seed=42)

    assert result["best_solution"] is not None
    assert result["total_cost"] >= 0
    assert len(result["cost_curve"]) <= 50
    assert len(result["temperature_curve"]) <= 50
    assert result["iterations"] <= 50
    assert result["num_hospitals"] >= 1
    assert result["runtime_seconds"] >= 0