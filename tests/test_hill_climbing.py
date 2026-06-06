import numpy as np

from algorithms.hill_climbing import (generate_initial_solution,generate_successors,random_restart_hill_climbing)


def test_generate_initial_solution(): #Check that we generate a solution with the correct number of hospitals
    solution = generate_initial_solution(100)
    assert len(solution) == 100


def test_solution_is_binary(): #Check that our solution contains only 0s and 1s
    solution = generate_initial_solution(100)
    assert np.all((solution == 0) | (solution == 1))

def test_generate_successor():
    solution = np.array([1, 0, 1, 0]) # Check that we generate one successor with the same length

    successor = generate_successors(solution)

    assert len(successor) == len(solution)

def test_successor_is_binary():
    # Check that the generated successor contains only 0s and 1s
    solution = np.array([1, 0, 1, 0])
    successor = generate_successors(solution)
    assert np.all((successor == 0) | (successor == 1))

def test_successor_changes_one_bit():
    # Check that the successor differs by exactly one decision
    solution = np.array([1, 0, 1, 0])

    successor = generate_successors(solution)

    diff = np.sum(solution != successor)

    assert diff == 1


def test_hill_climbing_runs():  # Check that algorithm runs without errors
    population_points = np.random.uniform(0, 100, size=(20, 2))
    weights = np.random.randint( 1, 11, size=20)
    candidate_hospitals = np.random.uniform(0, 100, size=(20, 2))
    result = random_restart_hill_climbing( population_points, weights, candidate_hospitals, lambd=10, restart_times=2, seed=42)

    assert result["best_solution"] is not None
    assert result["total_cost"] >= 0
    assert result["num_hospitals"] >= 1
    assert result["runtime_seconds"] >= 0
    assert "cost_history" in result
    assert "expanded_states" in result
    assert "viewed_states" in result