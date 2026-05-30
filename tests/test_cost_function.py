import numpy as np

from utils.cost_function import (generate_candidate_hospitals,euclidean_distance,calculate_cost)


def test_generate_candidate_hospitals(): #check that 100 candidate hospitals are generated
    hospitals = generate_candidate_hospitals()
    assert len(hospitals) == 100



def test_euclidean_distance(): #check Euclidean distance calculation

    point1 = np.array([0, 0])
    point2 = np.array([3, 4])
    distance = euclidean_distance(point1, point2)
    assert np.isclose(distance, 5.0)



def test_calculate_cost(): #check total cost calculation
    population_points = np.array([[0, 0],[10, 0]])
    weights = np.array([1, 1])
    candidate_hospitals = np.array([[0, 0],[10, 0]])
    solution = np.array([1, 0])
    lambd = 5
    cost = calculate_cost(population_points,weights,candidate_hospitals,solution,lambd )

    assert np.isclose(cost, 15.0)