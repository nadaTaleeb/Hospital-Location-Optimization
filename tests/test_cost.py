import numpy as np

#Import the functions
from data_and_cost import(
    generate_candidate_hospitals,
    euclidean_distance,
    calculate_cost
)


# Test hospital generation
def test_generate_candidate_hospitals():

    hospitals =generate_candidate_hospitals()

    # Make sure 100 hospitals are created
    assert len(hospitals) == 100


# Test distance function
def test_euclidean_distance():

    point1 =np.array([0, 0])
    point2 =np.array([3, 4])

    distance =euclidean_distance(point1, point2)

    # Expected distance = 5
    assert distance ==5


# Test cost function
def test_calculate_cost():

    # Population locations
    population_points =np.array([
        [0, 0],
        [10, 0]
    ])

    # Population weights
    weights = np.array([1, 1])

    # Hospital locations
    candidate_hospitals =np.array([
        [0, 0],
        [10, 0]
    ])

    # Select the first hospital
    solution = np.array([1, 0])

    # Building cost
    lambd =5

    cost = calculate_cost(
        population_points,
        weights,
        candidate_hospitals,
        solution,
        lambd
    )

    # Expected cost = 15
    assert cost ==15