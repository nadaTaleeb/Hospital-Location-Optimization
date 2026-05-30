from utils.data_generator import (generate_population_points,generate_population_weights)


def test_population_points_shape(): #check that 100 population are generated
    points = generate_population_points()
    assert points.shape == (100, 2)


def test_population_points_range(): #check that coordinates are within [0,100]
    points = generate_population_points()
    assert points.min() >= 0
    assert points.max() <= 100


def test_population_weights_shape(): #check that 100 weights are generated
    weights = generate_population_weights()
    assert weights.shape == (100,)


def test_population_weights_range():#check that weights are within [1,10]
    weights = generate_population_weights()
    assert weights.min() >= 1
    assert weights.max() <= 10