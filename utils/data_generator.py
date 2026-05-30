import numpy as np #NumPy is used to generate population points and weights efficiently


def generate_population_points(n=100, seed=42):
    np.random.seed(seed)   #Set a fixed random seed for experiments

    points = np.random.uniform( low=0,high=100,size=(n, 2))  #Generate n random (x,y) in the range [0,100]
    return points


def generate_population_weights(n=100, seed=42):
    np.random.seed(seed + 1)  # Use a different seed to generate independent weights

    weights = np.random.randint(low=1,high=11,size=n) #Generate integer weights
    return weights


def save_population_data(filename, points, weights):

    np.savez(filename,population_points=points,population_weights=weights)  #Store population and weights


def load_population_data(filename):
    data = np.load(filename) #Load the saved dataset
    points = data["population_points"]  #Extract coordinates and weights
    weights = data["population_weights"]
    return points, weights