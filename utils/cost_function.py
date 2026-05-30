import numpy as np

#Generate candidate hospital locations
def generate_candidate_hospitals(m=100, seed=42):
    np.random.seed(seed)
    candidate_hospitals = np.random.uniform( 0, 100, size=(m, 2) )
    return candidate_hospitals


#Calculate Euclidean distance between two points
def euclidean_distance(point1, point2):
    distance = np.sqrt((point1[0] - point2[0]) ** 2 +(point1[1] - point2[1]) ** 2)
    return distance


# Calculate total optimization cost
def calculate_cost(population_points,weights,candidate_hospitals,solution,lambd):

    selected_hospitals = candidate_hospitals[solution == 1] #Extract selected hospitals from solution vector

    
    if len(selected_hospitals) == 0:
        return float("inf")  #Invalid solution 

    total_travel_cost = 0

    for i, person_location in enumerate(population_points): #Find nearest hospital for each population point

        distances = [euclidean_distance(person_location, hospital) for hospital in selected_hospitals]
        nearest_distance = min(distances)
        total_travel_cost += weights[i] * nearest_distance

    # Construction cost
    building_cost = lambd * len(selected_hospitals)
    total_cost = total_travel_cost + building_cost
    return total_cost