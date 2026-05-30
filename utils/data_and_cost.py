import numpy as np

# Generate candidate hospital locations
def generate_candidate_hospitals(m=100,seed=42):

    np.random.seed(seed)

    candidate_hospitals = np.random.uniform(0, 100, size=(m,2))

    return candidate_hospitals


# Calculate distance between two points
def euclidean_distance(point1,point2):

    distance = np.sqrt(
        (point1[0] - point2[0])**2 +
        (point1[1] - point2[1])**2
    )

    return distance


# Calculate total cost
def calculate_cost(population_points,
                   weights,
                   candidate_hospitals,
                   solution,
                   lambd):

    # Get selected hospitals
    selected_hospitals = candidate_hospitals[solution == 1]

    # Return infinity if no hospital is selected
    if len(selected_hospitals) == 0:
        return float("inf")

    total_travel_cost =0

    # Loop through all population points
    for i, person_location in enumerate(population_points):

        # Calculate distances to hospitals
        distances = [
            euclidean_distance(person_location, hospital)
            for hospital in selected_hospitals
        ]

        # Find nearest hospital
        nearest_distance =min(distances)

        # Add weighted travel cost
        total_travel_cost += weights[i] *nearest_distance

    # Calculate hospital building cost
    building_cost =lambd * len(selected_hospitals)

    # Total cost
    total_cost =total_travel_cost + building_cost

    return total_cost


# Run simple tests
if __name__ == "__main__":

    hospitals = generate_candidate_hospitals()

    print(hospitals)

    print("Number of hospitals:",len(hospitals))

    # Distance test
    point1 = np.array([0,0])
    point2 = np.array([3,4])

    distance = euclidean_distance(point1, point2)

    print("Distance test:", distance)

    # Cost function test
    population_points = np.array([
        [0, 0],
        [10, 0]
    ])

    weights = np.array([1, 1])

    candidate_hospitals = np.array([
        [0, 0],
        [10, 0]
    ])

    solution =np.array([1, 0])

    lambd = 5

    cost =calculate_cost(
        population_points,
        weights,
        candidate_hospitals,
        solution,
        lambd
    )

    print("Cost test:", cost)