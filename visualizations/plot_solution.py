import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d

# Plot the solution map
def plot_solution_map(population_points, candidate_hospitals, solution, title="Solution Map"):
    
    # Extract selected hospitals
    selected_hospitals = candidate_hospitals[solution == 1]

    # Plot population points
    plt.figure(figsize=(8, 8))
    plt.scatter( population_points[:, 0], population_points[:, 1], c='blue', s=30, label='Population Points', alpha=0.6)

    # Plot selected hospitals
    plt.scatter(selected_hospitals[:, 0],selected_hospitals[:, 1],c='red',s=100,marker='^',label='Selected Hospitals')

    # Plot Voronoi regions if there is more than one hospital
    if len(selected_hospitals) > 1:
        vor = Voronoi(selected_hospitals)
        voronoi_plot_2d(vor,ax=plt.gca(),show_vertices=False,line_colors='orange',ine_width=2,ine_alpha=0.5,oint_size=0)

    plt.title(title)
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.legend()
    plt.grid(True)
    plt.show()


# Example usage
if __name__ == "__main__":
    # Generate random population points
    population_points = np.random.uniform(0, 100, size=(100, 2))
    # Generate random candidate hospitals
    candidate_hospitals = np.random.uniform(0, 100, size=(10, 2))
    # Create a sample solution
    solution = np.zeros(10)
    solution[[1, 3, 7]] = 1  # Select three hospitals
    # Plot the map
    plot_solution_map(population_points,candidate_hospitals,solution)