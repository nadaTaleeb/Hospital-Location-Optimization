import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d

# Plot the solution map
def plot_solution_map(population_points, candidate_hospitals, solution, title="Solution Map"):
    
    # Extract selected hospitals
    selected_hospitals = candidate_hospitals[solution == 1]

    # Plot population points
    plt.figure(figsize=(8, 8))
    plt.scatter(population_points[:, 0],population_points[:, 1],c='blue',s=30,label='Population Points',alpha=0.6)

    # Plot selected hospitals
    plt.scatter(selected_hospitals[:, 0],selected_hospitals[:, 1],c='red',s=100,marker='^',label='Selected Hospitals')

    # Plot Voronoi regions if there is more than one hospital
    if len(selected_hospitals) > 1:
        vor = Voronoi(selected_hospitals)
        voronoi_plot_2d(vor, ax=plt.gca(), show_vertices=False, line_colors='orange', line_width=2, line_alpha=0.5, point_size=0)

    plt.title(title)
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":

    from utils.data_generator import generate_population_points, generate_population_weights
    from utils.cost_function import generate_candidate_hospitals
    from algorithms.simulated_annealing import simulated_annealing
    

    population_points = generate_population_points()
    weights = generate_population_weights()
    candidate_hospitals = generate_candidate_hospitals()
    # Run Simulated Annealing
    result = simulated_annealing(population_points,weights,candidate_hospitals,lambd=100,max_iterations=500,initial_temperature=1000,alpha=0.95,selection_rate=0.15,seed=42)
    plot_solution_map( population_points, candidate_hospitals, result["best_solution"], title="Simulated Annealing Solution Map")