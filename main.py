# Hospital Location Optimization
# ENCS3340 Project 1

# Team Members:
# - Nada
# - Mayar

# Workflow:
# 1. Generate data
# 2. Run algorithm
# 3. Collect results
# 4. Generate plots
# 5. Save outputs
import numpy as np


# Data generation
from utils.data_generator import (generate_population_points,generate_population_weights)
from utils.cost_function import generate_candidate_hospitals

# Algorithms
from algorithms.hill_climbing import random_restart_hill_climbing
from algorithms.simulated_annealing import simulated_annealing


# Visualization
# from visualizations.plot_solution import plot_solution
# from visualizations.plot_convergence import plot_convergence
# from visualizations.plot_lambda_analysis import plot_lambda_analysis

def main():
    # Configuration parameters
    seed = 42
    lambd  = 50
    print("Main is running...")
    np.random.seed(seed)

    # Generate input data
    population_points = generate_population_points()
    weights = generate_population_weights()
    candidate_hospitals = generate_candidate_hospitals()

    # Run Hill Climbing
    hill_climbing_result = random_restart_hill_climbing(population_points,weights,candidate_hospitals,lambd=lambd,restart_times=10,max_iterations=500,selection_rate=0.15,seed=seed)

    # Run Simulated Annealing
    simulated_annealing_result = simulated_annealing(population_points,weights,candidate_hospitals,lambd=lambd,max_iterations=500,initial_temperature=1000,alpha=0.95,selection_rate=0.15,seed=seed)

    # Print Hill Climbing results
    print("\n Hill Climbing____________________ ")
    print("Best cost:", hill_climbing_result["total_cost"])
    print("Number of selected hospitals:",hill_climbing_result["num_hospitals"])
    print("Selected hospital indices:",np.where(hill_climbing_result["best_solution"] == 1)[0])
    print("Expanded states:",hill_climbing_result["expanded_states"])
    print("Viewed states:",hill_climbing_result["viewed_states"])
    print("Runtime:", hill_climbing_result["runtime_seconds"])


    # Print Simulated Annealing results
    print("\nSimulated Annealing_________________")
    print("Best cost:",simulated_annealing_result["total_cost"])
    print("Number of selected hospitals:",simulated_annealing_result["num_hospitals"])
    print("Selected hospital indices:",np.where(simulated_annealing_result["best_solution"] == 1)[0])
    print("Iterations:",simulated_annealing_result["iterations"])
    print("Runtime:",simulated_annealing_result["runtime_seconds"])
    


    # Compare algorithms
    print("\nComparison____________________________")

    if hill_climbing_result["total_cost"] < simulated_annealing_result["total_cost"]:
        print("Hill Climbing found the better solution.")

    elif simulated_annealing_result["total_cost"] < hill_climbing_result["total_cost"]:
        print("Simulated Annealing found the better solution.")

    else:
        print("Both algorithms found the same cost.")


    # ============================================
    # Future tasks
    # ============================================
    # plot_solution(...)
    # plot_convergence(...)
    # plot_lambda_analysis(...)

    # run_hill_climbing_experiments()
    # run_simulated_annealing_experiments()

    # save_results(...)


if __name__ == "__main__":
    main()