import time
import numpy as np
import pandas as pd

from algorithms.hill_climbing import random_restart_hill_climbing
from utils.data_generator import generate_population_points, generate_population_weights
from utils.cost_function import generate_candidate_hospitals


def run_hill_climbing_lambda_experiments():

    lambda_values = [1, 10, 50, 100]  #lambda values
    number_of_runs = 2  # We run each lambda 2 because hill climbing starts randomly
    results = []

    # We generate one fixed problem for fair comparison
    population_points = generate_population_points()
    weights = generate_population_weights()
    candidate_hospitals = generate_candidate_hospitals()

    for lambd in lambda_values:
        print("\nRunning lambda =", lambd)

        costs = []
        runtimes = []
        hospital_counts = []

        for seed in range(number_of_runs):
            print("  Run", seed + 1, "started")
            # We use different seeds to test the stability of the algorithm
            np.random.seed(seed)
            start_time = time.time()
            result= random_restart_hill_climbing(population_points,weights,candidate_hospitals,lambd=lambd,restart_times=2,max_iterations=100,selection_rate=0.15 )
            end_time = time.time()

            #We record the important values for this run
            costs.append(result["total_cost"])            
            runtimes.append(result["runtime_seconds"])    
            hospital_counts.append(result["num_hospitals"]) 

            print("Run", seed + 1, "finished")
            print("  Cost =", result["total_cost"])
            print("  Hospitals =", result["num_hospitals"])
            print("Time =", round(end_time - start_time, 4), "seconds")

        # We summarize the results all runs
        results.append({"lambda": lambd,"average_cost": np.mean(costs),"cost_variance": np.var(costs),"average_runtime": np.mean(runtimes),"average_hospitals": np.mean(hospital_counts)})
    
    # save results to CSV
    df = pd.DataFrame(results)
    df.to_csv("hc_summary_results.csv", index=False)
    print("Saved hc_summary_results.csv")
    
    return results


def run_hill_climbing_parameter_tuning():
    selection_rates = [0.10, 0.15]  # We tune the initial selection rate parameter
    lambd = 10
    number_of_runs = 2
    tuning_results = []

    # We use the same problem instance for all experiments
    population_points = generate_population_points()
    weights = generate_population_weights()
    candidate_hospitals = generate_candidate_hospitals()

    for selection_rate in selection_rates:
        print("\nRunning selection rate =", selection_rate)

        costs = []
        runtimes = []
        hospital_counts = []

        for seed in range(number_of_runs):
            print("  Run", seed + 1, "started")

            # We change the seed to see if the parameter is stable
            np.random.seed(seed)

            start_time = time.time()
            result= random_restart_hill_climbing(population_points,weights, candidate_hospitals,lambd=lambd,restart_times=2,max_iterations=100,selection_rate=selection_rate)
            end_time = time.time()

            # We record the important values for this run
            costs.append(result["total_cost"])            
            runtimes.append(result["runtime_seconds"])    
            hospital_counts.append(result["num_hospitals"]) 

            print("  Run", seed + 1, "finished")
            print("  Cost =", result["total_cost"])
            print("  Hospitals =", result["num_hospitals"])
            print("  Time =", round(end_time - start_time, 4), "seconds")

        # We summarize the results all runs
        tuning_results.append({"selection_rate": selection_rate,"average_cost": np.mean(costs),"cost_variance": np.var(costs),"average_runtime": np.mean(runtimes),"average_hospitals": np.mean(hospital_counts) })

   #save tuning results to CSV
    df = pd.DataFrame(tuning_results)
    df.to_csv("hc_tuning_results.csv", index=False)
    print("Saved hc_tuning_results.csv")
    
    return tuning_results


if __name__ == "__main__":

    print("Starting Hill Climbing lambda experiments...")
    lambda_results = run_hill_climbing_lambda_experiments()

    print("\nFinal Lambda Results")

    for row in lambda_results:
        print("Lambda =", row["lambda"])
        print("Average Cost =", row["average_cost"])
        print("Variance =", row["cost_variance"])
        print("Average Runtime =", row["average_runtime"])
        print("Average Hospitals =", row["average_hospitals"])
        print()

    print("Starting Hill Climbing parameter tuning...")
    tuning_results = run_hill_climbing_parameter_tuning()

    print("\nFinal Tuning Results")

    for row in tuning_results:
        print("Selection Rate =", row["selection_rate"])
        print("Average Cost =", row["average_cost"])
        print("Variance =", row["cost_variance"])
        print("Average Runtime =", row["average_runtime"])
        print("Average Hospitals =", row["average_hospitals"])
        print()