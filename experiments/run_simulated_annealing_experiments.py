import time
import numpy as np
import random
import pandas as pd

from algorithms.simulated_annealing import simulated_annealing
from utils.data_generator import generate_population_points, generate_population_weights
from utils.cost_function import generate_candidate_hospitals

def run_simulated_annealing_lambda_experiments():
    lambda_values = [1, 10, 50, 100]  # We use the lambda values required in the project
    number_of_runs = 10  # We run each lambda more than once because simulated annealing starts randomly
    results = []

    # We generate one fixed problem for fair comparison
    population_points = generate_population_points()
    weights = generate_population_weights()
    candidate_hospitals = generate_candidate_hospitals()

    for lambd in lambda_values:
        costs = []
        runtimes = []
        hospital_counts = []

        for seed in range(number_of_runs):

            #to test the stability of the algorithm
            np.random.seed(seed)
            random.seed(seed)

            
            result= simulated_annealing(population_points,weights,candidate_hospitals,lambd=lambd,max_iterations=500,initial_temperature=1000,alpha=0.95, selection_rate=0.15 ,seed=seed)


            #We record the important values for this run
            costs.append(result["total_cost"])           
            runtimes.append(result["runtime_seconds"])   
            hospital_counts.append(result["num_hospitals"])

        #We summarize the results all runs for this lambda
        results.append({"lambda": lambd,"average_cost": np.mean(costs),"cost_variance": np.var(costs),"average_runtime": np.mean(runtimes),"average_hospitals": np.mean(hospital_counts)})
       
    #save results to CSV so the plotting script can read it
    df = pd.DataFrame(results)
    df.to_csv("sa_summary_results.csv", index=False)
    print("Saved sa_summary_results.csv")

    
    return results


def run_simulated_annealing_parameter_tuning():
    alphas = [0.90, 0.95, 0.99]  # We tune the cooling rate parameter
    lambd = 10
    number_of_runs = 10
    tuning_results = []

    # We use the same problem instance for all experiments
    population_points = generate_population_points()
    weights = generate_population_weights()
    candidate_hospitals = generate_candidate_hospitals()

    for alpha in alphas:

        costs = []
        runtimes = []
        hospital_counts = []

        for seed in range(number_of_runs):
            start_time = time.time()
            result = simulated_annealing(population_points,weights,candidate_hospitals,lambd=lambd,max_iterations=500,initial_temperature=1000,alpha=alpha,selection_rate=0.15 , seed=seed)
            end_time = time.time()

            costs.append(result["total_cost"])
            runtimes.append(result["runtime_seconds"])
            hospital_counts.append(result["num_hospitals"])

        #We summarize the results across all runs for this parameter
        tuning_results.append({"alpha": alpha,"average_cost": np.mean(costs),"cost_variance": np.var(costs),"average_runtime": np.mean(runtimes),"average_hospitals": np.mean(hospital_counts)})
    df = pd.DataFrame(tuning_results)
    df.to_csv("sa_tuning_results.csv", index=False)
    print("Saved sa_tuning_results.csv")

    return tuning_results


if __name__ == "__main__":

    print("Starting Simulated Annealing lambda experiments...")
    lambda_results = run_simulated_annealing_lambda_experiments()
    print("\nFinal Lambda Results")
    for row in lambda_results:
        print("Lambda =", row["lambda"])
        print("Average Cost =", row["average_cost"])
        print("Variance =", row["cost_variance"])
        print("Average Runtime =", row["average_runtime"])
        print("Average Hospitals =", row["average_hospitals"])
        print()

    print("Starting Simulated Annealing parameter tuning...")
    tuning_results = run_simulated_annealing_parameter_tuning()
    
    print("\nFinal Tuning Results")

    for row in tuning_results:
        print("Alpha =", row["alpha"])
        print("Average Cost =", row["average_cost"])
        print("Variance =", row["cost_variance"])
        print("Average Runtime =", row["average_runtime"])
        print("Average Hospitals =", row["average_hospitals"])
        print()
