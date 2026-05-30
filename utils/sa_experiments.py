import numpy as np
import random
import time
import pandas as pd

# Import Simulated Annealing
from simulated_annealing import simulated_annealing


# Run experiments with different lambda values
def run_sa_experiments():

    # Lambda values to test
    lambda_values = [1,10,50,100]

    # Random seeds
    seeds = [1,2,3,4,5]

    # Store all results
    results = []

    # Loop through lambda values
    for lambd in lambda_values:

        # Loop through seeds
        for seed in seeds:

            # Set random seed
            np.random.seed(seed)
            random.seed(seed)

            # Generate test data
            population_points =np.random.uniform(0, 100, size=(100, 2))
            weights = np.random.randint(1, 11, size=100)
            candidate_hospitals = np.random.uniform(0, 100, size=(100, 2))

            # Start timer
            start_time = time.time()

            # Run SA
            best_solution, best_cost, cost_curve, temperature_curve = simulated_annealing(
                population_points,
                weights,
                candidate_hospitals,
                lambd,
                initial_temperature=1000,
                alpha=0.95
            )

            # Calculate runtime
            runtime = time.time() -start_time

            # Save results
            results.append({
                "lambda": lambd,
                "seed": seed,
                "best_cost": best_cost,
                "runtime": runtime,
                "selected_hospitals": np.sum(best_solution)
            })

    # Create results table
    df = pd.DataFrame(results)

    # Calculate mean and std
    summary = df.groupby("lambda").agg({
        "best_cost": ["mean", "std"],
        "runtime": ["mean", "std"],
        "selected_hospitals": ["mean", "std"]
    })

    # Print detailed results
    print("\nDetailed Results:")
    print(df)

    # Print summary table
    print("\nSummary Table:")
    print(summary)

    # Save results to CSV files
    df.to_csv("sa_detailed_results.csv", index=False)
    summary.to_csv("sa_summary_results.csv")

    # Show saved files
    print("\nFiles saved:")
    print("sa_detailed_results.csv")
    print("sa_summary_results.csv")


# Run the experiments
if __name__ == "__main__":
    run_sa_experiments()