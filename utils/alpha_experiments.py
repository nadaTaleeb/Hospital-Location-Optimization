import numpy as np
import random
import time
import pandas as pd

# Import Simulated Annealing
from simulated_annealing import simulated_annealing

# Alpha values to test
alpha_values =[0.85, 0.90, 0.95, 0.99]

# Random seeds
seeds =[1, 2, 3, 4, 5]

# Store results
results =[]

# Test different alpha values
for alpha in alpha_values:
    for seed in seeds:

        # Set random seed
        np.random.seed(seed)
        random.seed(seed)

        # Generate test data
        population_points =np.random.uniform(0, 100, size=(100, 2))
        weights = np.random.randint(1, 11, size=100)
        candidate_hospitals =np.random.uniform(0, 100, size=(100, 2))

        # Start timer
        start_time =time.time()

        # Run SA
        best_solution, best_cost, cost_curve, temperature_curve =simulated_annealing(
            population_points,
            weights,
            candidate_hospitals,
            lambd=10,
            initial_temperature=1000,
            alpha=alpha
        )

        # Calculate runtime
        runtime = time.time()-start_time

        # Save results
        results.append({
            "alpha": alpha,
            "seed": seed,
            "best_cost": best_cost,
            "runtime": runtime,
            "selected_hospitals": np.sum(best_solution)
        })

# Create results table
df = pd.DataFrame(results)

# Calculate mean and std
summary = df.groupby("alpha").agg({
    "best_cost": ["mean", "std"],
    "runtime": ["mean", "std"],
    "selected_hospitals": ["mean", "std"]
})

# Print results
print("\nAlpha Tuning Results:")
print(summary)

# Save results to CSV files
df.to_csv("sa_alpha_detailed_results.csv", index=False)
summary.to_csv("sa_alpha_summary_results.csv")

# Show saved files
print("\nFiles saved:")
print("sa_alpha_detailed_results.csv")
print("sa_alpha_summary_results.csv")