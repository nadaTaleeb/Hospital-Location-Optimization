import numpy as np
import random
import matplotlib.pyplot as plt

# Import Simulated Annealing
from algorithms.simulated_annealing import simulated_annealing

# Set random seed
np.random.seed(42)
random.seed(42)

# Generate test data
population_points = np.random.uniform(0,100, size=(100,2))
weights = np.random.randint(1, 11, size=100)
candidate_hospitals = np.random.uniform(0, 100, size=(100,2))

# Run SA
result = simulated_annealing(population_points,weights,candidate_hospitals,lambd=10,initial_temperature=1000,alpha=0.95 , seed=42)

cost_curve = result["cost_curve"]
temperature_curve = result["temperature_curve"]

# Convergence convergence graph
plt.figure()
plt.plot(cost_curve)
plt.title("SA Convergence Curve")
plt.xlabel("Iteration")
plt.ylabel("Best Cost")
plt.grid(True)
plt.savefig("sa_convergence_curve.png")
plt.show()

# Temperature cooling graph
plt.figure()
plt.plot(temperature_curve)
plt.title("SA Temperature Cooling Schedule")
plt.xlabel("Iteration")
plt.ylabel("Temperature")
plt.grid(True)
plt.savefig("sa_temperature_curve.png")
plt.show()