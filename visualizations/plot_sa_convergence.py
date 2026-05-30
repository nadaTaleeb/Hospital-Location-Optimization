import numpy as np
import random
import matplotlib.pyplot as plt

# Import Simulated Annealing
from simulated_annealing import simulated_annealing

# Set random seed
np.random.seed(42)
random.seed(42)

# Generate test data
population_points = np.random.uniform(0,100, size=(100,2))
weights = np.random.randint(1, 11, size=100)
candidate_hospitals = np.random.uniform(0, 100, size=(100,2))

# Run SA
best_solution, best_cost, cost_curve, temperature_curve = simulated_annealing(
    population_points,
    weights,
    candidate_hospitals,
    lambd=10,
    initial_temperature=1000,
    alpha=0.95
)

# Convergence graph
plt.figure()
plt.plot(cost_curve)
plt.title("SA Convergence Curve")
plt.xlabel("Iteration")
plt.ylabel("Best Cost")
plt.grid(True)
plt.savefig("sa_convergence_curve.png")
plt.show()

# Temperature graph
plt.figure()
plt.plot(temperature_curve)
plt.title("SA Temperature Cooling Schedule")
plt.xlabel("Iteration")
plt.ylabel("Temperature")
plt.grid(True)
plt.savefig("sa_temperature_curve.png")
plt.show()