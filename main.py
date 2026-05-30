"""
Hospital Location Optimization
ENCS3340 Project 1

Team Members:
- Nada
- Mayar

Workflow:
1. Generate data
2. Run algorithm
3. Collect results
4. Generate plots
5. Save outputs
"""

# =====================================================
# TODO (NADA)
# =====================================================
# 1. Generate population points
# 2. Generate population weights
# 3. Load input data
# 4. Run Hill Climbing algorithm
# 5. Generate solution map visualization
# 6. Run experiments for lambda values
# 7. Collect runtime and cost statistics
# 8. Save Hill Climbing results


# =====================================================
# TODO (MAYAR)
# =====================================================
# 1. Implement cost function
# 2. Run Simulated Annealing algorithm
# 3. Generate convergence plots
# 4. Generate lambda analysis plots
# 5. Collect runtime and cost statistics
# 6. Save Simulated Annealing results
# 7. Prepare comparison tables


# =====================================================
# TODO (SHARED)
# =====================================================
# 1. Load configuration parameters
# 2. Run selected algorithm
# 3. Save CSV results
# 4. Generate final report figures
# 5. Execute integration tests
# 6. Prepare demo-ready outputs
from utils.data_generator import (
    generate_population_points,
    generate_population_weights,
    save_population_data,
    load_population_data
)

def main():

    points = generate_population_points()
    weights = generate_population_weights()

    save_population_data(
        "population_data.npz",
        points,
        weights
    )

    loaded_points, loaded_weights = load_population_data(
        "population_data.npz"
    )

    print("First 5 points:")
    print(loaded_points[:5])

    print("\nFirst 5 weights:")
    print(loaded_weights[:5])


if __name__ == "__main__":
    main()