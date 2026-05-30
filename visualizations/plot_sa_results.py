import pandas as pd
import matplotlib.pyplot as plt

# Read results file
summary =pd.read_csv("sa_summary_results.csv", header=[0, 1], index_col=0)

# Get values from the table
lambda_values =summary.index
mean_cost =summary[("best_cost", "mean")]
mean_runtime =summary[("runtime", "mean")]
mean_hospitals =summary[("selected_hospitals", "mean")]

# Cost vs Lambda graph
plt.figure()
plt.plot(lambda_values, mean_cost, marker="o")
plt.title("SA Mean Cost vs Lambda")
plt.xlabel("Lambda")
plt.ylabel("Mean Best Cost")
plt.grid(True)
plt.savefig("sa_cost_vs_lambda.png")
plt.show()

# Runtime vs Lambda graph
plt.figure()
plt.plot(lambda_values, mean_runtime, marker="o")
plt.title("SA Mean Runtime vs Lambda")
plt.xlabel("Lambda")
plt.ylabel("Mean Runtime (seconds)")
plt.grid(True)
plt.savefig("sa_runtime_vs_lambda.png")
plt.show()

# Hospitals vs Lambda graph
plt.figure()
plt.plot(lambda_values, mean_hospitals, marker="o")
plt.title("SA Selected Hospitals vs Lambda")
plt.xlabel("Lambda")
plt.ylabel("Mean Number of Selected Hospitals")
plt.grid(True)
plt.savefig("sa_hospitals_vs_lambda.png")
plt.show()