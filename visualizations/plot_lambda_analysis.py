
import pandas as pd
import matplotlib.pyplot as plt

# Read SA summary results
summary = pd.read_csv("sa_summary_results.csv")

lambda_values= summary["lambda"]
mean_cost= summary["average_cost"]      
mean_runtime= summary["average_runtime"]   
mean_hospitals= summary["average_hospitals"] 
cost_variance = summary["cost_variance"]

# Cost vs Lambda
plt.figure()
plt.plot(lambda_values, mean_cost, marker="o")
plt.title("SA Mean Cost vs Lambda")
plt.xlabel("Lambda")
plt.ylabel("Mean Best Cost")
plt.grid(True)
plt.savefig("sa_cost_vs_lambda.png")
plt.show()

# Runtime vs Lambda
plt.figure()
plt.plot(lambda_values, mean_runtime, marker="o")
plt.title("SA Mean Runtime vs Lambda")
plt.xlabel("Lambda")
plt.ylabel("Mean Runtime (seconds)")
plt.grid(True)
plt.savefig("sa_runtime_vs_lambda.png")
plt.show()

# Hospitals vs Lambda
plt.figure()
plt.plot(lambda_values, mean_hospitals, marker="o")
plt.title("SA Selected Hospitals vs Lambda")
plt.xlabel("Lambda")
plt.ylabel("Mean Number of Selected Hospitals")
plt.grid(True)
plt.savefig("sa_hospitals_vs_lambda.png")
plt.show()

# Variance vs Lambda (bonus — useful for stability analysis in report)
plt.figure()
plt.plot(lambda_values, cost_variance, marker="o", color="orange")
plt.title("SA Cost Variance vs Lambda")
plt.xlabel("Lambda")
plt.ylabel("Cost Variance")
plt.grid(True)
plt.savefig("sa_variance_vs_lambda.png")
plt.show()