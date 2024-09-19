import numpy as np
import matplotlib.pyplot as plt

print("Question 2 C part")

# Generate the population D from 0.01 to 1000
population = np.linspace(0.01, 1000, 100000)

# Number of iterations and sample size per iteration
num_iterations = 1000
sample_size = 50

# Initialize the averages
Avg_s1_2 = 0
Avg_s2_2 = 0
Avg_s3_2 = 0

# Arrays to store the averages for plotting
avg_s1_values = []
avg_s2_values = []
avg_s3_values = []

# Compute the true population variance
true_variance = np.var(population)

# Iterating multiple times to take random samples and maintain averages
for iteration in range(1, num_iterations + 1):
    # Take a random sample of 50 points from the population
    sample = np.random.choice(population, size=sample_size, replace=True)

    # Calculate sample mean
    sample_mean = np.mean(sample)

    # Compute s1^2, s2^2, s3^2 using the formulas provided
    s1_2 = np.sum((sample - sample_mean) ** 2) / (sample_size + 1)
    s2_2 = np.sum((sample - sample_mean) ** 2) / sample_size
    s3_2 = np.sum((sample - sample_mean) ** 2) / (sample_size - 1)

    # Update the running averages after each iteration
    Avg_s1_2 = (Avg_s1_2 * (iteration - 1) + s1_2) / iteration
    Avg_s2_2 = (Avg_s2_2 * (iteration - 1) + s2_2) / iteration
    Avg_s3_2 = (Avg_s3_2 * (iteration - 1) + s3_2) / iteration

    # Append the averages to arrays for plotting later
    avg_s1_values.append(Avg_s1_2)
    avg_s2_values.append(Avg_s2_2)
    avg_s3_values.append(Avg_s3_2)

# Plot the results
iterations_range = range(1, num_iterations + 1)

plt.figure(figsize=(12, 8))

# Plot for Avg(s1^2)
plt.subplot(3, 1, 1)
plt.plot(iterations_range, avg_s1_values, label='Avg(s1^2)', color='blue')
plt.axhline(y=true_variance, color='red', linestyle='--', label='True Variance')
plt.title('Avg(s1^2) vs Iterations')
plt.xlabel('Iterations')
plt.ylabel('Avg(s1^2)')
plt.legend()

# Plot for Avg(s2^2)
plt.subplot(3, 1, 2)
plt.plot(iterations_range, avg_s2_values, label='Avg(s2^2)', color='green')
plt.axhline(y=true_variance, color='red', linestyle='--', label='True Variance')
plt.title('Avg(s2^2) vs Iterations')
plt.xlabel('Iterations')
plt.ylabel('Avg(s2^2)')
plt.legend()

# Plot for Avg(s3^2)
plt.subplot(3, 1, 3)
plt.plot(iterations_range, avg_s3_values, label='Avg(s3^2)', color='purple')
plt.axhline(y=true_variance, color='red', linestyle='--', label='True Variance')
plt.title('Avg(s3^2) vs Iterations')
plt.xlabel('Iterations')
plt.ylabel('Avg(s3^2)')
plt.legend()

# Display the plots
plt.tight_layout()
plt.show()

# Final result after all iterations
print("Final Averages after all iterations:")
print(f"Avg_s1^2 = {Avg_s1_2:.2f}")
print(f"Avg_s2^2 = {Avg_s2_2:.2f}")
print(f"Avg_s3^2 = {Avg_s3_2:.2f}")
print(f"True Variance of Population D = {true_variance:.2f}")
