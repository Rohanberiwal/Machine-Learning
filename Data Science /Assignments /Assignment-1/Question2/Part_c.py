import numpy as np
import matplotlib.pyplot as plt

print("Question 2 C part")

# Generate the population D from 0.01 to 1000
population = np.linspace(0.01, 1000, 100000)

# Number of iterations and sample size per iteration
num_iterations = 1000
sample_size = 50

# Initialize the averages and lists to store their values for plotting
Avg_s1_2 = 0
Avg_s2_2 = 0
Avg_s3_2 = 0

Avg_s1_2_list = []
Avg_s2_2_list = []
Avg_s3_2_list = []

# Iterating multiple times to take random samples and maintain averages
for iteration in range(1, num_iterations + 1):
    # Take a random sample of 50 points from the population
    sample = np.random.choice(population, size=sample_size, replace=False)
    
    # Here we will compute s1^2, s2^2, s3^2 as the sum of squares of first three values from the sample
    s1_2 = sample[0]**2
    s2_2 = sample[1]**2
    s3_2 = sample[2]**2
    
    # Update the running averages after each iteration
    Avg_s1_2 = (Avg_s1_2 * (iteration - 1) + s1_2) / iteration
    Avg_s2_2 = (Avg_s2_2 * (iteration - 1) + s2_2) / iteration
    Avg_s3_2 = (Avg_s3_2 * (iteration - 1) + s3_2) / iteration
    
    # Store the averages for plotting later
    Avg_s1_2_list.append(Avg_s1_2)
    Avg_s2_2_list.append(Avg_s2_2)
    Avg_s3_2_list.append(Avg_s3_2)
    
    # Print the updated averages after each iteration
    print(f"Iteration {iteration}:")
    print(f"  Avg_s1^2 = {Avg_s1_2:.2f}")
    print(f"  Avg_s2^2 = {Avg_s2_2:.2f}")
    print(f"  Avg_s3^2 = {Avg_s3_2:.2f}")
    print('-' * 30)

# Final result after all iterations
print("Final Averages after all iterations:")
print(f"Avg_s1^2 = {Avg_s1_2:.2f}")
print(f"Avg_s2^2 = {Avg_s2_2:.2f}")
print(f"Avg_s3^2 = {Avg_s3_2:.2f}")

# Plotting the averages over iterations
plt.figure(figsize=(10, 6))

# Plot for Avg_s1^2, Avg_s2^2, and Avg_s3^2
plt.plot(range(1, num_iterations + 1), Avg_s1_2_list, label="Avg_s1^2", color='r', linestyle='-', marker='o')
plt.plot(range(1, num_iterations + 1), Avg_s2_2_list, label="Avg_s2^2", color='g', linestyle='-', marker='x')
plt.plot(range(1, num_iterations + 1), Avg_s3_2_list, label="Avg_s3^2", color='b', linestyle='-', marker='s')

# Add title and labels
plt.title("Evolution of Averages Over Iterations")
plt.xlabel("Iteration")
plt.ylabel("Average Value")
plt.legend()

# Show the plot
plt.grid(True)
plt.show()
