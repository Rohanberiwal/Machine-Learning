print("This is the question 2 b code ")
import numpy as np

# Define parameters
num_samples = 50
alpha = 0.01
upper_limit = 1000
differ_point = 100000

def compute_mean(sample):
    return sum(sample) / len(sample)

def compute_variances(sample, mean):
    n = len(sample)
    s2_1 = sum((y - mean) ** 2 for y in sample) / (n + 1)
    s2_2 = sum((y - mean) ** 2 for y in sample) / n
    s2_3 = sum((y - mean) ** 2 for y in sample) / (n - 1)
    return s2_1, s2_2, s2_3

def main():
    population = np.linspace(alpha, upper_limit, differ_point)
    sample = np.random.choice(population, size=num_samples, replace=True)
    

    mean = compute_mean(sample)

    s2_1, s2_2, s2_3 = compute_variances(sample, mean)
    print("Sample:", sample)
    print("Mean (μ):", mean)
    print("Variance s^2_1:", s2_1)
    print("Variance s^2_2:", s2_2)
    print("Variance s^2_3:", s2_3)

main()
