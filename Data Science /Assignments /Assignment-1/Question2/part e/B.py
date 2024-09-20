import numpy as np

print("This is the e part for teh b running the code several time to recrod the output ")

num_samples = 50
num_iterations =  500
alpha = 0.01
upper_limit = 1000
differ_point = 100000

def printer(true_variance , Avg_s1_2 , Avg_s2_2 , Avg_s3_2) :
  dif1 = abs(true_variance - Avg_s1_2)
  dif2 = abs(true_variance - Avg_s2_2)
  dif3 = abs(true_variance - Avg_s3_2)
  lowest =  min(dif1, dif2 , dif3)

  print(lowest)

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
    true_mean = compute_mean(population)
    true_variance = compute_variances(population, true_mean)[1]  
    Avg_s2_1 = 0
    Avg_s2_2 = 0
    Avg_s2_3 = 0

    for iteration in range(num_iterations):
        sample = np.random.choice(population, size=num_samples, replace=True)
        mean = compute_mean(sample)
        
        s2_1, s2_2, s2_3 = compute_variances(sample, mean)
        
        Avg_s2_1 = (Avg_s2_1 * iteration + s2_1) / (iteration + 1)
        Avg_s2_2 = (Avg_s2_2 * iteration + s2_2) / (iteration + 1)
        Avg_s2_3 = (Avg_s2_3 * iteration + s2_3) / (iteration + 1)
        print(f"Iteration {iteration + 1}:")
        print("Sample:", sample)
        print("Mean (μ):", mean)
        print("Variance s^2_1:", s2_1)
        print("Variance s^2_2:", s2_2)
        print("Variance s^2_3:", s2_3)
        print()

    print("This is the true variance" , true_variance)
    print("Average Variance s^2_1:", Avg_s2_1)
    print("Average Variance s^2_2:", Avg_s2_2)
    print("Average Variance s^2_3:", Avg_s2_3)
    printer(true_variance , Avg_s1_2 , Avg_s2_2 , Avg_s3_2)
main()
