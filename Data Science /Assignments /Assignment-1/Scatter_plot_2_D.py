import numpy as np
import matplotlib.pyplot as plt

list_1 = []
list_2 = []
list_3 = []

num_iterations = 100  
sample_size = 50

def running_average(current_avg, new_value, iteration):
    return (current_avg * (iteration - 1) + new_value) / iteration

def plot_avg_variance(iterations, avg_s_values, true_variance, label, color):
    plt.scatter(iterations, avg_s_values, color=color, s=5, label=label)
    plt.axhline(y=true_variance, color='red', linestyle='--', label='True Variance')
    plt.xlabel("Iterations")
    plt.ylabel(label)
    plt.legend()

def compute_mean(data):
    total = sum(data)
    print("this is the total sum of the data ", total)
    return total / len(data)


def compute_variance(data, mean):
    squared_diffs = [(x - mean)**2 for x in data]
    variance = sum(squared_diffs) / len(squared_diffs)
    return variance


def main():

    alpha   = 0.01
    upper_limit =  1000
    differ_point =  100000
    population = np.linspace(alpha ,  upper_limit  , differ_point) 
    print(population) 
    Avg_s1_2 = 0
    Avg_s2_2 = 0
    Avg_s3_2 = 0

    iterations = []
    true_mean = compute_mean(population)
    true_variance = compute_variance(population, true_mean) 
    print("This is the true variance") 
    print(true_variance)
    print("This is hte mean of the population" , true_mean)
    for iteration in range(1, num_iterations + 1):
        sample = np.random.choice(population, size=sample_size, replace=False)
        s1_2 = sample[0]**2
        s2_2 = sample[1]**2
        s3_2 = sample[2]**2

       #print("Global running averge 1 ")
        Avg_s1_2 = running_average(Avg_s1_2, s1_2, iteration)
        #print("Global running avergae 2")
        Avg_s2_2 = running_average(Avg_s2_2, s2_2, iteration)
        #print("Global running average 3")
        Avg_s3_2 = running_average(Avg_s3_2, s3_2, iteration)

       
        list_1.append(Avg_s1_2)
        list_2.append(Avg_s2_2)
        list_3.append(Avg_s3_2)
        iterations.append(iteration)

    print("This is the iteration list")
    print(iterations)

    plt.figure(figsize=(15, 5))

    print("Plot for the first average")
    plt.subplot(1, 3, 1)
    plot_avg_variance(iterations, list_1, true_variance, 'Avg_s1^2', 'blue')

    print("Plot of the second average")
    plt.subplot(1, 3, 2)
    plot_avg_variance(iterations, list_2, true_variance, 'Avg_s2^2', 'green')

    print("Plot of the third average")
    plt.subplot(1, 3, 3)
    plot_avg_variance(iterations, list_3, true_variance, 'Avg_s3^2', 'purple')

    plt.tight_layout()
    plt.show()

main()

# Final result after all iterations
print("Final Averages after all iterations:")
print(f"Avg_s1^2 = {Avg_s1_2:.2f}")
print(f"Avg_s2^2 = {Avg_s2_2:.2f}")
print(f"Avg_s3^2 = {Avg_s3_2:.2f}")
print(f"True Variance of Population D = {true_variance:.2f}")
