import numpy as np
import matplotlib.pyplot as plt

list1 = 0
list2 = 0
list3 = 0

list1_values = []
list2_values = []
list3_values = []

def compute_true_variance_and_mean(population):
    n = len(population)
    mean = sum(population) / n
    variance = sum((x - mean) ** 2 for x in population) / n
    return mean, variance

def main():
    print("Question 2 C part")

    population = np.linspace(0.01, 1000, 100000)
    true_mean, true_variance = compute_true_variance_and_mean(population)
    print(f"True Mean of the population: {true_mean:.2f}")
    print(f"True Variance of the population: {true_variance:.2f}")

    num_iterations = 30
    sample_size = 50

    for iteration in range(1, num_iterations + 1):
        sample = np.random.choice(population, size=sample_size, replace=False)
        s1_2 = sample[0] ** 2
        s2_2 = sample[1] ** 2
        s3_2 = sample[2] ** 2

        global list1, list2, list3
        list1 = (list1 * (iteration - 1) + s1_2) / iteration
        list2 = (list2 * (iteration - 1) + s2_2) / iteration
        list3 = (list3 * (iteration - 1) + s3_2) / iteration

        list1_values.append(list1)
        list2_values.append(list2)
        list3_values.append(list3)

        print(f"Iteration {iteration}:")
        print(f"  Avg_s1^2 = {list1:.2f}")
        print(f"  Avg_s2^2 = {list2:.2f}")
        print(f"  Avg_s3^2 = {list3:.2f}")
        print('-' * 30)

    print("Final Averages after all iterations:")
    print(f"Avg_s1^2 = {list1:.2f}")
    print(f"Avg_s2^2 = {list2:.2f}")
    print(f"Avg_s3^2 = {list3:.2f}")

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_iterations + 1), list1_values, label="Avg_s1^2", color='r', linestyle='-', marker='o')
    plt.plot(range(1, num_iterations + 1), list2_values, label="Avg_s2^2", color='g', linestyle='-', marker='x')
    plt.plot(range(1, num_iterations + 1), list3_values, label="Avg_s3^2", color='b', linestyle='-', marker='s')

    plt.title("Evolution of Averages Over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Average Value")
    plt.legend()
    plt.grid(True)
    plt.show()

main()
