import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, gamma, pi

filepath = "/content/Assignment-1 - Hurricane.csv"

def load_data_from_csv(filepath):
    """
    Load data from a CSV file into a pandas DataFrame.
    """
    return pd.read_csv(filepath)  # Read the CSV file and return as DataFrame

def compute_pearson_correlation(x, y):
    """
    Compute the Pearson correlation coefficient between two variables.
    """
    return pearsonr(x, y)  # Calculate and return the Pearson correlation coefficient

def compute_t_statistics(r, n):
    """
    Compute the t-statistic for the correlation coefficient.
    """
    return r * sqrt((n - 2) / (1 - r**2))  # Calculate the t-statistic based on r and sample size n

def get_critical_t_value(alpha, df):
    """
    Compute the critical t-value for a given significance level and degrees of freedom using a t-distribution table or approximation.
    """
    # Use a simple approximation for the critical t-value based on degrees of freedom
    # This example uses a standard table for commonly used significance levels (e.g., 0.01)
    critical_t_values = {
        1: 63.656,
        2: 9.925,
        3: 5.841,
        4: 4.604,
        5: 4.032,
        6: 3.707,
        7: 3.499,
        8: 3.355,
        9: 3.250,
        10: 3.169,
        20: 2.845,
        30: 2.750,
        50: 2.676,
        100: 2.626,
        1000: 2.576
    }
    if df in critical_t_values:
        return critical_t_values[df]
    else:
        # For degrees of freedom not in the table, return an approximate value
        return 2.626  # This is a rough approximation for larger degrees of freedom

def print_hypotheses():
    """
    Print the null and alternative hypotheses for the correlation test.
    """
    print("Hypotheses:")
    print("Null Hypothesis (H0): There is no correlation between 'Max. sustained winds (mph)' and 'Minimum pressure (mbar)'. (ρ = 0)")
    print("Alternative Hypothesis (H1): There is a correlation between 'Max. sustained winds (mph)' and 'Minimum pressure (mbar)'. (ρ ≠ 0)")

def run_correlation_t_test(df):
    """
    Perform a correlation test and determine if it is statistically significant.
    """
    x = df['Max. sustained winds(mph)']  # Extract the 'Max. sustained winds (mph)' column
    y = df['Minimum pressure(mbar)']  # Extract the 'Minimum pressure (mbar)' column
    
    r, _ = compute_pearson_correlation(x, y)  # Compute the Pearson correlation coefficient
    n = len(x)  # Determine the sample size
    
    t_statistic = compute_t_statistics(r, n)  # Compute the t-statistic for the correlation coefficient
    alpha = 0.01  # Define the significance level (1%)
    critical_t = get_critical_t_value(alpha, n - 2)  # Get the critical t-value for the significance level
    
    print_hypotheses()  # Print the hypotheses
    print(f'\nPearson correlation coefficient: {r:.4f}')  # Print the Pearson correlation coefficient
    print(f'T-statistic: {t_statistic:.4f}')  # Print the calculated t-statistic
    print(f'Critical t-value for 1% significance level: {critical_t:.4f}')  # Print the critical t-value
    
    if abs(t_statistic) > critical_t:  # Check if the absolute t-statistic is greater than the critical value
        print("The correlation is statistically significant at the 1% level.")  # Print if significant
    else:
        print("The correlation is not statistically significant at the 1% level.")  # Print if not significant
    
    plot_t_distribution(t_statistic, critical_t, n - 2)  # Plot the t-distribution with the t-statistic and critical region

def plot_t_distribution(t_statistic, critical_t, df):
    """
    Plot the t-distribution, critical regions, and the t-statistic.
    """
    x = np.linspace(-4, 4, 1000)  # Generate a range of x values
    t_dist = t.pdf(x, df)  # Compute the t-distribution values for x
    norm_dist = norm.pdf(x)  # Compute the standard normal distribution values for x
    
    plt.figure(figsize=(10, 6))  # Create a new figure with a specified size
    plt.plot(x, t_dist, label='T-distribution', color='blue')  # Plot the t-distribution
    plt.plot(x, norm_dist, label='Standard Normal Distribution', color='green', linestyle='--')  # Plot the standard normal distribution
    plt.fill_between(x, t_dist, where=(x >= critical_t) | (x <= -critical_t), color='red', alpha=0.3, label='Critical Region')  # Highlight the critical region
    
    plt.axvline(t_statistic, color='black', linestyle='--', label=f'T-Statistic ({t_statistic:.2f})')  # Draw a line for the t-statistic
    plt.axvline(critical_t, color='red', linestyle=':', label=f'Critical T-value ({critical_t:.2f})')  # Draw a line for the positive critical t-value
    plt.axvline(-critical_t, color='red', linestyle=':', label=f'Critical T-value ({-critical_t:.2f})')  # Draw a line for the negative critical t-value
    
    plt.title('T-Distribution with T-Statistic and Critical Region')  # Set the plot title
    plt.xlabel('Value')  # Label the x-axis
    plt.ylabel('Probability Density')  # Label the y-axis
    plt.legend()  # Display the legend
    plt.grid(True)  # Add a grid to the plot
    plt.show()  # Show the plot

# Load data and run the analysis
df = load_data_from_csv(filepath)  # Load the data from the CSV file
run_correlation_t_test(df)  # Perform the correlation test and display the results

print("Part A is completed successfully")  # Print a completion message
