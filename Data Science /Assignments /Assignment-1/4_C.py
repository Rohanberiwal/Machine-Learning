import pandas as pd
import numpy as np
from scipy.stats import poisson, chisquare
import matplotlib.pyplot as plt

filepath = "/content/Assignment-1 - Hurricane.csv"

def load_data_from_csv(filepath):
    return pd.read_csv(filepath)

def test_poisson_distribution(df, alpha=0.10):
    # Extract the 'Max. sustained winds (mph)' column
    winds = df['Max. sustained winds(mph)'].dropna()

    if len(winds) == 0:
        print("Error: No valid data in 'Max. sustained winds(mph)' column.")
        return

    # Compute the observed frequencies
    observed_freqs = winds.value_counts().sort_index()

    # Determine the mean of the observed data
    mean_winds = np.mean(winds)

    # Compute the range of k values from 0 to the max observed value
    k_values = np.arange(0, observed_freqs.index.max() + 1)

    # Compute the expected frequencies under a Poisson distribution
    expected_freqs = poisson.pmf(k_values, mu=mean_winds) * len(winds)

    # Align observed frequencies with expected frequencies
    observed_freqs = observed_freqs.reindex(k_values, fill_value=0)

    # Check for small expected frequencies and combine categories
    min_expected_freq = 5
    low_freq_indices = expected_freqs < min_expected_freq
    if any(low_freq_indices):
        print("Warning: Some expected frequencies are below the threshold. Combining categories.")

        # Combine low-frequency categories
        combined_k_values = k_values[~low_freq_indices]
        combined_observed_freqs = observed_freqs[~low_freq_indices]
        combined_expected_freqs = expected_freqs[~low_freq_indices]

        # Add a combined category for low expected frequencies
        combined_observed_freqs = np.append(combined_observed_freqs, observed_freqs[low_freq_indices].sum())
        combined_expected_freqs = np.append(combined_expected_freqs, expected_freqs[low_freq_indices].sum())

        observed_freqs = combined_observed_freqs
        expected_freqs = combined_expected_freqs
    else:
        observed_freqs = observed_freqs
        expected_freqs = expected_freqs

    if len(observed_freqs) == 0:
        print("Error: No valid data for the Chi-square test after filtering.")
        return

    # Perform Chi-square goodness-of-fit test
    try:
        chi2_stat, p_value = chisquare(f_obs=observed_freqs, f_exp=expected_freqs)
    except ValueError as e:
        print(f"Error performing Chi-square test: {e}")
        return

    print(f"Chi-square statistic: {chi2_stat:.4f}")
    print(f"P-value: {p_value:.4f}")

    if p_value < alpha:
        print(f"The data does not follow a Poisson distribution at the {alpha*100}% level of significance.")
    else:
        print(f"The data follows a Poisson distribution at the {alpha*100}% level of significance.")

def plot_observed_vs_expected(df):
    winds = df['Max. sustained winds(mph)'].dropna()
    mean_winds = np.mean(winds)
    k_values = np.arange(max(winds)+1)

    observed_freqs = winds.value_counts().reindex(k_values, fill_value=0)
    expected_freqs = poisson.pmf(k_values, mu=mean_winds) * len(winds)

    plt.figure(figsize=(10, 6))
    plt.bar(k_values - 0.2, observed_freqs, width=0.4, label='Observed Frequencies', align='center')
    plt.bar(k_values + 0.2, expected_freqs, width=0.4, label='Expected Frequencies', align='center')

    plt.xlabel('Max. Sustained Winds (mph)')
    plt.ylabel('Frequency')
    plt.title('Observed vs Expected Frequencies')
    plt.legend()
    plt.grid(True)
    plt.show()

# Load data and run the test
df = load_data_from_csv(filepath)
test_poisson_distribution(df)

# Plot the observed vs expected frequencies
plot_observed_vs_expected(df)

print("Poisson distribution test completed.")
