import pandas as pd
import math
import matplotlib.pyplot as plt

filepath = "/content/Assignment-1 - Hurricane.csv"

def load_data_from_csv(filepath):
    return pd.read_csv(filepath)

POISSON_CONSTANT = 2.718281828459045
CHISQUARE_CRITICAL_VALUES = {
    1: {0.10: 2.706, 0.05: 3.841, 0.01: 6.635},
    2: {0.10: 4.605, 0.05: 5.991, 0.01: 9.210},
    3: {0.10: 6.251, 0.05: 7.815, 0.01: 11.345},
    4: {0.10: 7.779, 0.05: 9.488, 0.01: 13.277},
    5: {0.10: 9.236, 0.05: 11.070, 0.01: 15.086},
    6: {0.10: 10.645, 0.05: 12.592, 0.01: 16.812},
    7: {0.10: 12.017, 0.05: 14.067, 0.01: 19.266},
    8: {0.10: 13.362, 0.05: 15.507, 0.01: 21.955},
    9: {0.10: 14.684, 0.05: 16.919, 0.01: 24.331},
    10: {0.10: 15.987, 0.05: 18.307, 0.01: 26.758},
}

def compute_poisson_prob(x, mean):
    if x < 0:
        return 0
    try:
        numerator = (mean ** x) * (POISSON_CONSTANT ** -mean)
        denominator = math.factorial(x)
        return numerator / denominator
    except OverflowError:
        return float('inf')

def chi_square_test(observed, expected):
    chi_square_stat = 0
    for o, e in zip(observed, expected):
        if e > 0:  # Avoid division by zero
            chi_square_stat += (o - e) ** 2 / e
    return chi_square_stat

def get_chi_square_critical_value(alpha, df):
    return CHISQUARE_CRITICAL_VALUES.get(df, {}).get(alpha, 3.841)

def plot_poisson_distribution(mean):
    x = list(range(0, 20))
    y = [compute_poisson_prob(i, mean) for i in x]

    plt.figure(figsize=(10, 6))
    plt.stem(x, y, basefmt=" ", linefmt="b-", markerfmt="bo", label='Poisson Distribution')
    plt.title(f'Poisson Distribution (mean = {mean:.8f})')
    plt.xlabel('Number of Events')
    plt.ylabel('Probability')
    plt.grid(True)
    plt.legend()
    plt.show()

def compute_mean(values):
    return sum(values) / len(values)

def compute_standard_deviation(values, mean):
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return math.sqrt(variance)

def compute_z_scores(values, mean, std_dev):
    return [(x - mean) / std_dev for x in values]

def remove_outliers(df, column, threshold=3.0):
    values = df[column].tolist()
    print(f"Original Data:\n{values}\n")
    
    mean = compute_mean(values)
    std_dev = compute_standard_deviation(values, mean)
    print(f"Mean: {mean:.2f}")
    print(f"Standard Deviation: {std_dev:.2f}")

    z_scores = compute_z_scores(values, mean, std_dev)
    print(f"Z-Scores:\n{z_scores}\n")
    
    df_filtered = df[abs(pd.Series(z_scores)) < threshold]
    filtered_values = df_filtered[column].tolist()
    print(f"Filtered Data (without outliers):\n{filtered_values}\n")
    
    return df_filtered

def clean_frequencies(observed, expected):
    clean_observed = [o for o in observed if not (math.isnan(o) or math.isinf(o))]
    clean_expected = [e for e in expected if not (math.isnan(e) or math.isinf(e))]
    return clean_observed, clean_expected

def test_poisson_distribution(df):
    print("Initial Data:\n", df.head(), "\n")
    
    df_filtered = remove_outliers(df, 'Max. sustained winds(mph)')
    wind_speeds = df_filtered['Max. sustained winds(mph)']
    
    print(f"Filtered Wind Speeds:\n{wind_speeds.tolist()}\n")
    
    mean_wind_speed = compute_mean(wind_speeds)
    print(f"Mean Wind Speed: {mean_wind_speed:.2f}")
    
    max_wind_speed = int(max(wind_speeds))
    bins = list(range(0, max_wind_speed + 1))
    print(f"Bins:\n{bins}\n")
    
    observed_frequencies = [sum(1 for speed in wind_speeds if b <= speed < b + 1) for b in bins]
    print(f"Observed Frequencies:\n{observed_frequencies}\n")
    
    non_zero_indices = [i for i, o in enumerate(observed_frequencies) if o > 0]
    filtered_bins = [bins[i] for i in non_zero_indices]
    filtered_observed_frequencies = [observed_frequencies[i] for i in non_zero_indices]
    print(f"Filtered Bins:\n{filtered_bins}")
    print(f"Filtered Observed Frequencies:\n{filtered_observed_frequencies}\n")
    
    expected_frequencies = [len(wind_speeds) * compute_poisson_prob(b, mean_wind_speed) for b in filtered_bins]
    print(f"Expected Frequencies:\n{expected_frequencies}\n")
    
    filtered_observed, filtered_expected = clean_frequencies(filtered_observed_frequencies, expected_frequencies)
    
    print(f"Cleaned Observed Frequencies:\n{filtered_observed}")
    print(f"Cleaned Expected Frequencies:\n{filtered_expected}\n")
    
    if not filtered_observed or not filtered_expected:
        print("No valid data for Chi-Square test.")
        return

    chi_square_statistic = chi_square_test(filtered_observed, filtered_expected)
    df = len(filtered_bins) - 1
    critical_value = get_chi_square_critical_value(0.10, df)

    print(f'\nChi-Square Statistic: {chi_square_statistic:.8f}')
    print(f'Critical Chi-Square Value at 10% significance level: {critical_value:.8f}')

    if chi_square_statistic > critical_value:
        print("The data does not follow a Poisson distribution at the 10% significance level.")
    else:
        print("The data follows a Poisson distribution at the 10% significance level.")
    
    plot_poisson_distribution(mean_wind_speed)

def main():
    df = load_data_from_csv(filepath)
    test_poisson_distribution(df)
    
main()
print("this is the last part ofthe q4 ")
