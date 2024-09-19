import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

def load_and_print_csv(filepath):
    global df
    try:
        df = pd.read_csv(filepath)
        print("Data loaded successfully.")
        print("First few rows of the data:")
        print(df.head())
        print("\nColumn names:")
        print(df.columns.tolist())
        print("\nDataFrame information:")
        print(df.info())
    except FileNotFoundError:
        print(f"Error: The file at path '{filepath}' was not found.")
    except pd.errors.EmptyDataError:
        print("Error: The file is empty.")
    except pd.errors.ParserError:
        print("Error: There was a parsing error with the file.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def calculate_mean(values):
    return sum(values) / len(values)

def calculate_ss_between(groups, overall_mean):
    ss_between = 0
    for group in groups:
        group_mean = calculate_mean(group)
        ss_between += len(group) * (group_mean - overall_mean) ** 2
    return ss_between

def calculate_ss_within(groups):
    ss_within = 0
    for group in groups:
        group_mean = calculate_mean(group)
        ss_within += sum((x - group_mean) ** 2 for x in group)
    return ss_within

def calculate_degrees_of_freedom(k, n):
    df_between = k - 1
    df_within = n - k
    return df_between, df_within

def calculate_mean_squares(ss_between, ss_within, df_between, df_within):
    ms_between = ss_between / df_between
    ms_within = ss_within / df_within
    return ms_between, ms_within

def calculate_f_statistic(ms_between, ms_within):
    return ms_between / ms_within

def get_critical_f_value(alpha, df_between, df_within):
    return stats.f.ppf(1 - alpha, df_between, df_within)

def compute_anova(groups, alpha=0.05):
    global overall_mean, ss_between, ss_within, ms_between, ms_within, f_statistic, critical_value, significant

    k = len(groups)
    n = sum(len(group) for group in groups)
    
    all_values = [item for sublist in groups for item in sublist]
    overall_mean = calculate_mean(all_values)
    
    ss_between = calculate_ss_between(groups, overall_mean)
    ss_within = calculate_ss_within(groups)
    
    df_between, df_within = calculate_degrees_of_freedom(k, n)
    
    ms_between, ms_within = calculate_mean_squares(ss_between, ss_within, df_between, df_within)
    
    f_statistic = calculate_f_statistic(ms_between, ms_within)
    
    critical_value = get_critical_f_value(alpha, df_between, df_within)
    
    significant = f_statistic > critical_value

def prepare_data_for_anova(df):
    df['Month'] = df['Month'].str.split(', ').apply(lambda x: x[0])
    groups = [df[df['Month'] == month]['Max. sustained winds(mph)'].tolist() for month in df['Month'].unique()]
    return groups

def plot_raw_data_distribution(df):
    plt.figure(figsize=(12, 6))
    df['Max. sustained winds(mph)'].hist(bins=20, color='lightblue', edgecolor='black')
    plt.xlabel('Max Sustained Winds (mph)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Max Sustained Winds')
    plt.grid(True)
    plt.show()

def plot_group_means(groups, group_labels):
    means = [calculate_mean(group) for group in groups]
    std_devs = [stats.tstd(group) for group in groups]
    
    plt.figure(figsize=(12, 6))
    plt.bar(group_labels, means, yerr=std_devs, capsize=5, color='lightblue', ecolor='black')
    plt.xlabel('Month')
    plt.ylabel('Mean Max Sustained Winds (mph)')
    plt.title('Mean Max Sustained Winds by Month with Error Bars')
    plt.grid(axis='y')
    plt.show()

def plot_distribution_by_month(df):
    plt.figure(figsize=(12, 6))
    for month in df['Month'].unique():
        subset = df[df['Month'] == month]['Max. sustained winds(mph)']
        plt.hist(subset, bins=15, alpha=0.5, label=month, edgecolor='black')
    plt.xlabel('Max Sustained Winds (mph)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Max Sustained Winds by Month')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()

def main():
    filepath = '/content/Assignment-1 - Hurricane.csv'
    load_and_print_csv(filepath)
    
    groups = prepare_data_for_anova(df)
    group_labels = df['Month'].unique().tolist()
    compute_anova(groups)
    
    print(f'Overall Mean: {overall_mean:.2f}')
    print(f'Sum of Squares Between Groups (SS_between): {ss_between:.2f}')
    print(f'Sum of Squares Within Groups (SS_within): {ss_within:.2f}')
    print(f'Mean Square Between Groups (MS_between): {ms_between:.2f}')
    print(f'Mean Square Within Groups (MS_within): {ms_within:.2f}')
    print(f'F-Statistic: {f_statistic:.2f}')
    print(f'Critical F-value at alpha=0.05: {critical_value:.2f}')
    
    if significant:
        print("The difference between group means is statistically significant.")
    else:
        print("The difference between group means is not statistically significant.")
    
    print("This is the solution for question 4b.")
    
    # Plotting
    plot_raw_data_distribution(df)
    plot_group_means(groups, group_labels)
    plot_distribution_by_month(df)

main()
