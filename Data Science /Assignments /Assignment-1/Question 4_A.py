import pandas as pd
alpha = 0.01
path = "/content/Assignment-1 - Hurricane.csv"
critical_values = {
        0.001: 3.291,  
        0.01: 2.626,   
        0.025: 2.262,  
        0.05: 2.085,   
        0.10: 1.833,  
        0.20: 1.645    
    }

x = df['Max. sustained winds(mph)']
y = df['Minimum pressure(mbar)']

def load_data_from_csv(filepath):
    return pd.read_csv(filepath)

def compute_pearson_correlation(x, y):
    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    denominator_x = sum((xi - mean_x) ** 2 for xi in x)
    denominator_y = sum((yi - mean_y) ** 2 for yi in y)
    denominator = (denominator_x * denominator_y) ** 0.5
    correlation = numerator / denominator
    return correlation

def compute_t_statistics(r, n):
    result =  r * ((n - 2) / (1 - r ** 2)) ** 0.5
    return result

def get_critical_t_value(alpha, df):
    return critical_values.get(alpha, 1.96)

def printer():
    print("Hypotheses:")
    print("Null Hypothesis (H0): There is no correlation between 'Max. sustained winds (mph)' and 'Minimum pressure (mbar)'. (ρ = 0)")
    print("Alternative Hypothesis (H1): There is a correlation between 'Max. sustained winds (mph)' and 'Minimum pressure (mbar)'. (ρ ≠ 0)")

def run_correlation_t_test(df):
    n = len(x)
    r = compute_pearson_correlation(x, y)
    t_statistic = compute_t_statistics(r, n)
    critical_t = get_critical_t_value(alpha, n - 2)
    printer()
    print(f'\nPearson correlation coefficient: {r:.8f}')
    print(f'T-statistic: {t_statistic:.8f}')
    print(f'Critical t-value for 1% significance level: {critical_t:.8f}')

    if abs(t_statistic) > critical_t:
        print("The correlation is statistically significant at the 1% level.")
    else:
        print("The correlation is not statistically significant at the 1% level.")

def main():
    df = load_data_from_csv(path)
    run_correlation_t_test(df)
main()
print("This is the part A of the question 4")
