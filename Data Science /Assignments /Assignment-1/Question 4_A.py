import pandas as pd
from scipy.stats import pearsonr, t
import numpy as np
from scipy.stats import pearsonr, t, norm
import numpy as np
filepath ="/content/Assignment-1 - Hurricane.csv"

def load_data_from_csv(filepath):
    return pd.read_csv(filepath)

def p_compute(x, y):
    return pearsonr(x, y)

def t_statistics(r, n):
    return r * np.sqrt((n - 2) / (1 - r**2))

def get_critical_t_value(alpha, df):
    return t.ppf(1 - alpha / 2, df)

def printer():
    print("Hypotheses:")
    print("Null Hypothesis (H0): There is no correlation between 'Max. sustained winds (mph)' and 'Minimum pressure (mbar)'. (ρ = 0)")
    print("Alternative Hypothesis (H1): There is a correlation between 'Max. sustained winds (mph)' and 'Minimum pressure (mbar)'. (ρ ≠ 0)")

def run_correlation_t_test(df):
    x = df['Max. sustained winds(mph)']
    y = df['Minimum pressure(mbar)']
    
    r, _ = p_compute(x, y)
    n = len(x)
    
    t_statistic = t_statistics(r, n)
    alpha = 0.01
    critical_t = get_critical_t_value(alpha, n - 2)
    
    printer()
    print(f'\nPearson correlation coefficient: {r:.4f}')
    print(f'T-statistic: {t_statistic:.4f}')
    print(f'Critical t-value for 1% significance level: {critical_t:.4f}')
    
    if abs(t_statistic) > critical_t:
        print("The correlation is statistically significant at the 1% level.")
    else:
        print("The correlation is not statistically significant at the 1% level.")
    plot_t_distribution(t_statistic, critical_t, n - 2)

def plot_t_distribution(t_statistic, critical_t, df):
    x = np.linspace(-4, 4, 1000)
    t_dist = t.pdf(x, df)
    norm_dist = norm.pdf(x)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, t_dist, label='T-distribution', color='blue')
    plt.plot(x, norm_dist, label='Standard Normal Distribution', color='green', linestyle='--')
    plt.fill_between(x, t_dist, where=(x >= critical_t) | (x <= -critical_t), color='red', alpha=0.3, label='Critical Region')
    
    plt.axvline(t_statistic, color='black', linestyle='--', label=f'T-Statistic ({t_statistic:.2f})')
    plt.axvline(critical_t, color='red', linestyle=':', label=f'Critical T-value ({critical_t:.2f})')
    plt.axvline(-critical_t, color='red', linestyle=':', label=f'Critical T-value ({-critical_t:.2f})')
    
    plt.title('T-Distribution with T-Statistic and Critical Region')
    plt.xlabel('Value')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid(True)
    plt.show()

df = load_data_from_csv(filepath)
run_correlation_t_test(df)

print("part A is completed Sucessfully ")
