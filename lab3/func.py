import random
import numpy as np
from scipy import stats
from scipy.stats import norm
import matplotlib.pyplot as plt

def random_variable(s, a):
    r = sum([random.uniform(0,1) for _ in range(12)])
    result = (r - 6) * s + a
    return result

def build_histogram(data, n):
    hist, bin_edges = np.histogram(data, bins=10)
    hist = hist /n
    cumulative_hist = np.cumsum(hist) 
    
    plt.bar(bin_edges[1:], cumulative_hist, width=bin_edges[1] - bin_edges[0], edgecolor='black', alpha=0.5, color='blue')
    
    plt.xlabel('Значення X')
    plt.ylabel('Накопичена відносна частота')
    plt.title('Гістограма накопичених відносних частот')
    
    plt.grid(True)
    plt.show()

def build_cumulative_histogram(data, n):
    hist, bin_edges = np.histogram(data, bins=10)
    hist = hist /n
    cumulative_hist = np.cumsum(hist)
    
    plt.bar(bin_edges[1:], cumulative_hist, width=bin_edges[1] - bin_edges[0], edgecolor='black', alpha=0.5, color='blue')
    
    plt.xlabel('Значення X')
    plt.ylabel('Накопичена відносна частота')
    plt.title('Гістограма накопичених відносних частот')
    
    mean, std = norm.fit(data)
    min, max = plt.xlim()
    
    x = np.linspace(min, max, 100)
    p = norm.cdf(x, loc=mean, scale=std)

    plt.plot(x, p, 'r-', linewidth=2, label='Нормальний розподіл')
    
    plt.legend()
    plt.grid(True)
    plt.show()

# def chi_square_test(data, expected_counts, alpha):
#     observed_counts, bin_edges = np.histogram(data, bins=len(expected_counts))
#     dof = len(expected_counts) - 1
#     chi_square_statistic = np.sum((observed_counts - expected_counts)**2 / expected_counts)
#     p_value = 1 - chi2.cdf(chi_square_statistic, dof)
    
#     if p_value < alpha:
#         print("Гіпотезу H0 відхиляють")
#     else:
#         print("Гіпотезу H0 приймають")

def selective_average(data):
    average = np.mean(data)
    print("Математичне сподівання випадкової величини:", average)
    return average

def sample_variable(data, average, n):
    variance = np.sum((data - average)**2) / n
    print("Дисперсія випадкової величини:", variance)
    return variance

def check_fit(data):
    theoretical_distribution = stats.norm
    params = theoretical_distribution.fit(data)

    observed_frequencies, bin_edges = np.histogram(data, bins=10)
    expected_frequencies = []
    for i in range(len(observed_frequencies)):
        cdf_low = theoretical_distribution.cdf(bin_edges[i], *params)
        cdf_high = theoretical_distribution.cdf(bin_edges[i + 1], *params)
        expected_frequency = len(data) * (cdf_high - cdf_low)
        expected_frequencies.append(expected_frequency)

    expected_frequencies = np.array(expected_frequencies)
    expected_frequencies = (expected_frequencies / expected_frequencies.sum()) * observed_frequencies.sum()

    chi2_stat, p_value = stats.chisquare(f_obs=observed_frequencies, f_exp=expected_frequencies)

    print(f"Chi-squared statistic: {chi2_stat}")
    print(f"P-value: {p_value}")
    
    if p_value > 0.05:
        print('Не відхиляємо нульову гіпотезу, дані відповідають нормальному розподілу')
    else:
        print('Відхиляємо нульову гіпотезу, дані не відповідають нормальному розподілу')
