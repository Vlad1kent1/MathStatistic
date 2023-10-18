import random
import numpy as np
from scipy import stats
from scipy.stats import norm
import matplotlib.pyplot as plt

def random_variable(s, a):
    r = sum([random.uniform(0,1) for _ in range(12)])
    result = (r - 6) * s + a
    return result

def build_histogram(data,):
    hist, bin_edges = np.histogram(data, bins=10, density=True)
    
    cumulative_hist = np.cumsum(hist) / 0.4
    
    plt.bar(bin_edges[1:], cumulative_hist, width=bin_edges[1] - bin_edges[0], edgecolor='black', alpha=0.5, color='blue')
    
    plt.xlabel('Значення X')
    plt.ylabel('Накопичена відносна частота')
    plt.title('Гістограма накопичених відносних частот')
    
    plt.grid(True)
    plt.show()

def build_cumulative_histogram(data):
    hist, bin_edges = np.histogram(data, bins=10, density=True)
    
    cumulative_hist = np.cumsum(hist) / 0.4
    
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

    return p

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
    print("Математичне сподівання випадкової велечини:", average)
    return average

def sample_variable(data, average, n):
    variance = np.sum((data - average)**2) / n
    print("Дисперсія випадкової велечини:", variance)
    return variance

def check_fit(data):
    num_bins = 'auto'

    observed_frequencies, bins = np.histogram(data, bins=num_bins, density=False)

    expected_density, _ = np.histogram(data, bins=bins, density=True)
    bin_widths = np.diff(bins)
    expected_frequencies = expected_density * bin_widths * len(data)

    expected_frequencies[
        expected_frequencies == 0] = 0.1

    chi_square_stat, p_value = stats.chisquare(f_obs=observed_frequencies, f_exp=expected_frequencies)

    print("Chi-square Statistic:", chi_square_stat)
    print("P-value:", p_value)

    if p_value > 0.05:
        print('Не відхиляємо нульову гіпотезу, дані відповідають нормальному розподілу')
    else:
        print('Відхиляємо нульову гіпотезу, дані не відповідають нормальному розподілу')
