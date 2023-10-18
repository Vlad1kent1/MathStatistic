import matplotlib.pyplot as plt
import random
import numpy as np
from scipy.stats import norm

def random_variable():
    r = sum([random.uniform(0,1) for _ in range(12)])
    result = (r - 6) * 4.2 + 2

    return result

def collect_data():
    return [random_variable() for _ in range(350)]

def analyze_intervals(data, num_intervals):

    minimum = min(data)
    maximum = max(data)
    interval_width = (maximum - minimum) / num_intervals
    intervals_table = []
    numbers_count_in_intervals = [0] * num_intervals
    
    for i in range(num_intervals):
        interval_start = minimum + i * interval_width
        interval_end = interval_start + interval_width
        interval = (interval_start, interval_end)
        intervals_table.append(interval)
        
        for number in data:
            if interval_start <= number < interval_end:
                numbers_count_in_intervals[i] += 1
                
    return intervals_table, numbers_count_in_intervals

def plot_histogram(data, num_bins):

    plt.hist(data, bins=num_bins, edgecolor='black', alpha=0.7)
    plt.xlabel('Intervals')
    plt.ylabel('Number Count')
    plt.title('Histogram')
    
    minimum = min(data)
    maximum = max(data)
    interval_width = (maximum - minimum) / num_bins
    interval_ranges = [round(minimum + i * interval_width, 2) for i in range(num_bins + 1)]
    plt.xticks(interval_ranges)
    
    plt.grid(True)
    plt.show()

def sample_mean(data):
    if len(data) == 0:
        return None  
    
    total = sum(data)
    mean = total / len(data)
    return mean

def sample_variance(data):
    if len(data) < 2:
        return None
    
    mean = sample_mean(data)
    squared_differences = [(x - mean) ** 2 for x in data]
    variance = sum(squared_differences) / (len(data))
    
    return variance

def fit_normal_distribution(data):
    mu, std = norm.fit(data)

    plt.hist(data, bins=10, density=True, alpha=0.6, color='b', edgecolor='black')

    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    
    plt.plot(x, p, 'k', linewidth=2)
    plt.xlabel('Значення')
    plt.ylabel('Щільність ймовірності')
    plt.title('Апроксимація нормального розподілу')

    plt.show()

    print("Fitted Normal Distribution - Mean:", mu)
    print("Fitted Normal Distribution - Standard Deviation:", std)
