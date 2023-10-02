import math
import random
import numpy as np
from scipy.stats import t
import matplotlib.pyplot as plt

def random_variable(s, a):
    r = sum([random.uniform(0,1) for _ in range(12)])
    result = (r - 6) * s + a

    return result

def correlation_field(xdata, ydata):
    correlation_field = np.correlate(xdata, ydata, mode='full')
    plt.scatter(xdata, ydata, alpha=0.5)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Кореляційне поле")
    plt.grid(True)
    plt.show()

def selective_average(s, data):
    average = np.mean(data)
    print("Sample mean of variable", s, average)
    return average

def sample_variable(s, data, average, n):
    variance = np.sum((data - average)**2) / n
    print("Sample variance of variable", s, variance)
    return variance

def coefficient_correlation(xdata, ydata, xaverage, yaverage, n, sx, sy):
    correlation = np.sum((xdata - xaverage) * (ydata - yaverage)) / (n * sx * sy)
    print("Sample correlation coefficient", correlation)
    return correlation

def check_theorem(r, n, a):
    T = (r * math.sqrt(n - 2)) / math.sqrt(1 - r ** 2)
    tkp = t.ppf(1 - a / 2, n - 2)

    if np.abs(T) < tkp:
        print("Нема підстав відкидати нульову гіпотезу")
    else:
        print("Нульову гіпотезу відкидаєм")
