from constants import *
from func import *

x = random_variable(sigma, a)

print("X =", x)

data = np.array([random_variable(sigma, a) for _ in range(n)])

build_histogram(data, n)

build_cumulative_histogram(data, n)

average = selective_average(data)
sample_variable(data, average, n)

check_fit(data)
