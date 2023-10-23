from constants import *
from func import *

x = random_variable(sigma, a)

print("X =", x)

data = np.array([random_variable(sigma, a) for _ in range(n)])
# print(data)

build_histogram(data, n)

build_cumulative_histogram(data, n)
# print(theoretical_distribution)

# expected_counts = np.array([len(data) / 10] * 10)
# chi_square_test(data, expected_counts, alpha)

average = selective_average(data)
sample_variable(data, average, n)

check_fit(data)
