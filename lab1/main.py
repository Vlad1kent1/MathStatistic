from func import *

print(random_variable())

data = collect_data()
print(data)

# num_intervals = 10
# intervals, numbers_count = analyze_intervals(data, num_intervals)

# for i, interval in enumerate(intervals):
#     print(f"Interval {i + 1}: {interval}, Number Count: {numbers_count[i]}")

# plot_histogram(data, num_intervals)

# print("Sample Mean:", sample_mean(data))

# print("Sample Variance:", sample_variance(data))

fit_normal_distribution(data)
