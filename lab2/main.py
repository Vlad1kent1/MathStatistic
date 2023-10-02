from constants import *
from func import *

x = random_variable(sx, ax)
y = random_variable(sy, ay)

print("X =", x)
print("Y =", y)

xdata = np.array([random_variable(sx, ax) for _ in range(n)])
ydata = np.array([random_variable(sy, ay) for _ in range(n)])

# print("\n", xdata)
# print("\n", ydata)

correlation_field(xdata, ydata)

xaverage = selective_average("X", xdata)
yaverage = selective_average("Y", ydata)

sample_variable("X", xdata, xaverage, n)
sample_variable("Y", ydata, yaverage, n)

cor = coefficient_correlation(xdata, ydata, xaverage, yaverage, n, sx, sy)

check_theorem(cor, n, alpha)
