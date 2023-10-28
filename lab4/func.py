import random
from constants import *
import scipy.stats as stats
import statsmodels.api as sm
import matplotlib.pyplot as plt

def calc_data(a, Ni, i):
    N = random.uniform(0, 1)
    data = i + (a + Ni) / a * (N - 1/2)
    return list(data)

def build_regresive_model(X, Y):
    model = sm.OLS(Y, X).fit()
    b0, b1 = model.params[0], model.params[1] 
    print("\nОцінка b0:", b0)
    print("Оцінка b1:", b1)
    print("Model view", f"\ny = {b0} + {b1} * x")
    print("\n", model.summary())
    f_statistic = model.fvalue
    p_value = model.f_pvalue
    print("\nF-statistic:", f_statistic)
    print("p-value:", p_value)
    return model

def regression_eval(model, X):
    b0, b1 = model.params[0], model.params[1] 
    std_err = model.bse

    t_stat_b0 = (b0 - 0) / std_err[0]
    p_value_b0 = 2 * (1 - stats.t.cdf(abs(t_stat_b0), len(X) - 2))
    if p_value_b0 < alpha:
        print("\nПараметр b0 є значущим.")
    else:
        print("Параметр b0 не є значущим.")

    t_stat_b1 = (b1 - 0) / std_err[1]
    p_value_b1 = 2 * (1 - stats.t.cdf(abs(t_stat_b1), len(X) - 2))
    if p_value_b1 < alpha:
        print("Параметр b1 є значущим.")
    else:
        print("Параметр b1 не є значущим.")

    conf_int_b0 = model.conf_int(alpha=alpha)[0]
    conf_int_b1 = model.conf_int(alpha=alpha)[1]
    print(f"Інтервал довіри для b0: ({conf_int_b0[0]}, {conf_int_b0[1]})")
    print(f"Інтервал довіри для b1: ({conf_int_b1[0]}, {conf_int_b1[1]})")

    f_statistic = model.fvalue
    p_value_f = model.f_pvalue
    if p_value_f < alpha:
        print("Регресійна модель є адекватною.")
    else:
        print("Регресійна модель не є адекватною.")

def build_graph(model, x, X, Y):
    plt.scatter(x, Y, label="Data Points")
    plt.plot(x, model.predict(X), color='red', label="Regression Line")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()

def check_pred(model):
    x19 = 9.6 + 0.1 * N
    y_pred = model.predict([1, x19])
    print("\nПрогнозне значення для x19:", y_pred[0])
