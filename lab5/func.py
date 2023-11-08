import random
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt

def calc_data(a, i):
    N = random.uniform(0, 1)
    data = i + a * (N - 1/2)
    return data

def find_params(y, X):
    B = np.linalg.inv(X.T @ X) @ (X.T @ y)
    print('B: ', B)
    return B

def find_param_u(y, X, B):
    y_hat = X @ B
    residuals = y - y_hat
    print('Залишки: ', residuals)
    return residuals

def visualization(y, x, s):
    plt.scatter(x, y, label=s, color='blue', marker='o')
    plt.xlabel(s)
    plt.ylabel('y')
    plt.title('Залежність між ' + s + ' та y')
    plt.legend(loc='upper right')
    plt.show()

def depend_analysis(data):
    correlation_matrix = data.corr()
    print("\nКореляційна матриця:")
    print(correlation_matrix)

def build_regresive_model(x1, x2, y):
    X = sm.add_constant(pd.DataFrame({'x1': x1, 'x2': x2}))
    model = sm.OLS(y, X).fit()
    print('\n', model.summary())
    print("\nРегресійна модель:", f"\ny = {model.params['const']} + {model.params['x1']} * x1 + {model.params['x2']} * x2")
    return model

def calculate_F_value(X, Y, B, U):
    n = len(Y)
    m = len(B)

    Y_hat = X @ B
    SSR = ((Y_hat - Y.mean()) ** 2).sum()
    SSE = (U ** 2).sum()
    MSR = SSR / (m - 1)
    MSE = SSE / (n - m)
    F_value = MSR / MSE

    print('\nF-value: ', F_value)
    return F_value

def check_Fisher(F_value):
    alpha = 0.05
    if F_value < alpha:
        print("Модель є адекватною за F-критерієм Фішера")
    else:
        print("Модель не є адекватною за F-критерієм Фішера")

def evaluate_significance(y, X, B, U):
    degrees_of_freedom = len(y) - X.shape[1]
    mse = np.sum(U**2) / degrees_of_freedom
    var_B = mse * np.linalg.inv(X.T @ X)

    se_B = np.sqrt(np.diag(var_B))
    print('Стандартні помилки: ', se_B)
    
    t_stats = B / se_B
    print('t-статистика: ', t_stats)
    
    p_values = [2 * (1 - stats.t.cdf(np.abs(t), degrees_of_freedom)) for t in t_stats]
    print('p-значення: ', p_values)
    
    confidence_interval = [stats.t.interval(0.95, degrees_of_freedom, loc=b, scale=se) for b, se in zip(B, se_B)]
    print('Інтервали довіри: ', confidence_interval)

def predict_confidence_interval(x_new, X, y, B, U, confidence=0.95):
    y_hat = x_new @ B
    degrees_of_freedom = len(y) - len(B)
    mse = np.sum(U**2) / degrees_of_freedom
    se = np.sqrt(mse * (1 + x_new @ np.linalg.inv(X.T @ X) @ x_new.T))
    t_value = stats.t.ppf((1 + confidence) / 2, degrees_of_freedom)
    confidence_interval = (y_hat - t_value * se, y_hat + t_value * se)
    
    return confidence_interval
