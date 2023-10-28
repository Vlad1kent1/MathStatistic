from func import *
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Завантажимо дані з таблиці
data = pd.read_csv('lab4/data.csv')  # Вам потрібно вказати власний шлях до файлу з даними

Ni = data['Ni']
Pi = data['Pi']
Qi = data['Qi']

# Опишемо вашу однофакторну регресійну модель
x = calc_data(30, Ni, Qi)  # Ознака x
# print(X)
X = sm.add_constant(x)  # Додамо стовпчик константи для b0
Y = calc_data(20, Ni, Pi)  # Ознака y
# print(Y)

# Побудуємо регресійну модель за допомогою методу найменших квадратів
model = build_regresive_model(X, Y)

# Проведемо оцінку значимості параметрів рівняння регресії
regression_eval(model, X)

# Нанесемо на координатну площину кореляційне поле і теоретичну лінію парної регресії
build_graph(model, x, X, Y)

# Розрахуємо та оцінимо прогнозні значення
check_pred(model)
