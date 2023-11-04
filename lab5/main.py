from func import *
import pandas as pd

data = pd.read_csv('lab5/data.csv')

y = calc_data(1.5, data['y'])
x1 = calc_data(2.5, data['x1'])
x2 = calc_data(10, data['x2'])
print("y:", y, "\nx1:", x1, "\nx2:", x2)

# 1. Аналіз залежності між факторами
depend_analysis(data)

# 2. Побудувати регресійну модель
model = build_regresive_model(x1, x2, y)

# 3. Перевірити модель на адекватність за F-критерієм Фішера
check_Fisher(model)

# 4. Оцінка значимості параметрів та інтервали довіри
estimation_conf_intervals(model)

# 5. Довірчий інтервал для окремого значення залежної змінної
y_pred_ci = conf_interval(model)

# 6. Тлумачення параметрів b0, b1, b2
interpretation_params(model, y_pred_ci)
