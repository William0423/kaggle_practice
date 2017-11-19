# coding

import numpy as np

food_materials = [[56.0, 0.0, 4.4, 68.0], [
    1.2, 104.0, 52.0, 8.0], [1.8, 135.0, 99.0, 0.9]]

food_materials_m = np.array(food_materials)
print food_materials_m
total_cols = food_materials_m.sum(axis=0)

print food_materials_m / total_cols.reshape(1, 4)  # 1x4 matrix


total_rows = food_materials_m.sum(axis=1)

print total_rows.transpose(3, 0, 1)
print total_rows.reshape(3, 1)
print food_materials_m / total_rows.reshape(3, 1)
