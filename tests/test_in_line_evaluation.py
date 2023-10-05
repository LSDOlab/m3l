import m3l
import numpy as np


m3l_model = m3l.Model()

var1 = m3l_model.create_input('var_1', val=2)
var2 = m3l_model.create_input('var_2', val=4)
var3 = m3l_model.create_input('var3', val=np.array([np.sqrt(2)/ 2, 0, np.sqrt(2)/2]).reshape(1, 3))
var4 = m3l_model.create_input('var4', val=np.array([0, np.sqrt(2)/ 2, np.sqrt(2)/2]).reshape(1, 3))

test_addition = var1 + var2
test_subtraction = var1 - var2
test_multiplication = var1 * var2
test_division = var1 / var2
test_norm = m3l.norm(var3)
test_cross = m3l.cross(var4, var3, axis=1)
test_vstack = m3l.vstack(var3, var4)

print(test_addition.value)
print(test_subtraction.value)
print(test_multiplication.value)
print(test_division.value)
print(test_norm.value)
print(test_cross.value)
print(test_vstack.value)