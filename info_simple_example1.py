import numpy as np
import info
import numpy as np

"""
    X has 6 features (columns). Features 5 and 6 are not independent from the rest.
      x1, x2, x3, x4 have random numbers from 0 to 100.
      x5 = x2 + 0.8*r , with r a random number from 0 to 100.
      x6 = x1 + 0.3*x3.
    Y(x1,x2,x3,x4) = 10 sin(pi x1 x2) + 20(x3 - 0.5)^2 + 10x4
"""

# Fijamos el seed
seed = 13
np.random.seed(seed)

# We create a data of 6 columns, 4 columns of random numbers from 0 to 100.
# Then a 5th column where x_5 = x_2 + 0.8*random
# And a 6th column where x_6 = x_1 + 0.3*x_3
data = np.random.rand(6000,4)*100
# We add two "empty" columns to manipulate with.
data = np.c_[data,np.zeros( len(data) )]
data = np.c_[data,np.zeros( len(data) )]
for row in data:
    row[-2] = row[1] + 0.8*np.random.rand()*100
    row[-1] = row[0] + 0.3*row[2]
# Y function is number (20) in page 120 of https://sci-hub.tw/10.1016/S0893-6080(03)00169-2
# Y(X) = 10 sin(pi x1 x2) + 20(x3 - 0.5)^2 + 10x4
Y = []
for row in data:
    Y.append( 10*np.sin(np.pi * row[0] * row[1]) + 20*(row[2] - 0.5)**2 + 10*row[3] )
Y = np.array(Y, np.float64)

# We pass from string to float, and we drop the line naming the features.
data = np.array(data, np.float64)

print(Y)

# info_object = info.Informacion(X=data,last_feature_is_Y=True)
#
# print()
# print("INDEXES OF FEATURES SELECTED:")
# mi_f = info_object.filter_MI()
# print("MI:",mi_f)
# delta_f = info_object.filter_Delta()
# print("Delta:",delta_f)
# mix_f = info_object.filter_MI_Delta()
# print("Mix:",mix_f)
#
# print()
# print("LOGISTIC REGRESSION. Porcentaje bien clasificado:")
# orig = info_object.logistic_regression_lerner(seed=seed)[0]
# print("Original:",orig)
# mi_l = info_object.logistic_regression_lerner(cols_pos= mi_f[0] ,seed=seed)[0]
# print("MI:",mi_l)
# delta_l = info_object.logistic_regression_lerner(cols_pos= delta_f[0] ,seed=seed)[0]
# print("Delta:",delta_l)
# mix_l = info_object.logistic_regression_lerner(cols_pos= mix_f[0] ,seed=seed)[0]
# print("Mix:",mix_l)
