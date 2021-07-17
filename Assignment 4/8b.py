import numpy as np
import pandas as pd


df = pd.read_csv('q8_b_X.csv')
x = df.iloc[:, 1].values
df = pd.read_csv('q8_b_Y.csv')
y = df.iloc[:, 1].values
nx = len(x)
ny = len(y)

xs_mean = np.mean(x)
ys_mean = np.mean(y)

print("Hypothesis H0 is mean of X and Y distribution is same")

x_variance = np.sum(np.square(x - xs_mean)) / (nx)
y_variance = np.sum(np.square(y - ys_mean)) / (ny)
delta_0 = 0
delta_hat = xs_mean - ys_mean
se_hat = np.sqrt((x_variance / nx) + (y_variance / ny))
w = np.abs((delta_hat - delta_0) / se_hat)
zalphaby2 = 1.96

print("|w| = ", w)
print("Z alpha/2 = ", zalphaby2)

if (w <= zalphaby2):
    print("Hypothesis is accepted")
else:
    print("Hypothesis is rejected")


# Output :
# Hypothesis H0 is mean of X and Y distribution is same
# |w| =  7.7480225509526015
# Z alpha/2 =  1.96
# Hypothesis is rejected