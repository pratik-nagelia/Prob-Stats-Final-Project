import numpy as np
import pandas as pd

df = pd.read_csv('q8_a.csv')
data = df.iloc[:,1].values
n = len(data)
true_mean = 0.5
print("Hypothesis H0 is that Theta0 is equal to 0.5")

sample_mean = np.sum(data)/n
corrected_variance = (np.sum(np.square(data-sample_mean)))/(n-1)
std_error_hat = np.sqrt(corrected_variance/n)
w = np.abs((sample_mean - true_mean) / std_error_hat)
zalphaby2 = 2.33


print("|w| = ", w)
print("Z alpha/2 = ", zalphaby2)

if(w <= zalphaby2):
    print("Hypothesis is accepted with true mean = ", true_mean)
else:
    print("Hypothesis is rejected")

# Output :
# Hypothesis H0 is that Theta0 is equal to 0.5
# |w| =  12.547674743376623
# Z alpha/2 =  2.33
# Hypothesis is rejected