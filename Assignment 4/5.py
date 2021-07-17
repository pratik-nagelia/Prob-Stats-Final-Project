import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd

accel = pd.read_csv('acceleration_normal.csv')
modelData = pd.read_csv('model_uniform.csv')
mpgData = pd.read_csv('mpg_exponential.csv')


acc = accel.iloc[:,0].values
amean = np.mean(acc)
print ("The MME Mean for Normal Dist Acceleration data : %.3g"% amean)
avar = np.mean(np.square(acc)) - np.square(amean)
print ("The MME Variance for Normal Dist Acceleration data :  %.3g " % avar)


model = modelData.iloc[:,0].values
model_mean = np.mean(model)
model_var = (np.sum(np.square(model-model_mean)))/len(model)
model_std = np.sqrt(model_var)
a = model_mean - (np.sqrt(3)*model_std)
b = model_mean + (np.sqrt(3)*model_std)
print ("The a MME for Uniform Dist Model data :  %.3g " % a)
print ("The b MME for Uniform Dist Model data :  %.3g " % b)


mpg  = mpgData.iloc[:,0].values
mme_mpg = mle_mpg =  len(mpg) / np.sum(mpg)
print ("The MME Lamda for Exp Dist MPG data : %.3g" %  mme_mpg)

print ("---------------------------------------------------")
print ("The Mean MLE  for Normal Dist Acceleration data : %.3g"% amean)
print ("The Variance MLE for Normal Dist Acceleration data :  %.3g " % avar)

print ("The a MLE for Uniform Dist Model data :  %.3g " % np.min(model))
print ("The b MLE for Uniform Dist Model data :  %.3g " % np.max(model))

print ("The Lamda MLE for Exp Dist MPG data : %.3g" %  mle_mpg)


# Output:
# The MME Mean for Normal Dist Acceleration data : 15.6
# The MME Variance for Normal Dist Acceleration data :  7.57
# The a MME for Uniform Dist Model data :  69.6
# The b MME for Uniform Dist Model data :  82.4
# The MME Lamda for Exp Dist MPG data : 0.0425
# ---------------------------------------------------
# The Mean MLE  for Normal Dist Acceleration data : 15.6
# The Variance MLE for Normal Dist Acceleration data :  7.57
# The a MLE for Uniform Dist Model data :  70
# The b MLE for Uniform Dist Model data :  82
# The Lamda MLE for Exp Dist MPG data : 0.0425