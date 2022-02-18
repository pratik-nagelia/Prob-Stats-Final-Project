import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
from scipy import stats

df = pd.read_csv("a3_q7.csv")
df.columns = ["x","y"]
D = df['y'].values


def normal_kde(x,D,h):
  numerator = np.exp(-(np.square((x-D)/h)/2))
  denominator = np.sqrt(2 * np.pi)
  result = np.sum(numerator/denominator)
  return result/(len(D)*h) 


def uniform_kde(x, D, h):
  pdfs = []
  for i in D:
    temp = (x -i)/h
    if x>=-1 and x<=1:
      pdfs.append(1/2)
    else:
      pdfs.append(0)


  K = np.sum(pdfs)
  K = K/(len(D)*h)
  return K


def triangular_kde(x, D, h):
  pdfs = []
  for i in D:
    temp = (x -i)/h
    if np.abs(temp)<=1:
      pdfs.append(1 - np.abs(temp))
    else:
      pdfs.append(0)


  K = np.sum(pdfs)
  K = K/(len(D)*h)
  return K 



def draw_normal_kde():
	xs = np.linspace(0,1,100)
	pdf = stats.norm.pdf(xs, 0.5,0.1)

	bestMSE = np.inf
	bestH = 0
	for h in [0.0001,  0.0005,  0.001,  0.005,  0.05]:
	  kdes=[]
	  for i in xs:
	    K =normal_kde(i,D,h=h)
	    kdes.append(K)


	  sum = 0
	  count=0
	  for kde in kdes:
	    sum = sum + kde
	    count = count + 1

	  sample_mean = sum/count

	  sum=0
	  for kde in kdes:
	    sum = sum + np.square(kde - sample_mean)

	  sample_variance = sum/count

	  percent_variance = (sample_variance - np.var(pdf))*100/np.var(pdf)
	  trueMean = np.mean(pdf)
	  percent_mean = (sample_mean - trueMean) * 100 / trueMean

	  mse = np.square(sample_mean - trueMean) + sample_variance
	  if mse < bestMSE:
		  bestMSE = mse
		  bestH = h

	  print()
	  plt.figure(figsize=(6, 4))
	  plt.plot(xs, pdf, label="PDF")
	  plt.plot(xs, kdes, label="KDE")

	  plt.legend()
	  plt.title("Normal KDE h: "+ str(h))
	  plt.show()
	  print ("******************************************************************")
	  print("Normal KDE")
	  print("H : " ,h)
	  print("Sample Mean: "+str(sample_mean)+" Sample Variance: "+str(sample_variance))
	  print("True Mean: " + str(trueMean) + " True Variance: " + str(np.var(pdf)))
	  print("Percentage Deviation from Mean: "+str(percent_mean))
	  print("Percentage Deviation from Variance: "+str(percent_variance))

	print("Best MSE for Normal KDE =", bestMSE, " with H = ", bestH)


def draw_uniform_kde():
	xs = np.linspace(0,1,100)
	pdf = stats.norm.pdf(xs, 0.5,1)

	bestMSE = np.inf
	bestH = 0
	for h in [0.0001,  0.0005,  0.001,  0.005,  0.05]:
	  kdes=[]
	  for i in xs:
	    K =uniform_kde(i,D,h=h)
	    kdes.append(K)

	  sum = 0
	  count=0
	  for kde in kdes:
	    sum = sum + kde
	    count = count + 1

	  sample_mean = sum/count

	  sum=0
	  for kde in kdes:
	    sum = sum + np.square(kde - sample_mean)

	  sample_variance = sum/count

	  percent_variance = (sample_variance - np.var(pdf))*100/np.var(pdf)
	  trueMean = np.mean(pdf)
	  percent_mean = (sample_mean - trueMean) * 100 / trueMean

	  mse = np.square(sample_mean - trueMean) + sample_variance
	  if mse < bestMSE:
		  bestMSE = mse
		  bestH = h
	  print()
	  plt.figure(figsize=(6, 4))
	  plt.plot(xs, pdf, label="PDF")
	  plt.plot(xs, kdes, label="KDE")

	  plt.legend()
	  plt.title("Uniform KDE h: "+ str(h))
	  plt.show()
	  print("******************************************************************")
	  print("Uniform KDE")
	  print("H : ", h)
	  print("Sample Mean: "+str(sample_mean)+" Sample Variance: "+str(sample_variance))
	  print("True Mean: " + str(trueMean) + " True Variance: " + str(np.var(pdf)))
	  print("Percentage Deviation from Mean: "+str(percent_mean))
	  print("Percentage Deviation from Variance: "+str(percent_variance))

	print("Best MSE for Uniform KDE =", bestMSE, " with H = ", bestH)


def draw_triangular_kde():
	xs = np.linspace(0,1,100)
	pdf = stats.norm.pdf(xs, 0.5,0.1)

	bestMSE = np.inf
	bestH = 0
	for h in [0.0001,  0.0005,  0.001,  0.005,  0.05]:
	  kdes=[]
	  for i in pdf:
	    K =triangular_kde(i,D,h=h)
	    kdes.append(K)


	  sum = 0
	  count=0
	  for kde in kdes:
	    sum = sum + kde
	    count = count + 1

	  sample_mean = sum/count

	  sum=0
	  for kde in kdes:
	    sum = sum + np.square(kde - sample_mean)

	  sample_variance = sum/count

	  trueMean = np.mean(pdf)
	  percent_variance = (sample_variance - np.var(pdf))*100/np.var(pdf)
	  percent_mean = (sample_mean - trueMean) * 100 / trueMean

	  mse = np.square(sample_mean - trueMean) + sample_variance
	  if mse < bestMSE:
		  bestMSE = mse
		  bestH = h
	  print()
	  plt.figure(figsize=(6, 4))
	  plt.plot(xs, pdf, label="PDF")
	  plt.plot(xs, kdes, label="KDE")
	  plt.legend()
	  plt.title("Triangular KDE h: "+ str(h))
	  plt.show()
	  print("******************************************************************")
	  print("Triangular KDE")
	  print("H : ", h)
	  print("Sample Mean: "+str(sample_mean)+" Sample Variance: "+str(sample_variance))
	  print("True Mean: " + str(trueMean) + " True Variance: " + str(np.var(pdf)))
	  print("Percentage Deviation from Mean: "+str(percent_mean))
	  print("Percentage Deviation from Variance: "+str(percent_variance))

	print("Best MSE for Triangular KDE =", bestMSE, " with H = ", bestH)


draw_normal_kde()
draw_uniform_kde()
draw_triangular_kde()




