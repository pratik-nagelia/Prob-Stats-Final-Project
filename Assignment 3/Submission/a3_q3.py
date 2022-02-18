import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd


#to get x,y for ecdf
def get_xy(x):

  n = len(x)
  x = sorted(x)
  x_ecdf = []
  y_ecdf = []
  y = 0

  x_ecdf.append(0)
  y_ecdf.append(0)

  for i in x:
    y += 1/n   #add 1/n everytime an element in present in x
    y_ecdf.append(y)
    x_ecdf.append(i)

  return x_ecdf,y_ecdf


#to draw ecdf 3a
def draw_ecdf(x, y):
    plt.figure(figsize=(20,10))
    plt.step(x, y, where="post", label="CDF")
    plt.xticks(x, rotation = 90)
    plt.yticks(np.arange(0, 1, 1/10))
    plt.title("Empirical CDF")
    plt.xlabel("Sample Points")
    plt.ylabel("Pr[X<=x]")
    plt.legend(loc="upper left")
    plt.title('eCDF with %d samples. Sample mean = %.2f.' % (len(x)-1, np.mean(x[1:])), fontsize=18)
    plt.scatter(x, [0]*len(x), color='red', marker='x', s=100, label='samples')
    plt.grid(which="both")
    plt.show()


#3b
for i in [10,100,1000]:
  x = np.random.randint(1,100,i)
  x_cdf, y_cdf = get_xy(x)
  draw_ecdf(x_cdf,y_cdf)


#compute x,y from 2d array
def get_xy2d(arr2d, rows):
  ys = np.zeros(shape=(rows,100))
  count=0
  for arr in arr2d:
    x,y = get_xy(arr)

    for j in range(len(x)):
      ys[count][x[j]] = y[j] #arrange y in 100 bins according to x


    prev = 0
    for j in range(100):
      if ys[count][j] == 0:
        ys[count][j] = prev    #fill the 0s with the previous number
      else:
        prev = ys[count][j]

  
    count=count+1


  ys = np.mean(ys, axis=0)
  x = [i for i in range(0, 100)]
  x = np.array(x)
  
  return x, ys


#to draw ecdf 3c
def draw_ecdf(x, y, m):
    plt.figure(figsize=(20,10))
    plt.step(x, y, where="post", label="CDF")
    plt.xticks(x, rotation = 90)
    plt.yticks(np.arange(0, 1, 1/10))
    plt.title("Empirical CDF")
    plt.xlabel("Sample Points")
    plt.ylabel("Pr[X<=x]")
    plt.legend(loc="upper left")
    plt.title('eCDF with ( %d , 10 ) samples. Sample mean = %.2f.' % (m, np.mean(x)), fontsize=18)
    plt.scatter(x, [0]*len(x), color='red', marker='x', s=100, label='samples')
    plt.grid(which="both")
    plt.show()


#3d
for m in [10,100,1000]:
  arr2d = np.random.randint(1, 100, size = (m,10))
  x,y = get_xy2d(arr2d, m)
  draw_ecdf(x,y,m)





#to draw ecdf and Normal CI
def draw_ecdf_CI(x, y, CI0, CI1):
    plt.figure(figsize=(20,10))
    plt.step(x, y, where="post", label="CDF")
    plt.step(x, CI0, where="post", label="Normal CI (Lower)")
    plt.step(x, CI1, where="post", label="Normal CI (Upper)")
    plt.xticks(np.arange(np.min(x), np.max(x), 0.3), rotation = 90)
    plt.yticks(np.arange(0, 1, 1/10))
    plt.title("Empirical CDF")
    plt.xlabel("Sample Points")
    plt.ylabel("Pr[X<=x]")
    plt.legend(loc="upper left")
    plt.title('eCDF with %d samples. Sample mean = %.2f.' % (len(x)-1, np.mean(x[1:])), fontsize=18)
    plt.scatter(x, [0]*len(x), color='red', marker='x', s=10, label='samples')
    plt.grid(axis = "both")
    plt.show()


df = pd.read_csv("a3_q3.csv", header=None)
x=df.values.flatten()

print(len(x))

x_cdf, y_cdf = get_xy(x)
y_var = np.var(y_cdf)
n = len(x)
CI0 = y_cdf - (1.96*np.sqrt(y_var/n))
CI1 = y_cdf + (1.96*np.sqrt(y_var/n))

#3 e
draw_ecdf_CI(x_cdf,y_cdf, CI0,CI1)



#to draw ecdf and Normal and DKW CI
def draw_ecdf_N_and_DKW(x, y, CI0, CI1, DKW0, DKW1):
    plt.figure(figsize=(20,10))
    plt.step(x, y, where="post", label="CDF")
    plt.step(x, CI0, where="post", label="Normal CI (Lower)")
    plt.step(x, CI1, where="post", label="Normal CI (Upper)")
    plt.step(x, DKW0, where="post", label="DKW CI (Lower)")
    plt.step(x, DKW1, where="post", label="DKW CI (Upper)")
    plt.xticks(np.arange(np.min(x), np.max(x), 0.3), rotation = 90)
    plt.yticks(np.arange(0, 1, 1/10))
    plt.title("Empirical CDF")
    plt.xlabel("Sample Points")
    plt.ylabel("Pr[X<=x]")
    plt.legend(loc="upper left")
    plt.title('eCDF with %d samples. Sample mean = %.2f.' % (len(x)-1, np.mean(x[1:])), fontsize=18)
    plt.scatter(x, [0]*len(x), color='red', marker='x', s=10, label='samples')
    plt.grid(axis = "both")
    plt.show()



x_cdf, y_cdf = get_xy(x)
y_var = np.var(y_cdf)
n = len(x)
CI0 = y_cdf - (1.96*np.sqrt(y_var/n))
CI1 = y_cdf + (1.96*np.sqrt(y_var/n))
DKW_CI0 = y_cdf - np.sqrt(np.log(2/0.05) / (2*n) )
DKW_CI1 = y_cdf + np.sqrt(np.log(2/0.05) / (2*n) )


#3 f
draw_ecdf_N_and_DKW(x_cdf,y_cdf, CI0,CI1, DKW_CI0, DKW_CI1)





