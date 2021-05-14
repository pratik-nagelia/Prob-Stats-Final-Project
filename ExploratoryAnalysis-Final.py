#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter('ignore')


# # Inferencing Border Crossing Data with Covid Confirmed/ Death Cases for New York State

# # Reading Data

# In[2]:


master_us_confirmed = pd.read_csv('./data/US_confirmed.csv')
master_us_death = pd.read_csv('./data/US_deaths.csv')
master_ny_bordercrossing_data = pd.read_csv('./data/Monthly_Table_Full_Data_data.csv', encoding='UTF-16', sep='\t')


# # Master US Confirmed Cases Dataset

# In[3]:


master_us_confirmed.head()


# # Master US Death Cases Dataset

# In[4]:


master_us_death.head()


# # Master Border Crossing Dataset

# In[5]:


master_ny_bordercrossing_data.head()


# # Extracting New York Data from the US master Data Set for Confirmed/ Death Cases

# In[6]:


ny_confirmed = master_us_confirmed.iloc[34]
ny_confirmed = ny_confirmed.to_frame().reset_index().iloc[1:]
ny_confirmed.columns = ['Date', 'Confirmed Cases']
ny_confirmed['Date'] = pd.to_datetime(ny_confirmed['Date'])
ny_confirmed.head()


# In[7]:


ny_deaths = master_us_death.iloc[34]
ny_deaths = ny_deaths.to_frame().reset_index().iloc[1:]
ny_deaths.columns = ['Date', 'Death Cases']
ny_deaths['Date'] = pd.to_datetime(ny_deaths['Date'])
ny_deaths.head()


# # Extracting Monthly Data Values from the Day Value Dataset

# In[8]:


dates = ['2020-01-01','2020-02-01','2020-03-01','2020-04-01','2020-05-01','2020-06-01','2020-07-01','2020-08-01','2020-09-01','2020-10-01','2020-11-01','2020-12-01','2021-01-01','2021-02-01','2021-03-01','2021-04-01']

ny_confirmed_monthly = ny_confirmed[ny_confirmed.Date.isin(dates)].reset_index(drop=True)
ny_confirmed_monthly.head() 


# In[9]:


ny_confirmed_bymonth = ny_confirmed_monthly.iloc[:,1:].diff(axis=0)
ny_confirmed_bymonth.iloc[:1] = 0.0
ny_confirmed_bymonth['Date'] = ny_confirmed_monthly.Date.values
ny_confirmed_bymonth.head()


# In[10]:


ny_deaths_monthly = ny_deaths[ny_deaths.Date.isin(dates)].reset_index(drop=True)
ny_deaths_monthly.head()


# In[11]:


ny_deaths_bymonth = ny_deaths_monthly.iloc[:,1:].diff(axis=0)
ny_deaths_bymonth.iloc[:1] = 0.0
ny_deaths_bymonth['Date'] = ny_deaths_monthly.Date.values
ny_deaths_bymonth.head()


# # Extracting Total Border Crossing Numbers including Passenger and Non-Passenger Vechiles

# In[12]:


total_vechiles_bydate = master_ny_bordercrossing_data.groupby(['Date']).sum().reset_index()
total_vechiles_bydate.head()


# In[13]:


total_vechiles = total_vechiles_bydate[['Date','Value']]
total_vechiles['Date'] = pd.to_datetime(total_vechiles['Date'])
total_vechiles.columns = ['Date','Total Border Crossing Value']
total_vechiles.head()


# In[14]:


master_ny_bordercrossing_data['Measure'].unique()


# # Extracting Passenger Data

# In[15]:


passenger_list = ['Bus Passengers', 'Pedestrians', 'Train Passengers', 'Personal Vehicle Passengers', 'Personal Vehicles']
non_passenger_list = ['Buses', 'Rail Containers Loaded', 'Trains', 'Truck Containers Empty', 'Truck Containers Loaded', 'Trucks', 'Rail Containers Empty']
passenger_data = master_ny_bordercrossing_data[master_ny_bordercrossing_data.Measure.isin(passenger_list)]
passenger_data_vechiles_bydate = passenger_data.groupby(['Date']).sum().reset_index()

passenger_df = passenger_data_vechiles_bydate[['Date','Value']]
passenger_df['Date'] = pd.to_datetime(passenger_df['Date'])
passenger_df.columns = ['Date','Passenger Border Crossing Value']
passenger_df.head()


# # Extracting Non-Passenger Data

# In[16]:


non_passenger_data = master_ny_bordercrossing_data[master_ny_bordercrossing_data.Measure.isin(non_passenger_list)]
non_passenger_data_vechiles_bydate = non_passenger_data.groupby(['Date']).sum().reset_index()

non_passenger_df = non_passenger_data_vechiles_bydate[['Date','Value']]
non_passenger_df['Date'] = pd.to_datetime(non_passenger_df['Date'])
non_passenger_df.columns = ['Date','Non Passenger Border Crossing Value']
non_passenger_df.head()


# The border crossing data has "'Bus Passengers', 'Buses', 'Pedestrians', 'Rail Containers Loaded','Train Passengers', 'Personal Vehicle Passengers','Personal Vehicles', 'Trains', 'Truck Containers Empty', 'Truck Containers Loaded', 'Trucks', 'Rail Containers Empty'" measures. We have drilled down the data set to passenger and non passenger( goods) border crossing to see how it has impacted both the sectors.

# # Combining all data sets

# In[17]:


master_combined_df = pd.merge(left=passenger_df,right=non_passenger_df, on='Date')
master_combined_df = pd.merge(left=master_combined_df,right=total_vechiles, on='Date')
master_combined_df = pd.merge(left=master_combined_df,right=ny_deaths_bymonth, on='Date')
master_combined_df = pd.merge(left=master_combined_df,right=ny_confirmed_bymonth, on='Date')
master_combined_df[['Confirmed Cases', 'Death Cases','Total Border Crossing Value','Passenger Border Crossing Value','Non Passenger Border Crossing Value']] = master_combined_df[['Confirmed Cases', 'Death Cases','Total Border Crossing Value','Passenger Border Crossing Value','Non Passenger Border Crossing Value']].apply(pd.to_numeric)
master_combined_df.sort_values(by='Date', inplace=True)
master_combined_df = master_combined_df[2:]
master_combined_df.head()


# The above dataset is a combined data set of all the extracted and useful data values which we will be using in our inference.
# We are dealing with 6 permutations of the above columns to see how it is impacting one another. The 6 permutations are as follows:
# 1. Total Border Crossing Value VS Confirmed Cases
# 2. Total Border Crossing Value VS Death Cases
# 3. Passenger Border Crossing Value VS Confirmed Cases
# 4. Passenger Border Crossing Value VS Death Cases
# 5. Non Passenger Border Crossing Value VS Confirmed Cases
# 6. Non Passenger Border Crossing Value VS Death Cases

# In[18]:


master_combined_df.plot(x='Date', figsize = (15,5))


# # Inference 1
# 
# # Pearson Correlation Test

# ## H0 -> Both the data sets are not correlated
# ## H1 -> Both the data sets are correlated

# In[19]:


def corr(x,y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    x_temp = x-x_mean
    y_temp = y-y_mean
    numerator = np.sum(x_temp*y_temp)
    denx = np.sum(np.square(x-x_mean))
    deny = np.sum(np.square(y-y_mean))
    den = np.sqrt(denx*deny)
    return numerator/den


# In[20]:


comparison = []
correlation_val = []


# In[21]:


# Total Border Crossing Value VS Confirmed Cases

comparison.append("Total Border Crossing Value VS Confirmed Cases")
total_border_crossing_data = master_combined_df['Total Border Crossing Value']
confirmed_cases = master_combined_df['Confirmed Cases']
correlation_val.append(corr(total_border_crossing_data, confirmed_cases))


# In[22]:


# Total Border Crossing Value VS Death Cases
comparison.append("Total Border Crossing Value VS Death Cases")
death_cases = master_combined_df['Death Cases']
correlation_val.append(corr(total_border_crossing_data, death_cases))


# In[23]:


# Passenger Border Crossing Value VS Confirmed Cases
comparison.append("Passenger Border Crossing Value VS Confirmed Cases")
passenger_border_crossing_data = master_combined_df['Passenger Border Crossing Value']
correlation_val.append(corr(passenger_border_crossing_data, confirmed_cases))


# In[24]:


# Passenger Border Crossing Value VS Death Cases
comparison.append("Passenger Border Crossing Value VS Death Cases")
correlation_val.append(corr(passenger_border_crossing_data, death_cases))


# In[25]:


# Non Passenger Border Crossing Value VS Confirmed Cases
comparison.append("Non Passenger Border Crossing Value VS Confirmed Cases")
non_passenger_border_crossing_data = master_combined_df['Non Passenger Border Crossing Value']
correlation_val.append(corr(non_passenger_border_crossing_data, confirmed_cases))


# In[26]:


# Non Passenger Border Crossing Value VS Death Cases
comparison.append("Non Passenger Border Crossing Value VS Death Cases")
correlation_val.append(corr(non_passenger_border_crossing_data, death_cases))


# In[27]:


finalcomparison = pd.DataFrame({
    'Comparison': comparison,
    'Correlation coeffecient': correlation_val,
})
print("Comparison of Correlation Coeffecient")
print(finalcomparison)


# Here, on a general level, we can see that the values between the border crossing values and the confirmed/ death cases are negatively correlated. To be more precise:
# 
# 1. Total Border Crossing Value VS Confirmed Cases is less than 0.5, hence it is not correlated. But, we can see that the value is close to 0.5 and the data seems somewhat negatively correlated.
# 2. Total Border Crossing Value VS Death Cases is greater than 0.5, hence it is correlated in a negative way. Therefore, we can observe, that as death cases were increasing, the border crossing number decreased.
# 3. Passenger Border Crossing Value VS Confirmed Cases is less than 0.5, hence it is not correlated. But, we can see that the value is close to 0.5 and the data seems somewhat negatively correlated.
# 4. Passenger Border Crossing Value VS Death Cases is less than 0.5, hence it is not correlated. But, we can see that the value is very very close to 0.5 and the data seems somewhat negatively correlated.
# 5. Non Passenger Border Crossing Value VS Confirmed Cases is less than 0.5, hence it is not correlated. But, we can see that the value is close to 0.5 and the data seems somewhat negatively correlated.
# 6. Non Passenger Border Crossing Value VS Death Cases is greater than 0.5, hence it is correlated in a negative way. Therefore, we can observe, that as death cases were increasing, the border crossing number decreased.
# 
# As cases started to increase, people used to migrate less. Also non passenger vechiles had less impact than passenger vechiles due to covid as non passegener vechiles require less human intervention.

# # Inference 2
# 
# # Kolmogorov - Smirnov  - KS TEST

# ## H0 -> Both the data sets follow the same distribution
# 
# ## H1 -> Both the data sets do not follow the same distribution

# In[28]:


def get_xy(x):

  n = len(x)
  x = sorted(x)
  x_cdf = []
  y_cdf = []
  y_curr = 0

  x_cdf.append(0)
  y_cdf.append(0)

  for i in x:
    y_curr += 1/n
    y_cdf.append(y_curr)
    x_cdf.append(i)

  return x_cdf,y_cdf

def draw_ecdf(x1, y1, x2, y2, max_diff, max_ind):
    plt.figure(figsize=(10,5))
    plt.step(x1, y1, where="post", label="CDF-D1")
    plt.step(x2, y2, where="post", label="CDF-D2")
    # plt.xticks(x1 + x2, rotation = 90)
    plt.yticks(np.arange(0, 1.1, 1/10))
    plt.title("Empirical CDF")
    plt.xlabel("Sample Points")
    plt.ylabel("Pr[X<x]")
    plt.scatter([max_ind],[0], color='red', marker='x', s=100, label=f'Max Diff {max_diff} at {max_ind}')
    # plt.scatter(x, [0]*len(x), color='red', marker='x', s=100, label='samples')
    plt.grid(which="both")
    plt.legend()
    plt.show()

def ks_2_sample_test(data1,data2, threshold=0.05, draw=True):
  x1, y1 = get_xy(data1)
  x2, y2 = get_xy(data2)

  n = int(min([max(x1),max(x2)])) +10

  y1_all = []
  temp=0
  for i in np.arange(n):
    ind = np.where(np.array(x1) == i)[0]
    if len(ind)==0:
      y1_all.append(temp)
    else:
      y1_all.append(y1[ind[-1]])
      temp = y1[ind[-1]]

  y2_all = []
  temp=0
  for i in np.arange(n):
    ind = np.where(np.array(x2) == i)[0]
    if len(ind)==0:
      y2_all.append(temp)
    else:
      y2_all.append(y2[ind[-1]])
      temp = y2[ind[-1]]

  diff=[]
  for i in range(n):
    diff.append( np.absolute( y1_all[i] - y2_all[i]  ) )

  max_diff = np.max(diff)

  max_ind = np.argmax(diff)

  if draw:
    draw_ecdf(x1,y1,x2,y2, max_diff, max_ind)

  if max_diff > threshold:
    print(f"Max value = {max_diff} > C: {threshold}, We reject H0")
  else:
    print(f"Max value = {max_diff} <= C: {threshold}, We reject H0")


# In[29]:


ks_2_sample_test(total_border_crossing_data, confirmed_cases)


# In[30]:


ks_2_sample_test(total_border_crossing_data, death_cases)


# In[31]:


ks_2_sample_test(passenger_border_crossing_data, confirmed_cases)


# In[32]:


ks_2_sample_test(passenger_border_crossing_data, death_cases)


# In[33]:


ks_2_sample_test(non_passenger_border_crossing_data, confirmed_cases)


# In[34]:


ks_2_sample_test(non_passenger_border_crossing_data, death_cases)


# For all the cases, we reject H0 i.e. all the combinations of the datasets do not follow the same distribution. Covid cases/deaths and border crossing data are inversely correlated, these data sets are bound not to follow the same distribution.

# # Inference 3
# 
# # Chi  - Square Test

# ## Ho -> Distributions are Independent
# ## H1 -> Distributions are not independent

# In[35]:


def chi_square(data,x,y):
  rows = data.shape[0]
  cols = data.shape[1]
  observed = data[[x,y]]
  observed.loc['Column_Total']= observed.sum(numeric_only=True, axis=0)
  observed.loc[:,'Row_Total'] = observed.sum(numeric_only=True, axis=1)

  expected = pd.DataFrame(columns=[x,y])

  for col in expected.columns:
    for ind in observed.index:
      expected.loc[ind,col] = (observed.loc[ind,'Row_Total'] * observed.loc['Column_Total', col])/observed.loc['Column_Total','Row_Total']
  

  Q_obs = np.sum(np.square(np.array((observed.iloc[:-1,:-1] - expected.iloc[:-1,:])).flatten())/np.array(expected.iloc[:-1,:]).flatten())

  print(f"{x} VS {y} : Q_obs: {Q_obs}")


# In[36]:


chi_square(master_combined_df, 'Total Border Crossing Value', 'Confirmed Cases')
chi_square(master_combined_df, 'Total Border Crossing Value', 'Death Cases')
chi_square(master_combined_df, 'Passenger Border Crossing Value', 'Confirmed Cases')
chi_square(master_combined_df, 'Passenger Border Crossing Value', 'Death Cases')
chi_square(master_combined_df, 'Non Passenger Border Crossing Value', 'Confirmed Cases')
chi_square(master_combined_df, 'Non Passenger Border Crossing Value', 'Death Cases')


# In Chi-Square test, all the above values have a P value less than 0.05. Hence, we reject the hypotheis that the distributions are independent. All the features compared above are dependent.
