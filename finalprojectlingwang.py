# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 15:52:18 2024

@author: mingyue_jiang
"""

import sys
def exit():
    sys.exit()

def skip(k=1):
    print("\n"*k)

import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
    
year = [2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022,2023]

revenue = [24.20, 25.40 ,27.80 ,29.80 ,30.30 ,28.80 ,25.80 ,26.40 ,26.60 ,27.70 ,29.00 ,30.30 ,31.60 ,32.60 ,34.10 ,35.70 ,41.70 ,47.90 ,57.00, 65.40 ,72.80]

dic = {'year':year,'revenue':revenue}
df = pd.DataFrame(dic)
print(df)

y = df[['revenue']]
x = df[['year']]
x = sm.add_constant(x)
results = sm.OLS(y, x).fit()
print(results.summary())
output = results.params
print(output)
print(type(results.params))
c = results.params.tolist()
print(c)

print(f"r squared {results.rsquared}")
print("results.params: the good stuff")
output = results.params
print(output)

const = output[0]
coeff_year = output[1]

df['y_pred'] = const \
    + coeff_year * df['year'] 
print("predicted")
print(df['y_pred'])
revenue_pred = df['y_pred'].values


plt.figure(figsize=(12,8))
plt.plot(year, revenue, marker='o', linestyle='-', color='b') 
 
plt.plot(year, revenue_pred, linestyle='--', color='r')  

plt.title('Sales revenue of the flower market',fontsize=28)  
plt.xlabel('year',fontsize=20)  
plt.ylabel('revenue',fontsize=20) 
plt.xticks( fontsize = 16)
plt.yticks(  fontsize = 16) 
plt.grid(True)  
plt.legend()     
plt.show()  


# Dollar Volume of Holiday Flower sales
skip()
sales = np.array([0.28,0.29,0.24,0.06,0.08,0.04])
mylabels = ["Valentine’s Day","Christmas","Mother’s Day","Easter","Thanksgiving","Father’s Day"]

plt.figure(figsize=(12,8))
plt.pie(sales, labels = mylabels,autopct='%1.0f%%',textprops={'fontsize': 22})
plt.title('Dollar Volume of Holiday Flower sales', fontsize = 30)
plt.show()


# Top five States for Floriculture Sales in 2023
lable1s = ["Califorlia","Florida","Michigan","Texas","New Jersey"]
value1s = [983,1208,695,319,319]

plt.figure(figsize=(10,6)) 
plt.bar(lable1s,value1s,width = 0.5)
plt.title("Top five States for Floriculture Sales in 2023",fontsize=24)
plt.ylabel("sales revenue in millions",fontsize=16)
plt.xticks( fontsize = 14)
plt.yticks( np.arange(0, 1600, 200), fontsize = 14)
plt.tight_layout()
plt.show()




