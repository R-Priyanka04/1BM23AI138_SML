#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

data = {
    "shear": [2160.70, 1680.15, 2160.70, 1680.15, 2318.00, 2063.30, 2209.50, 1710.30,
              1786.70, 2577.00, 2359.90, 2258.70, 2167.20, 2401.55, 1781.80, 2338.75,
              1767.30, 2055.50, 2416.40, 2202.50, 2656.20, 1755.70],
    "age": [15.50, 23.75, 8.00, 17.00, 5.50, 19.00, 24.00, 2.50, 7.50, 
            11.00, 13.00, 3.75, 25.00, 9.75, 22.00, 18.00, 6.00, 12.50, 2.00,
            21.50, 14.00, 20.00]
}


df = pd.DataFrame(data)
print("First 5 rows of the dataset:")
print(df.head())

y = df['shear']
X = df['age']
X = sm.add_constant(X) 

linear_regression = sm.OLS(y, X)
fitted_model = linear_regression.fit()


print("\nLinear Regression Summary:")
print(fitted_model.summary())


intercept = fitted_model.params[0]
slope = fitted_model.params[1]
print("\nIntercept:", intercept)
print("Slope:", slope)

plt.scatter(df['age'], df['shear'], color='blue', label='Data points')
plt.plot(df['age'], fitted_model.predict(X), color='red', label='Regression line')
plt.xlabel('Age')
plt.ylabel('Shear')
plt.title('Linear Regression: Shear vs Age')
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:




