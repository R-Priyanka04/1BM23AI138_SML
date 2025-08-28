#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np

# Dataset
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

# Prepare data for statsmodels
y = df['shear']
X = df['age']
X_sm = sm.add_constant(X)  # Add intercept term

# Fit model with statsmodels
model = sm.OLS(y, X_sm)
results = model.fit()

print("\nLinear Regression Summary:")
print(results.summary())

intercept = results.params['const']
slope = results.params['age']
print(f"\nIntercept: {intercept}")
print(f"Slope: {slope}")

# Plot
plt.scatter(df['age'], df['shear'], color='blue', label='Data points')
plt.plot(df['age'], results.predict(X_sm), color='red', label='Regression line')
plt.xlabel('Age')
plt.ylabel('Shear')
plt.title('Linear Regression: Shear vs Age')
plt.legend()
plt.grid(True)
plt.show()

# Convert to numpy arrays for gradient descent
X_np = np.c_[np.ones(len(X)), X]  # Add intercept term to X as numpy array
y_np = y.values

# Gradient Descent function
def gradient_descent(X, y, initial_learning_rate=0.01, decay_rate=0.01, n_iterations=1000):
    m = len(y)
    theta = np.random.randn(X.shape[1])
    for iteration in range(n_iterations):
        gradients = (2/m) * X.T.dot(X.dot(theta) - y)
        learning_rate = initial_learning_rate / (1 + decay_rate * iteration)
        theta -= learning_rate * gradients
    return theta

theta_gd = gradient_descent(X_np, y_np)
print("\nGradient Descent:")
print(f"Intercept: {theta_gd[0]}, Slope: {theta_gd[1]}")

def stochastic_gradient_descent(X, y, learning_rate=0.001, n_iterations=100000):
    m = len(y)
    theta = np.array([3000, -0.1])  
    for iteration in range(n_iterations):
        random_index = np.random.randint(m)
        xi = X[random_index:random_index+1]
        yi = y[random_index:random_index+1]
        gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
        theta -= learning_rate * gradients
    return theta

theta_sgd = stochastic_gradient_descent(X_np, y_np)
print("\nStochastic Gradient Descent:")
print(f"Intercept: {theta_sgd[0]}, Slope: {theta_sgd[1]}")


# In[ ]:




