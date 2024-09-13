### DEVELOPED BY: G Chethan Kumar
### REGISTER NO: 212222240022
### DATE:

# Ex.No: 02 LINEAR AND POLYNOMIAL TREND ESTIMATION
### AIM:
To Implement Linear and Polynomial Trend Estiamtion Using Python.

### ALGORITHM:

**Step 1:** Import necessary libraries (NumPy, Matplotlib)

**Step 2:** Load the dataset

**Step 3:** Calculate the linear trend values using lLinearRegression Function.

**Step 4:** Calculate the polynomial trend values using PolynomialFeatures Function.

**Step 5:** End the program

### PROGRAM:
```
Developed By: G.Chethan Kumar
Registration NO.: 212222240022
```

## A - LINEAR TREND ESTIMATION

```python
# LINEAR TREND ESTIMATION

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('india-gdp.csv',nrows=50)
data['date'] = pd.to_datetime(data['date'], format='%d-%m-%Y')
daily_average = data.groupby('date')[' AnnualChange'].mean().reset_index()

# Linear trend estimation
x = np.arange(len(daily_average))  # This should have the same length as daily_average['Revenue']
y = daily_average[' AnnualChange']

linear_coeffs = np.polyfit(x, y, 1)
linear_trend = np.polyval(linear_coeffs, x)

# Plotting
plt.figure(figsize=(14, 7))
plt.plot(daily_average['date'], daily_average[' AnnualChange'], label='Original Data', marker='o')
plt.plot(daily_average['date'], linear_trend, label='Linear Trend', color='red')
plt.title('Linear Trend Estimation')
plt.xlabel('Year')
plt.ylabel('Annual % Change')
plt.legend()
plt.grid()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

## B- POLYNOMIAL TREND ESTIMATION
```python
# POLYNOMIAL TREND ESTIMATION

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('india-gdp.csv',nrows=50)
data['date'] = pd.to_datetime(data['date'], format='%d-%m-%Y')
daily_average = data.groupby('date')['AnnualChange'].mean().reset_index()

# Polynomial trend estimation (degree 2)
x = np.arange(len(daily_average))
y = daily_average['AnnualChange']
poly_coeffs = np.polyfit(x, y, 2)
poly_trend = np.polyval(poly_coeffs, x)

# Plotting
plt.figure(figsize=(14, 7))
plt.plot(daily_average['date'], daily_average['AnnualChange'], label='Original Data', marker='o')
plt.plot(daily_average['date'], poly_trend, label='Polynomial Trend (Degree 2)', color='green')
plt.title('Polynomial Trend Estimation')
plt.xlabel('Year')
plt.ylabel('Annual % Change')
plt.legend()
plt.grid()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```
### OUTPUT


### A - LINEAR TREND ESTIMATION

![Screenshot 2024-09-13 084556](https://github.com/user-attachments/assets/7c637e18-4a58-4210-abc1-21643987e70e)

### B- POLYNOMIAL TREND ESTIMATION

![Uploading Screenshot 2024-09-13 084611.png…]()

### RESULT:
Thus the python program for linear and Polynomial Trend Estiamtion has been executed successfully.
