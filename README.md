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
data = pd.read_csv('MonthValue.csv',nrows=50)
data['Period'] = pd.to_datetime(data['Period'])
daily_average = data.groupby('Period')['Revenue'].mean().reset_index()

# Linear trend estimation
x = np.arange(len(daily_average))
y = daily_average['Revenue']
linear_coeffs = np.polyfit(x, y, 1)
linear_trend = np.polyval(linear_coeffs, x)

# Plotting
plt.figure(figsize=(14, 7))
plt.plot(daily_average['Period'], daily_average['Revenue'], label='Original Data', marker='o')
plt.plot(daily_average['Period'], linear_trend, label='Linear Trend', color='red')
plt.title('Linear Trend Estimation')
plt.xlabel('Period')
plt.ylabel('Revenue')
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
data = pd.read_csv('MonthValue.csv',nrows=50)
data['Period'] = pd.to_datetime(data['Period'])
daily_average = data.groupby('Period')['Revenue'].mean().reset_index()


# Polynomial trend estimation (degree 2)
x = np.arange(len(daily_average))
y = daily_average['Revenue']
poly_coeffs = np.polyfit(x, y, 2)
poly_trend = np.polyval(poly_coeffs, x)

# Plotting
plt.figure(figsize=(14, 7))
plt.plot(daily_average['Period'], daily_average['Revenue'], label='Original Data', marker='o')
plt.plot(daily_average['Period'], poly_trend, label='Polynomial Trend (Degree 2)', color='green')
plt.title('Polynomial Trend Estimation')
plt.xlabel('Period')
plt.ylabel('Revenue')
plt.legend()
plt.grid()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


```
### OUTPUT


### A - LINEAR TREND ESTIMATION

![Screenshot 2024-09-09 113247](https://github.com/user-attachments/assets/f84bcf79-2ee6-4268-b5d0-dd9e7b75a000)


### B- POLYNOMIAL TREND ESTIMATION

![Screenshot 2024-09-09 113304](https://github.com/user-attachments/assets/5f936b97-688f-4ed2-bbf2-06520f145ecd)


### RESULT:
Thus the python program for linear and Polynomial Trend Estiamtion has been executed successfully.
