# SVR

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
# for SVR regresssion y needs to be an (n, 1) array
y = dataset.iloc[:, 2:3].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling is important in SVR
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X) # transform is scaling X and y
y = sc_y.fit_transform(y)
y1d = np.ravel(y)
# StandardScaler method works with 2d arrays, so y has to be changed

# Fitting SVR Model to the dataset
# Create your regressor and fit the regressor here
from sklearn.svm import SVR
regressor = SVR(kernel='rbf') # Gaußian kernel
regressor.fit(X, y1d) # has to be fitted to the 1D y

# Predicting a new result (employee)
y_test = np.array([6.5]) 
# transform the reshaped y_test, then predict, then inverse transform to get dollar value
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(y_test.reshape(-1,1))))

# Visualising the transformed SVR results
plt.figure(1)
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (Transformed Regression Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
# CEO is not fit well because it is considered an outlier

# plot with inverse transformed SVR values
plt.figure(2)
X_plot = sc_X.inverse_transform(X)
y_plot = sc_y.inverse_transform(y)
plt.scatter(X_plot, y_plot, color = 'red')
plt.plot(X_plot, sc_y.inverse_transform(regressor.predict(X)), color = 'blue')
plt.title('Truth or Bluff (Regression Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the SVR results (for higher resolution and smoother curve)
#X_grid = np.arange(min(X), max(X), 0.1)
#X_grid = X_grid.reshape((len(X_grid), 1))
#plt.scatter(X, y, color = 'red')
#plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
#plt.title('Truth or Bluff (Regression Model)')
#plt.xlabel('Position level')
#plt.ylabel('Salary')
#plt.show()