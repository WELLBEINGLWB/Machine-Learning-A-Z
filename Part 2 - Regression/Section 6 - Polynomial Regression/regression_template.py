# regression template

# Polynomial Regression
# y = b_0 + b_1 * x_1 + b_2 * x_1^2 + b_3 * x_1^3

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset, dude is level 6.5
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values # variable needs to be matrix
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) """
# there is not enough data to make a test and train set

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting the regression model to data set
# Create regressor object here and fit_transform the data to the regressor


# Predicting a new result with Polynomial Regression
y_pred = regressor.predict(6.5)

# visualizing the regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff Polynomial Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# visualizing the regression results (with higher resolution)
# create new X_new matrix with np.arange(), and X_new.reshape
plt.scatter(X_new, y, color = 'red')
plt.plot(X_new, regressor.predict(X_new), color = 'blue')
plt.title('Truth or Bluff Polynomial Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

