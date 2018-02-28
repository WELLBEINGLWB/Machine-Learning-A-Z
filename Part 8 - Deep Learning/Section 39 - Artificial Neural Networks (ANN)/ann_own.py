# Artificial neural networks

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 1] = labelencoder_X.fit_transform(X[:, 1])
labelencoder_X = LabelEncoder()
X[:, 2] = labelencoder_X.fit_transform(X[:, 2])
# create dummy variables for Geography
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
# Avoid dummy variable trap
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Feature Scaling is necessary for ANN
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - create ANN

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Either define sequence of layers or a graph for ANN
# Initialising the ANN
classifier = Sequential()

# take the average of the number of inputs and outputs for hidden layer nodes
# here (11 inputs + 1 output)/2 = 6

# addidng the input layer and the first hidden layer
classifier.add(Dense(output_dim=6, init='uniform', activation='relu', input_dim=11))

# add the second hidden layer - input_dim specifier is not necessary
classifier.add(Dense(output_dim=6, init='uniform', activation='relu'))

# adding the output layer - output_dim = 1 for our case
# when DV has more than two categories, output_dim has to be changed and activation function is called softmax
classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))

# Compiling the ANN (apply stochastic gradient descent)
# logarithmic loss function due to categorical output with sigmoid activation function
# metrics expects a list of metrics, thats why using brackets
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics= ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size=10,epochs=100)

# Predicting the Test set results
# probability that each customer leaves the bank
y_pred = classifier.predict(X_test)
# create binary prediction with threshold
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Step 1 initialise the weights to small numbers

# Step 2 - input the input values in the input layer.

# Step 3 - Forward Propagation, neurons are activated 
# influenced by activation function and weights (use rectifier here)
# Propagate activations until y_pred is calculated (use sigmoid function for output layer)

# Step 4 - compare the predicted result to actual result - Measure error

# Step 5 - Back Propagation. The error is back propagated. Update weights
# according to how much they are responsible for the error.
# learning rate decides how much we update the weights

# Step 6 - Repeat steps 1 - 5 and update weights after each observation
# or repeat steps 1 - 5 and update weights after several iterations (batch)

# Step Steps 1 - 5 is one epoche, repeat many epochs