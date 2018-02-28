# Apriori

# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Data Preprocessing
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header=None)
transactions = []
for i in range(0, len(dataset)) :
    transactions.append([str(dataset.values[i,j]) for j in range(0, int(dataset.size/len(dataset)) )])  

# Training Apriori on the dataset
from apyori import apriori
# 3*7/7500 = 0.003 support of a product bought three times a day
rules = apriori(transactions, min_support=0.003, min_confidence=0.2, min_lift=3)

# Visualizing the results
results = list(rules)
