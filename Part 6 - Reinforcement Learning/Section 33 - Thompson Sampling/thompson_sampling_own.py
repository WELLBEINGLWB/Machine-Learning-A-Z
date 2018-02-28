# Thompson Sampling

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# Implementing Thompson Sampling
import random
N = 10000 # number of rounds
d = 10   # number of ads
ads_selected = []
numbers_of_rewards_1 = [0] * d
numbers_of_rewards_0 = [0] * d
total_reward = 0
upper_bound = 0
# Initialize by choosing each ad once
for n in range(0, N):
    ad = 0
    max_random = 0
    for i in range(0, d):
        # generate random draws
        random_beta = random.betavariate(numbers_of_rewards_1[i] + 1, numbers_of_rewards_0[i] + 1)
        if random_beta > max_random:
            max_random = random_beta
            ad = i
    ads_selected.append(ad)
    reward = dataset.values[n, ad]
    total_reward = total_reward + reward
    if reward == 1 :
        numbers_of_rewards_1[ad] = numbers_of_rewards_1[ad] + 1
    else :
        numbers_of_rewards_0[ad] = numbers_of_rewards_0[ad] + 1
    
    
# Visualizing the results
plt.figure(1)
plt.hist(ads_selected)
plt.title('Histogram of selected ads')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()