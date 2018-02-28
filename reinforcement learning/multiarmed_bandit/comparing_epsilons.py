 # comparing epsilons
import numpy as np
from operator import attrgetter

class Bandit:
      mean_reward = 0
      num_pulls = 0
      
      def __init__(self, true_mean):
            self.true_mean = true_mean
      
      def pull(self):
            self.num_pulls += 1
            if np.random.rand() < self.true_mean:
                  reward = 1
                  self.mean_reward = (1-1/self.num_pulls)*self.mean_reward + 1/self.num_pulls* reward
                  return reward
            else:
                  reward = 0
                  self.mean_reward = (1-1/self.num_pulls)*self.mean_reward + 1/self.num_pulls* reward
                  return reward

      
bandit_1 = Bandit(0.3)
bandit_2 = Bandit(0.45)
bandit_3 = Bandit(0.58)
epsilon_1 = 0.1
epsilon_2 = 0.05
epsilon_3 = 0.01

bandit_list = [bandit_1, bandit_2, bandit_3]

for i in range(len(bandit_list)):
      print(bandit_list[i].mean_reward)

# get the max value of instance of a list of objectss
mean_max = max(bandit_list, key=attrgetter('mean_reward')).mean_reward

# find all occurences where machines have same expected mean
indices = [p.mean_reward == mean_max for p in bandit_list]

bandit_list[0]

for i in range(100000):
      bandit_1.pull()
      bandit_2.pull()
      bandit_3.pull()
      
print(bandit_1.mean_reward, bandit_1.num_pulls)
print(bandit_2.mean_reward, bandit_2.num_pulls)
print(bandit_3.mean_reward, bandit_3.num_pulls)

def run_experiment(*bandits, N, epsilon):
      roll = np.random.rand()
      mean_max = max(bandits, key=attrgetter('mean_reward')).mean_reward
      indices = [b.mean_reward == mean_max for b in bandit_list]

      for i in range(N):
            if roll < epsilon :
                  random_bandit = np.random.randint(len(bandits))
                  bandits[random_bandit].pull()
            elif sum(indices) > 1:
                  # choose random machine amongst machine with same mean_max
                  indices_same_mean_max = [index for index, value in enumerate(indices) if value == 1]
                  for i in range(sum(indices)):
                        roll_machine = np.random.rand(len(indices_same_mean_max))
                        random_bandit = indices_same_mean_max[roll_machine]
                  bandits[random_bandit].pull()
            else:
                  bandits[indices.index(1)].pull()
      
      
      
      
            