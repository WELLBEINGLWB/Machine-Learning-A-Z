import numpy as np
import matplotlib.pyplot as plt
from comparing_epsilons_optimistic_initial_values_solution import run_experiment_optimistic as run_experiment_optimistic

class Bandit:
      def __init__(self, m, upper_limit):
            self.m = m
            self.mean = upper_limit
            self.N = 0
      
      def pull(self):
            return np.random.randn() + self.m
      
      def update(self, x):
            # x is the reward in that pull
            self.N += 1
            self.mean = (1-1/self.N)*self.mean + 1/self.N * x 

def run_experiment_eps_greedy(m1, m2, m3, upper_limit,eps, N):
      bandits = [Bandit(m1, upper_limit), Bandit(m2, upper_limit), Bandit(m3, upper_limit)]
            
      data = np.empty(N)
      
      
      for i in range(N):
            if np.random.rand() < eps:
                  j = np.random.choice(3)
            else:  
                  j = np.argmax([b.mean for b in bandits]) # list comprehension
            x = bandits[j].pull()
            bandits[j].update(x)

            # data for plot
            data[i] = x
      cumulative_average = np.cumsum(data) / (np.arange(N) + 1)

      # plot moving average ctr
      plt.show()
      plt.plot(cumulative_average)
      plt.plot(np.ones(N)*m1)
      plt.plot(np.ones(N)*m2)
      plt.plot(np.ones(N)*m3)
      plt.title('Epsilon greedy')
      plt.ylabel('Cumulative average')
      plt.xlabel('pulls')
      plt.xscale('log')
      plt.draw()
      plt.pause(4)
      plt.close()
      
      return cumulative_average

def run_experiment_ucb(m1, m2, m3, upper_limit, N):
      bandits = [Bandit(m1, upper_limit), Bandit(m2, upper_limit), Bandit(m3, upper_limit)]
            
      data = np.empty(N)

      for i in range(N):
            # choose the machine with highest X_ucb, avoid dividing by 0
            j = np.argmax([b.mean+np.sqrt(2*np.log(N)/(b.N+0.001)) for b in bandits]) # list comprehension
            x = bandits[j].pull()
            bandits[j].update(x)

            # data for plot
            data[i] = x
      cumulative_average = np.cumsum(data) / (np.arange(N) + 1)

      # plot moving average ctr
      plt.show()
      plt.plot(cumulative_average)
      plt.plot(np.ones(N)*m1)
      plt.plot(np.ones(N)*m2)
      plt.plot(np.ones(N)*m3)
      plt.title('UCB')
      plt.ylabel('Cumulative average')
      plt.xlabel('pulls')
      plt.xscale('log')
      plt.draw()
      plt.pause(4)
      plt.close()
      
      return cumulative_average
      
if __name__ == '__main__':
      N = 100000
      c_opt = run_experiment_optimistic(1.0, 2.0, 3.0, 10, N)
      c_1 = run_experiment_eps_greedy(1.0, 2.0, 3.0, 0, 0.1, N)
      c_ucb = run_experiment_ucb(1.0, 2.0, 3.0, 0, N)

# log scale plot
      plt.figure(4)
      plt.plot(c_1, label='eps = 0.1')
      plt.plot(c_opt, label='optimsitc')
      plt.plot(c_ucb, label='ucb')
      plt.plot(np.ones(N)*3)
      plt.ylabel('Cumulative average')
      plt.xlabel('pulls')
      plt.title('Comparison log scale plot')
      plt.legend()
      plt.xscale('log')
      plt.show()
      
      # linear plot
      plt.figure(5)
      plt.plot(c_1, label='eps = 0.1')
      plt.plot(c_opt, label='optmistic')
      plt.plot(c_ucb, label='ucb')
      plt.plot(np.ones(N)*3)
      plt.title('Comparison linear scale plot')
      plt.ylabel('Cumulative average')
      plt.xlabel('pulls')
      plt.legend()
      plt.show()
