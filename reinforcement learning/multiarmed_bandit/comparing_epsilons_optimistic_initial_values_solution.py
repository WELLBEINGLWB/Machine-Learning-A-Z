import numpy as np
import matplotlib.pyplot as plt

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
            
def run_experiment_optimistic(m1, m2, m3, upper_limit, N):
      bandits = [Bandit(m1, upper_limit), Bandit(m2, upper_limit), Bandit(m3, upper_limit)]
            
      data = np.empty(N)

      for i in range(N):
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
      plt.ylabel('Cumulative average')
      plt.xlabel('pulls')
      plt.xscale('log')
      plt.draw()
      plt.pause(3)
      plt.close()
      
      return cumulative_average

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
#      plt.show()
#      plt.plot(cumulative_average)
#      plt.plot(np.ones(N)*m1)
#      plt.plot(np.ones(N)*m2)
#      plt.plot(np.ones(N)*m3)
#      plt.ylabel('Cumulative average')
#      plt.xlabel('pulls')
#      plt.xscale('log')
#      plt.draw()
#      plt.pause(3)
#      plt.close()
      
      return cumulative_average
      
if __name__ == '__main__':
      c_opt = run_experiment_optimistic(1.0, 2.0, 3.0, 10, 100000)
      c_1 = run_experiment_eps_greedy(1.0, 2.0, 3.0, 10, 0.1, 100000)
#      c_01 = run_experiment(1.0, 2.0, 3.0, 10, 100000)

# log scale plot
      plt.figure(4)
      plt.plot(c_1, label='eps = 0.1')
      plt.plot(c_opt, label='optimsitc')
#      plt.plot(c_01, label='eps = 0.01')
      plt.ylabel('Cumulative average')
      plt.xlabel('pulls')
      plt.legend()
      plt.xscale('log')
      plt.show()
      
      # linear plot
      plt.figure(5)
      plt.plot(c_1, label='eps = 0.1')
      plt.plot(c_opt, label='optmistic')
#      plt.plot(c_01, label='eps = 0.01')
      plt.ylabel('Cumulative average')
      plt.xlabel('pulls')
      plt.legend()
      plt.show()
