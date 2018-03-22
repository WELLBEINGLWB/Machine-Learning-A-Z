import gym
import numpy as np
import random

env = gym.make('FrozenLake-v0')

# initialization of Q table is very important here, because
# learning is slow due to single positive reward of 1 when finishing in goal
Q = 0.01*np.random.randn(env.observation_space.n, env.action_space.n)
# set terminal states to 0
Q[15] = 0
Q[12] = 0
Q[11] = 0
Q[5]  = 0

# rows = states
# columns = actions 0=LEFT, 1=DOWN, 2=RIGHT, 3=UP

DISCOUNT = 0.95
NUM_EPISODES = 2000
LEARNING_RATE = 0.8

r_steps_list = []
r_arr = np.zeros(NUM_EPISODES)
for episode in range(NUM_EPISODES):

   s = env.reset()
   ep_steps = 0
#   done = False
   # define first action
#   a = np.argmax(Q[s,:])

   while ep_steps < 99:

      # epsilon greedy choice
      epsilon = 1/(episode+1e-10)
      roll = np.random.rand()
      if roll < epsilon:
         a = np.random.randint(4)
         print("EPISODE %d RANDOM ACTION" % (episode))
      else:
         a = np.argmax(Q[s,:])

      s_, r, done, _ = env.step(a)
      ep_steps += 1

      if s_ == 15:
         print("EPISODE %d WIN" % (episode))

      Q[s,a] = Q[s,a] + LEARNING_RATE * ( r + DISCOUNT * np.max(Q[s_,:]) - Q[s,a] )

      s = s_

      if done == True:
         break

   r_arr[episode] = r
   r_steps_list.append([r, ep_steps])

print(r_arr.mean())
