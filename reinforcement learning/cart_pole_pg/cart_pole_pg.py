""" policy gradient cart pole """
from __future__ import division

import numpy as np
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
import math
import gym

env = gym.make("CartPole-v0")
env.reset()

# hyper parameter
H = 64 # number of hidden layer neurons
batch_size = 5 # every how many episodes to do a param update?
learning_rate = 1e-2
gamma = 0.99

D = env.observation_space.shape[0] # input dimensionality

tf.reset_default_graph()

""" forward pass of the network """
# state of the environment is used as input of the neural network
observations = tf.placeholder(tf.float32, [None, D], name="input_x")
# first weight matrix
W1 = tf.get_variable(name="W1",
                     shape=[D,H],
                     initializer=tf.contrib.layers.xavier_initializer())
# calculate weights of the hidden layer and use relu activation fn
layer1 = tf.nn.relu(tf.matmul(observations, W1))
# second weight matrix
W2 = tf.get_variable(name="W2",
                     shape=[H,1],
                     initializer=tf.contrib.layers.xavier_initializer())
score = tf.matmul(layer1, W2)
probability = tf.nn.sigmoid(score)

""" learning the policy and backpropagation """
tvars = tf.trainable_variables()
input_y = tf.placeholder(tf.float32, [None, 1], name="input_y")
advantages = tf.placeholder(tf.float32, name="reward_signal")
# The loss function. This sends the weights in the direction of making actions
# that gave good advantage (reward over time) more likely, and actions that didn't less likely.
loglik = tf.log(input_y*(input_y-probability) + (1-input_y)*(input_y+probability))
loss = -tf.reduce_mean(loglik * advantages)
newGrads = tf.gradients(loss, tvars)

# Once we have collected a series of gradients from multiple episodes, we apply them.
# We don't just apply gradeients after every episode in order to account for noise in the reward signal.
adam = tf.train.AdamOptimizer(learning_rate=learning_rate)
W1Grad = tf.placeholder(tf.float32, name="batch_grad1") # placeholders to send the final gradients through when we update
W2Grad = tf.placeholder(tf.float32, name="batch_grad2")
batchGrad = [W1Grad, W2Grad]
updateGrads = adam.apply_gradients(zip(batchGrad, tvars))


def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

xs, hs, dlogps, drs, ys, tfps = [],[],[],[],[],[]
running_reward = None
reward_sum = 0
episode_number = 1
total_episodes = 10000
batch_number = 1

init = tf.global_variables_initializer()
# launch the graph
with tf.Session() as sess:
   rendering = False
   sess.run(init)
   observation = env.reset()

   # reset the gradient placeholder. We will collect gradients in gradBuffer until we are ready to update our policy network
   # gradBuffer contains both matrices W1, W2 so ix = 0, 1
   gradBuffer = sess.run(tvars)
   for ix, grad in enumerate(gradBuffer):
      gradBuffer[ix] = grad * 0

   while episode_number <= total_episodes:
      # rendering only slows things down
      if reward_sum/batch_size > 200 or rendering == True:
         env.render()
         rendering = True

      # create column vector from observation (current state)
      x = np.reshape(observation, [1,D])

      # run policy
      tfprob = sess.run(probability, feed_dict={observations: x})
      action = 1 if np.random.uniform() < tfprob else 0

      xs.append(x) # append observation
      y = 1 if action == 0 else 0 # a "fake label"
      ys.append(y)

      # step environment and get new measurements
      observation, reward, done, info = env.step(action)
      reward_sum += reward

      drs.append(reward) # record reward after environment step()

      if done == True:
         episode_number += 1

         # stack all inputs, hidden states, action gradients and rewards for episode
         epx = np.vstack(xs)
         epy = np.vstack(ys)
         epr = np.vstack(drs)
         tfp = tfps
         xs,hs,dlogps,drs,ys,tfps = [],[],[],[],[],[] # reset array memory

         # compute the discounted reward backwards through time
         discounted_epr = discount_rewards(epr)
         # size the rewards to be unit normal (helps control gradient estimator variance)
         discounted_epr -= np.mean(discounted_epr)
         discounted_epr //= np.std(discounted_epr)

         # get the gradient for this episode and save it in the gradbuffffer
         tGrad = sess.run(newGrads, feed_dict={observations: epx,
                                               input_y: epy,
                                               advantages: discounted_epr})
         for ix, grad in enumerate(tGrad):
            gradBuffer[ix] += grad

         # after completing enough episodes update policy network with gradients
         if episode_number % batch_size == 0:
            batch_number += 1
            sess.run(updateGrads, feed_dict={W1Grad: gradBuffer[0],
                                             W2Grad: gradBuffer[1]})
            for ix, grad in enumerate(gradBuffer):
               gradBuffer[ix] = grad * 0
            # summary of how well network is doing
            running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum *0.01
            print("Average reward for batch#%d = %.0f. Total average reward = %.0f" % (batch_number, reward_sum//batch_size, running_reward//batch_size ))

            if reward_sum//batch_size > 195:
               print("Task solved in ", episode_number, " episodes!")
               break

            reward_sum = 0

         observation = env.reset()

