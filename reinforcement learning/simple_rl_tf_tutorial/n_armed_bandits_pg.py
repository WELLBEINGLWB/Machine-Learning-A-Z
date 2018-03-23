import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

n_bandits = 5
# probability of returning reward = 1
probs_bandits = np.random.rand(n_bandits)
probs_bandits = np.array([.8, .2, .3, .1, .1])

# reset tf graph in case one is running in the background
tf.reset_default_graph()
# this is the feed forward part of the network, choosing the action
weights = tf.Variable( tf.ones(shape=[n_bandits]) )
output = tf.argmax(weights)
#output = tf.nn.softmax(weights)
# establish training procedure by feeding the reward and chosen action into the network
reward_holder = tf.placeholder(shape=[1], dtype=tf.float32)
action_holder = tf.placeholder(shape=[1], dtype=tf.int32)
responsible_weight = tf.slice(weights, action_holder, [1])
# loss fn is the negative log of the probability, but why times reward_holder
loss = -(tf.log(responsible_weight) * reward_holder)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
update = optimizer.minimize(loss)

# define loss
#loss = -tf.log(W)
# train the network

# play the game
n_episodes = 1000

init = tf.global_variables_initializer()

with tf.Session() as sess:
   sess.run(init)
   total_reward = np.zeros(n_bandits)
   bandits_chosen = []
   for episode in range(n_episodes):
      a = sess.run(output)
      # choose epsilon greedily
      epsilon = 1.0/((episode+1e-10)/3)
      if np.random.rand() < epsilon:
         a = np.random.randint(5)
      # calculate reward according to probability of that bandit
      r = 1 if np.random.rand() < probs_bandits[a] else -1

      total_reward[a] += r
      bandits_chosen.append(a)

      # update the network
      _, resp_, ww = sess.run([update, responsible_weight, weights],
                              feed_dict={reward_holder:[r],
                                         action_holder:[a]})

      if episode % 50 == 0:
         print("Running reward for ", n_bandits, " bandits: \n",
               str(total_reward))
