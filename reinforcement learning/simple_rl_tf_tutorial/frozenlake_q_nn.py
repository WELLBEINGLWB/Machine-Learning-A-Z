import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v0')

# initialization of Q table is very important here, because
# learning is slow due to single positive reward of 1 when finishing in goal

# neural network
n_states = env.observation_space.n
n_qvalues = env.action_space.n

tf.reset_default_graph()

# establish the feedword graph that depends on itself
inputs = tf.placeholder(shape=[1, n_states], dtype=tf.float32)
W = tf.Variable(tf.random_uniform([n_states, n_qvalues], minval=0, maxval=0.01))
Qout = tf.matmul(inputs, W)
predict = tf.argmax(Qout, 1)

# establish the loss function and the optimization step
nextQ = tf.placeholder(shape=[1,n_qvalues], dtype=tf.float32)

loss = tf.reduce_sum(tf.square(nextQ - Qout))

trainerGradient = tf.train.GradientDescentOptimizer(learning_rate=0.1)
trainerAdam = tf.train.AdamOptimizer()
#
updateModelGradient = trainerGradient.minimize(loss)
updateModelAdam = trainerAdam.minimize(loss)

DISCOUNT = 0.99
NUM_EPISODES = 2000
LEARNING_RATE = 0.8
EPSILON = 0.1


jList = []
rList = []

init = tf.initialize_all_variables()

with tf.Session() as sess:
   sess.run(init)

   for episode in range(NUM_EPISODES):
      # reset environment
      s = env.reset()
      rAll = 0
      d = False
      j = 0

      if episode % 100 == 0:
         print("episode", episode)

      while True:
         j += 1
         # np.identy(n_states)[s] returns flat one-hot encoded state array
         a, allQ = sess.run([predict, Qout], feed_dict={inputs: np.identity(n_states)[s:s+1]})

         # choose random action for epsilon greedy
         if np.random.rand() < EPSILON:
            a[0] = env.action_space.sample()

         s_, r, done, _ = env.step(a[0])

         # obtain Q_ values by feeding the state through the network
         Q_ = sess.run(Qout, feed_dict={inputs: np.identity(n_states)[s_:s_+1]})

         maxQ_ = np.max(Q_)
         targetQ = allQ
         targetQ[0, a[0]] = r + DISCOUNT * maxQ_

         # train our network using target and predicted Q values
#         _, W_ = sess.run([updateModelGradient, W], feed_dict={inputs: np.identity(n_states)[s:s+1], nextQ: targetQ})
         _, W_ = sess.run([updateModelAdam, W], feed_dict={inputs: np.identity(n_states)[s:s+1], nextQ: targetQ})
         rAll += r
         s = s_
         if done == True:
            break
      jList.append(j)
      rList.append(rAll)

s = "average reward " + str(np.array(rList).mean())
plt.text(x=10, y=0, s=s)
plt.plot(rList[NUM_EPISODES-200:NUM_EPISODES])