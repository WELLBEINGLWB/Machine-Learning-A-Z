import numpy as np
import tensorflow as tf
import gym

def discount_reward(r, gamma):
   return np.array([val*(gamma**i) for i, val in enumerate(r)])

logs_path = '/tmp/tensorflow/g1'

env = gym.make("CartPole-v0")

dimensionality = env.observation_space.shape[0]
learning_rate = 1e-2 # learning rate used in gradient descent
nodes_hl1 = 50 # nodes of hidden neuron layers
info_size = 5
gamma = 0.99

# reset graph before constructing own graph
tf.reset_default_graph()

# define the feed forward neural network
with tf.name_scope('Model'):
   observations = tf.placeholder(name='InputData',
                                 dtype=tf.float32,
                                 shape=[None, dimensionality])
   input_y = tf.placeholder(name="LabelData", dtype=tf.float32, shape=[None, 1])
   W1 = tf.get_variable(name="W1",
                        shape=[dimensionality, nodes_hl1],
                        dtype=tf.float32,
                        initializer=tf.contrib.layers.xavier_initializer())
   l1 = tf.nn.relu(tf.matmul(observations, W1))
   W2 = tf.get_variable(name="W2",
                        shape=[nodes_hl1, 1],
                        dtype=tf.float32,
                        initializer=tf.contrib.layers.xavier_initializer())
   output_prob = tf.sigmoid(tf.matmul(l1, W2))

# define loss
with tf.name_scope("Loss"):
   advantages = tf.placeholder(dtype=tf.float32, name="RewardSignal")
   # log likelihood of going right -> action = input_y = 1
   # input_y == 1 -> tf.log(output_prob)
   # input_y == 0 -> tflog(1-output_prob)
   loglik = tf.log((1-input_y)*(1-output_prob) + input_y*(output_prob))
   loss = -tf.reduce_mean( loglik * advantages )

with tf.name_scope("Train"):
   train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# create empty tensors for states, inputs, and rewards
xs = np.empty(0).reshape(0, dimensionality)
ys = np.empty(0).reshape(0,1)
drs = np.empty(0).reshape(0,1)


# create summary to monitor cost tensor
tf.summary.scalar("loss", loss)
#tf.summary.scalar("output_prob", output_prob)
# merge all summaries
merged_summary_op = tf.summary.merge_all()

init = tf.global_variables_initializer()

with tf.Session() as sess:
   sess.run(init)
   # op to write summary to logs
   summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
   max_episodes = 2000
   episode = 0
   observation = env.reset()
   reward_sum = 0

   while episode <= max_episodes:

      # get the current state of cartpole ready for tf graph
      x = np.reshape(observation, (1,4))
      # calculate the probability of going right
      tf_prob = sess.run(output_prob, feed_dict={observations: x})

      # set action to 1/right with introducing some randomness
      action = 1 if tf_prob > np.random.uniform() else 0

      observation, reward, done, _ = env.step(action)

      reward_sum += reward
      xs = np.vstack((xs, x))
      ys = np.vstack((ys, action))
      drs = np.vstack((drs, reward))

      if done:
         episode += 1
         observation = env.reset()

         discounted_epr = discount_reward(drs, gamma)
         discounted_epr -= discounted_epr.mean()
         discounted_epr /= discounted_epr.std()

         # train the with policy gradient
         print(xs.shape, ys.shape, drs.shape)
         sess.run(train, feed_dict={observations: xs,
                                    input_y: ys,
                                    advantages: discounted_epr})

         # write logs at every iteration
         summary = sess.run(merged_summary_op, feed_dict={observations: xs,
                                                          input_y: ys,
                                                          advantages: discounted_epr})
         summary_writer.add_summary(summary)

         # empty arrays again
         xs = np.empty(0).reshape(0, dimensionality)
         ys = np.empty(0).reshape(0,1)
         drs = np.empty(0).reshape(0,1)

         if episode % info_size == 0:
            tf.Print(loss, [loss])
            print("reward for batch#%d: %f" % ((int(episode/info_size)), reward_sum/info_size))


            if reward_sum/info_size > 195:
               print("task solved in %d episodes" % (episode))
   #            writer.close()
               break

            reward_sum = 0
