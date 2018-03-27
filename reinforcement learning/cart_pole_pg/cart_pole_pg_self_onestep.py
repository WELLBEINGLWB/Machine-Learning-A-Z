import numpy as np
import tensorflow as tf
import gym

env = gym.make("CartPole-v0")

dimensionality = env.observation_space.shape[0] # dimensionality of action space
H = 50 # hidden layer neurons
learning_rate = 1e-2
batch_size = 5 # every how many episodes to do policy gradient update
gamma = 0.99

def discount_rewards(r, gamma=0.99):
   """ returns discounted 1D np.array of floats
   e.g. f([1,1,1], 0.99) -> [1, 0.99, 0.9801]"""
   return np.array([val * (gamma**i) for i, val in enumerate(r)])

tf.reset_default_graph()
# define network graph
observations = tf.placeholder(dtype=tf.float32, shape=[None, dimensionality], name="input_x")
W1 = tf.get_variable(name="W1", shape=[dimensionality, H],
                     initializer=tf.contrib.layers.xavier_initializer())
layer_1 = tf.nn.relu(tf.matmul(observations, W1))
W2 = tf.get_variable(name="W2", shape=[H, 1],
                     initializer=tf.contrib.layers.xavier_initializer())
probability = tf.sigmoid(tf.matmul(layer_1, W2)) # squashed probability assigned to going right (a_right = 1, a_left = 0)


# define loss fn, optimizer, and train the model
#trainable_vars = tf.trainable_variables() # these are the matrices W1, and W2
trainable_vars = [W1, W2]
input_y = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="input_y") # will be the input action of cartpole environment
advantages = tf.placeholder(dtype=tf.float32, name="reward_signal") # will be discounted reward

# loglik is basically the log of the proability of the actual action taken
# if input_y == 0 -> loglik = tf.log(probability)
# if input_y == 1 -> loglik = tf.log(1-probability)
loglik = tf.log(input_y*(input_y-probability) + (1-input_y)*(input_y+probability))
# define the loss fn
loss = -tf.reduce_mean(loglik * advantages)

# Gradients
#new_grads = tf.gradients(ys=loss, xs=trainable_vars)
#W1_grad = tf.placeholder(dtype=tf.float32, name="batch_grad1")
#W2_grad = tf.placeholder(dtype=tf.float32, name="batch_grad2")

# Learning
#batchGrad = [W1_grad, W2_grad]
#adam = tf.train.AdamOptimizer(learning_rate=learning_rate)
#update_grads = adam.apply_gradients(zip(batchGrad, [W1, W2]))
# ALTERNATIVE learning for updating policy every episode
# this is equal to tf.gradients -> optimizer.apply_gradients
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# try out
#updateGradW1 = adam.apply_gradients(zip(W1_grad, W1))
#updageGradW2 = adam.apply_gradients(zip(W2_grad, W2))


init = tf.global_variables_initializer()
# placeholders for observations, outputs rewards
xs = np.empty(0).reshape(0, dimensionality)
ys = np.empty(0).reshape(0,1)
rewards = np.empty(0).reshape(0,1)
# launch the graph
with tf.Session() as sess:
   sess.run(init)
   observation = env.reset()
   n_episodes = 1000
   reward_sum = 0

   # placeholder for our gradients
#   gradients = np.array([np.zeros(var.get_shape()) for var in trainable_vars])
   current_episode = 1
   while current_episode <= n_episodes:
#      env.render()

      x = np.reshape(observation, [1, dimensionality])

      # run neural net to determine porbability
      # probability determining going left - action = 0
      tf_prob = sess.run(probability, feed_dict={observations: x})

      # determine the output based on our net, allowing for some randomeness
      y = 0 if tf_prob > np.random.uniform() else 1

      # append observations and outputs for learning
      xs = np.vstack([xs, x])
      ys = np.vstack([ys, y])

      observation, reward, done, _ = env.step(y)
      reward_sum += reward
      rewards = np.vstack([rewards, reward])

      if done == True:
         current_episode += 1
         observation = env.reset()
         # determine standardized rewards
         discounted_rewards = discount_rewards(rewards, gamma)
         discounted_rewards -= discounted_rewards.mean()
         discounted_rewards /= discounted_rewards.std()

         # append gradients for case to running gradients
#         gradients += np.array(sess.run(new_grads,
#                                        feed_dict={observations: xs,
#                                                   input_y: ys,
#                                                   advantages: discounted_rewards}))


         # this can be replaced by updating the gradients
         # this way the batch size would be set to 1
         sess.run(train, feed_dict={observations: xs,
                                    input_y: ys,
                                    advantages: discounted_rewards})

#         gradients *= 0 # clear gradients

         #clear out game variables
         xs = np.empty(0).reshape(0, dimensionality)
         ys = np.empty(0).reshape(0,1)
         rewards = np.empty(0).reshape(0,1)

         # update gradients once batch is full
         if current_episode % batch_size == 0:
#            sess.run(update_grads, feed_dict={W1_grad: gradients[0],
#                                             W2_grad: gradients[1]})
#
#            gradients *= 0 # clear gradients

            print("Average reward for batch#%d: %.2f" % (int(current_episode/batch_size), reward_sum/batch_size))

            if reward_sum/batch_size > 195:
               print("Solved in %d episodes" % (current_episode))
               break
            reward_sum = 0


   print("running trained agent")
   observation = env.reset()
   trained_episodes = 5
   current_episode = 0
   reward_sum = 0
   while current_episode <= trained_episodes:
      env.render()
      x = np.reshape(observation, [1, dimensionality])
      tf_prob = sess.run(probability, feed_dict={observations: x})

      y = 0 if tf_prob > 0.5 else 1
      observation, reward, done, _ = env.step(y)
      reward_sum += reward

      if done:
         current_episode += 1

         if current_episode == trained_episodes:
            print("average reward = ", reward_sum / trained_episodes)
            env.close()
            break
         observation = env.reset()
