import numpy as np
import tensorflow as tf
import gym

env = gym.make("LunarLander-v2")
logs_path = "/home/andi/Documents/Machine-Learning-A-Z/reinforcement learning/lunar_lander"

n_states = env.observation_space.shape[0]
n_actions = env.action_space.n
n_neurons_hl1 = 16
n_neurons_hl2 = 16
discount = 0.95
learning_rate = 1e-2
info_size = 10

tf.reset_default_graph()

def discount_and_normalize_rewards(r, gamma=0.99):
   discounted_norm_rewards = np.zeros_like(r)
   running_return = 0
   for t in reversed(range(len(r))):
      running_return = r[t] + gamma * running_return
      discounted_norm_rewards[t] = running_return

   discounted_norm_rewards -= np.mean(discounted_norm_rewards)
   discounted_norm_rewards /= np.std(discounted_norm_rewards)

   return discounted_norm_rewards

# create the tf model
with tf.name_scope("inputs"):
   X = tf.placeholder(tf.float32, shape=(None, n_states), name="X")
   Y = tf.placeholder(tf.float32, shape=(None, n_actions), name="Y")
   discounted_norm_ep_rewards = tf.placeholder(tf.float32, shape=[None, ], name="disc_norm_reward_weight")
   ep_rewards = tf.placeholder(tf.float32, shape=[None, ], name="true_episode_rewards")

with tf.name_scope("parameters"):
   W1 = tf.get_variable(name="W1", shape=[n_states, n_neurons_hl1], initializer=tf.contrib.layers.xavier_initializer())
   b1 = tf.get_variable(name="b1", shape=[1, n_neurons_hl1], initializer=tf.contrib.layers.xavier_initializer())
   W2 = tf.get_variable(name="W2", shape=[n_neurons_hl1, n_neurons_hl2], initializer=tf.contrib.layers.xavier_initializer())
   b2 = tf.get_variable(name="b2", shape=[1, n_neurons_hl2], initializer=tf.contrib.layers.xavier_initializer())
   W3 = tf.get_variable(name="W3", shape=[n_neurons_hl2, n_actions], initializer=tf.contrib.layers.xavier_initializer())
   b3 = tf.get_variable(name="b3", shape=[1, n_actions], initializer=tf.contrib.layers.xavier_initializer())

# feed forward
with tf.name_scope("layer_1"):
   layer1 = tf.nn.relu(tf.add(tf.matmul(X, W1), b1))
with tf.name_scope("layer_2"):
   layer2 = tf.nn.relu(tf.add(tf.matmul(layer1, W2), b2))
with tf.name_scope("output_layer"):
   logits = tf.add(tf.matmul(layer2, W3), b3, name="logits")
   output_probabilities = tf.nn.softmax(logits, name="output_probs_softmax")

with tf.name_scope("loss"):
   neg_log_prob = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y)
   loss = tf.reduce_mean(neg_log_prob * discounted_norm_ep_rewards) # reward guided loss
   tf.summary.scalar(name='loss_value', tensor=loss)

with tf.name_scope("train"):
   train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

with tf.name_scope("other_scalars"):
   episode_reward_sum = tf.reduce_sum(ep_rewards)
   tf.summary.scalar(name="episode_reward_sum_value", tensor=episode_reward_sum)

with tf.Session() as sess:
   summary_writer = tf.summary.FileWriter(logs_path, sess.graph)
   merged = tf.summary.merge_all()
   saver = tf.train.Saver()
   sess.run(tf.global_variables_initializer())
   # add ops to save and restore all variables
   saver = tf.train.Saver()

   state = env.reset()
   render = False

   episode_states = []
   episode_actions = []
   episode_rewards = []

   reward_sum = 0
   episode = 0
   max_episodes = 20000

   save_path = saver.save(sess, logs_path+"/trained_model.cpkt")

   while episode < max_episodes:
      if render:
         env.render()
      # choose action
      input_state = np.reshape(state, (1, n_states))
      action_probs = sess.run(output_probabilities, feed_dict={X: input_state})
      # sample random action (exploration i guess)
      action = np.random.choice(range(len(action_probs.ravel())), p=action_probs.ravel() )

      state_, reward, done, _ = env.step(action)

      episode_states.append(state)
      action_onehot = np.identity(n_actions)[action]
      episode_actions.append(action_onehot)
      episode_rewards.append(reward)

      reward_sum += reward

      state = state_

      if done:
         episode += 1

         discounted_normalized_episode_rewards = discount_and_normalize_rewards(episode_rewards, discount)

#         print(np.array(episode_states).shape, np.array(episode_actions).shape, np.array(discount_and_normalize_rewards(episode_rewards, discount)).shape)
#         print(np.array(episode_rewards).sum())
         feed_dict={X: episode_states,
                    Y: episode_actions,
                    discounted_norm_ep_rewards: discounted_normalized_episode_rewards}

         _, cost = sess.run([train_op, loss], feed_dict=feed_dict)

         summary = sess.run(merged, feed_dict={X: episode_states,
                                               Y: episode_actions,
                                               discounted_norm_ep_rewards: discounted_normalized_episode_rewards,
                                               ep_rewards: episode_rewards})
         summary_writer.add_summary(summary)

         if episode % info_size == 0:
            print("EPISODE#%d, BATCH_REWARD %.2f" % (episode, reward_sum/info_size))

            if reward_sum/info_size > 0:
               print("FIRST POSITIVE REWARD AFTER %d EPISODES, RENDER ACTIVE" % (episode))
               render = True
            reward_sum = 0

            if reward_sum/info_size > 190:
               print("TASK SOLVED AFTER %d EPSIODES" % (episode))
               break
         # emtpy episode memory
         episode_states = []; episode_actions = []; episode_rewards = []

         state = env.reset()

# saving the variables in the model file
save_path = saver.save(sess, logs_path+"/trained_model.cpkt")












