import numpy as np
import tensorflow as tf
import gym
from tensorflow.python.tools import inspect_checkpoint as chkp


env = gym.make("LunarLander-v2")
logs_path = "/home/andi/Documents/Machine-Learning-A-Z/reinforcement learning/lunar_lander/pg2"

n_states = env.observation_space.shape[0]
n_actions = env.action_space.n
n_neurons_hl1 = 32
n_neurons_hl2 = 32
discount = 0.99
learning_rate = 5e-4
info_size = 10

tf.reset_default_graph()

def discount_and_normalize_rewards(r, gamma=0.99):
   discounted_norm_rewards = np.zeros_like(r)
   running_return = 0
   # Gt = Rt+1 + gamma * Rt+2 + gamma^2 * Rt+3 + ... + gamma^(T-t+1)RT
   for t in reversed(range(len(r))):
      running_return = r[t] + gamma * running_return
      discounted_norm_rewards[t] = running_return

   # subtract mean and divide by standard deviation / normalize rewards
   discounted_norm_rewards -= np.mean(discounted_norm_rewards)
   discounted_norm_rewards /= np.std(discounted_norm_rewards)

   return discounted_norm_rewards

# create the tf model
with tf.name_scope("inputs"):
   tf_states = tf.placeholder(tf.float32, shape=(None, n_states), name="tf_states")
   tf_actions = tf.placeholder(tf.float32, shape=(None, n_actions), name="tf_actions")
   discounted_norm_ep_rewards = tf.placeholder(tf.float32, shape=[None, ], name="disc_norm_reward_weight")
   tf_rewards = tf.placeholder(tf.float32, shape=[None, ], name="true_episode_rewards")

""" histogramm über gradienten und gewichte """

with tf.name_scope("parameters"):
   W1 = tf.get_variable(name="W1", shape=[n_states, n_neurons_hl1], initializer=tf.contrib.layers.xavier_initializer())
   b1 = tf.get_variable(name="b1", shape=[1, n_neurons_hl1], initializer=tf.contrib.layers.xavier_initializer())
   W2 = tf.get_variable(name="W2", shape=[n_neurons_hl1, n_neurons_hl2], initializer=tf.contrib.layers.xavier_initializer())
   b2 = tf.get_variable(name="b2", shape=[1, n_neurons_hl2], initializer=tf.contrib.layers.xavier_initializer())
   W3 = tf.get_variable(name="W3", shape=[n_neurons_hl2, n_actions], initializer=tf.contrib.layers.xavier_initializer())
   b3 = tf.get_variable(name="b3", shape=[1, n_actions], initializer=tf.contrib.layers.xavier_initializer())

# feed forward
with tf.name_scope("layer_1"):
   layer1 = tf.nn.relu(tf.add(tf.matmul(tf_states, W1), b1))
with tf.name_scope("layer_2"):
   layer2 = tf.nn.relu(tf.add(tf.matmul(layer1, W2), b2))
with tf.name_scope("output_layer"):
   logits = tf.add(tf.matmul(layer2, W3), b3, name="logits")
   output_probabilities = tf.nn.softmax(logits, name="output_probs_softmax")

with tf.name_scope("loss"):
   neg_log_prob = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_actions)
   loss = tf.reduce_mean(neg_log_prob * discounted_norm_ep_rewards) # reward guided loss
   tf.summary.scalar(name='loss_value', tensor=loss)

with tf.name_scope("train"):
   train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

with tf.name_scope("other_scalars"):
   episode_reward_sum = tf.reduce_sum(tf_rewards)
   tf.summary.scalar(name="episode_reward_sum_value", tensor=episode_reward_sum)

train = False
#saver = tf.train.import_meta_graph('/home/andi/Documents/Machine-Learning-A-Z/reinforcement learning/lunar_lander/trained_model.ckpt.meta')
with tf.Session() as sess:
   saver = tf.train.Saver() # used to save/restore tf variables
#   saver = tf.train.import_meta_graph('trained_model.ckpt.meta')
   summary_writer = tf.summary.FileWriter(logs_path, sess.graph)
   merged = tf.summary.merge_all()
   chkp.print_tensors_in_checkpoint_file("trained_model.ckpt", tensor_name='', all_tensors=True, all_tensor_names='')
   if train == True:
      sess.run(tf.global_variables_initializer())
#   else:
#      saver.restore(sess, save_path='./trained_model')
#      ckpt = tf.train.get_checkpoint_state()
#      if ckpt and ckpt.model_checkpoint_path:
#         saver.restore(sess, ckpt.model_checkpoint_path)

   state = env.reset()
   render = True

   episode_states = []
   episode_actions = []
   episode_rewards = []

   reward_sum = 0
   episode = 0
   max_episodes = 5000

#   save_path = saver.save(sess, logs_path+"/trained_model.cpkt")

   while episode < max_episodes:
      if render:
         env.render()
      # choose action
      input_state = np.reshape(state, (1, n_states))
      action_probs = sess.run(output_probabilities, feed_dict={tf_states: input_state})
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
         feed_dict={tf_states: episode_states,
                    tf_actions: episode_actions,
                    discounted_norm_ep_rewards: discounted_normalized_episode_rewards}

         _, cost = sess.run([train_op, loss], feed_dict=feed_dict)

         summary = sess.run(merged, feed_dict={tf_states: episode_states,
                                               tf_actions: episode_actions,
                                               discounted_norm_ep_rewards: discounted_normalized_episode_rewards,
                                               tf_rewards: episode_rewards})
         summary_writer.add_summary(summary)

         if episode % info_size == 0:
            print("EPISODE#%d, BATCH_REWARD %.2f" % (episode, reward_sum/info_size))

#            if reward_sum/info_size > 180 and render == False:
#               print("FIRST POSITIVE REWARD AFTER %d EPISODES, RENDER ACTIVE" % (episode))
#               render = True

            if reward_sum/info_size > 190:
               print("TASK SOLVED AFTER %d EPSIODES" % (episode))
               save_path = saver.save(sess, logs_path+"/trained_model.cpkt")
               print("saving final model after %d episodes" % (episode))
               break

            reward_sum = 0

         # save model every 500 episodes
         if episode % 500 == 0:
            print("saving model after %d episodes" % (episode))
            save_path = saver.save(sess, logs_path+"/trained_model.cpkt")

         # emtpy episode memory
         episode_states = []; episode_actions = []; episode_rewards = []

         state = env.reset()

# saving final model










