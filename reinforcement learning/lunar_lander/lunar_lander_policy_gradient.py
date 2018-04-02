import numpy as np
import tensorflow as tf
import gym

env = gym.make("LunarLander-v2")
logs_path = '/tmp/tensorflow/lunar_batch'

n_states = env.observation_space.shape[0]
n_neurons_hl1 = 50
n_actions = env.action_space.n
discount = 0.99
learning_rate = 1e-2
info_size = 5

tf.reset_default_graph()

def discount_rewards(r, gamma=0.99):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

with tf.name_scope("Model"):
   input_state = tf.placeholder(dtype=tf.float32, shape=[None, n_states], name="input_state")
   W1 = tf.get_variable(name="W1", shape=[n_states, n_neurons_hl1], dtype=tf.float32,
                        initializer=tf.contrib.layers.xavier_initializer())
   l1 = tf.nn.relu(tf.matmul(input_state, W1))
   W2 = tf.get_variable(name="W2", shape=[n_neurons_hl1, n_actions], dtype=tf.float32,
                        initializer=tf.contrib.layers.xavier_initializer())
   # (N*T) x n_actions tensor of action logits
   logits = tf.matmul(l1, W2, name="action_logits")
   probabilities = tf.nn.softmax(logits, name="action_probabilities")
   chosen_action = tf.argmax(probabilities, axis=1)

   ep_rewards = tf.placeholder(tf.float32, shape=[None, ], name="true_episode_rewards")

with tf.name_scope("training"):
   action_holder = tf.placeholder(dtype=tf.int32, shape=[None], name="action_holder")
   reward_holder = tf.placeholder(dtype=tf.float32, shape=[None], name="reward_holder")
#   negative_likelihoods = tf.nn.softmax_cross_entropy_with_logits(labels=input_actions, logits=logits, name="negative_likelihoods")
#   weighted_negative_likelihoods = tf.multiply(negative_likelihoods, advantages, name="weighted_negative_likelihoods")
#   loss = tf.reduce_mean(weighted_negative_likelihoods, name="loss_value")

   # this should be just action_holder in the case of lunar lander?
   indexes = tf.range(0, tf.shape(probabilities)[0]) * tf.shape(probabilities)[1] + action_holder
   responsible_probabilities = tf.gather(tf.reshape(probabilities, [-1]), indexes, name="probability_of_chosen_action")
   loss = -tf.reduce_mean(tf.log(responsible_probabilities) * reward_holder, name="loss")

   tvars = tf.trainable_variables()
   gradient_holders = []
   for idx, var in enumerate(tvars):
      placeholder = tf.placeholder(tf.float32, name=str(idx)+'_holder')
      gradient_holders.append(placeholder)

   gradients = tf.gradients(loss, tvars, name="gradients")

   optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
   """ try tf.train.gradientdescent , learning rate verringern"""
   update_batch = optimizer.apply_gradients(zip(gradient_holders, tvars))

with tf.name_scope("other_scalars"):
   episode_reward_sum = tf.reduce_sum(ep_rewards)
   tf.summary.scalar(name="episode_reward_sum_value", tensor=episode_reward_sum)

tf.summary.scalar("loss", loss)
merged_summary_op = tf.summary.merge_all()

# tensors for collecting states, actions, rewards
states = np.empty(0).reshape((0, n_states))
actions = np.empty(0).reshape((0, 1))
rewards = np.empty(0).reshape((0, 1))

with tf.Session() as sess:
   sess.run(tf.global_variables_initializer())
   summary_writer = tf.summary.FileWriter(logs_path, sess.graph)
   state = env.reset()
   max_episodes = 10000
   episode = 0
   reward_sum = 0
   ep_history = []

   grad_buffer = sess.run(tf.trainable_variables())
   for idx, grad in enumerate(grad_buffer):
      grad_buffer[idx] = grad * 0

   while episode < max_episodes:
#      env.render()
      # reshape the state for the tf operations
      input_x = np.reshape(state, (1, n_states))
      # choose action allowing for some randomness
      action_probs, best_action = sess.run([probabilities, chosen_action], feed_dict={input_state: input_x})
      chosen_action_prob = np.random.choice(action_probs[0], p=action_probs[0])
      action = np.argmax(action_probs == chosen_action_prob)

      state_, reward, done, _ = env.step(action)
      # stack arrays for updating policy after episode is finished
      states = np.vstack((states, input_x))
      actions = np.vstack((actions, action))
      rewards = np.vstack((rewards, reward))
      ep_history.append([state, action, reward, state_])

      reward_sum += reward
      # set state for next time step
      state = state_

      if done:
         episode += 1
         # reset environment
         state = env.reset()

         ep_history = np.array(ep_history)
         # discount rewards and normalize them
         discounted_rewards = discount_rewards(ep_history[:,2], gamma=discount)
         discounted_rewards -= np.mean(discounted_rewards)
         discounted_rewards /= np.std(discounted_rewards)

         # train the policy
         feed_dict = {input_state: np.vstack(ep_history[:,0]),
                      action_holder: ep_history[:,1],
                      reward_holder: discounted_rewards}

         grads = sess.run(gradients, feed_dict=feed_dict)
         for idx, grad in enumerate(grads):
            grad_buffer[idx] += grad

         # write logs to summary
         summary = sess.run(merged_summary_op, feed_dict={input_state: np.vstack(ep_history[:,0]),
                                                          action_holder: ep_history[:,1],
                                                          reward_holder: discounted_rewards,
                                                          ep_rewards: rewards.ravel()})
         summary_writer.add_summary(summary)

         ep_history = []

         if episode % info_size == 0:
            feed_dict = dictionary = dict(zip(gradient_holders, grad_buffer))
            _ = sess.run(update_batch, feed_dict=feed_dict)

            for idx, grad in enumerate(grad_buffer):
               grad_buffer[idx] = grad * 0

            print("Average reward for batch %d: %.2f" % (int(episode/info_size), reward_sum/info_size))

            if reward_sum/info_size > 180:
               print("Solved after %d episodes" % (episode))
               break

            reward_sum = 0

env.close()
