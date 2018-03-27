import gym
import numpy as np
import tensorflow as tf
import sys
import itertools

import sklearn.preprocessing
import sklearn.pipeline
import collections


from sklearn.kernel_approximation import RBFSampler

env = gym.envs.make("MountainCarContinuous-v0")

# Feature Preprocessing: Normalize to zero mean and unit variance
# We use a few samples from the observation space to do this
observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(observation_examples)

# Used to converte a state to a featurizes represenation.
# We use RBF kernels with different variances to cover different parts of the space
featurizer = sklearn.pipeline.FeatureUnion([
        ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
        ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
        ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
        ("rbf4", RBFSampler(gamma=0.5, n_components=100))
        ])
featurizer.fit(scaler.transform(observation_examples))

def featurize_state(state):
   """ Returns the featurized representation for a state. """
   scaled = scaler.transform([state])
#   print("SCALED")
#   print(scaled)
   featurized = featurizer.transform(scaled)
#   print("FEATURIZED")
#   print(featurized[0])
   return featurized[0]

class PolicyEstimator():
   """ Policy Function approximator """
   def __init__(self, learning_rate=0.01, scope="policy_estimator"):
      with tf.variable_scope(scope):
         self.state = tf.placeholder(tf.float32, shape=[400], name="state")
         self.target = tf.placeholder(dtype=tf.float32, name="target")

         # This is just s linear classifier
         self.mu = tf.contrib.layers.fully_connected(
               inputs=tf.expand_dims(self.state, 0),
               num_outputs=1,
               activation_fn=None,
               weights_initializer=tf.zeros_initializer)
         self.mu = tf.squeeze(self.mu)

         self.sigma = tf.contrib.layers.fully_connected(
               inputs=tf.expand_dims(self.state, 0),
               num_outputs=1,
               activation_fn=None,
               weights_initializer=tf.zeros_initializer)

         self.sigma = tf.squeeze(self.sigma)
         self.sigma = tf.nn.softplus(self.sigma) + 1e-5
         self.normal_dist = tf.contrib.distributions.Normal(self.mu, self.sigma)
         self.action = self.normal_dist._sample_n(1)
         self.action = tf.clip_by_value(self.action, env.action_space.low[0], env.action_space.high[0])

         # loss and train op
         self.loss = -self.normal_dist.log_prob(self.action) * self.target
         # add cross entropy cost to encourage exploration
         self.loss -= 1e-1 * self.normal_dist.entropy()

         self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
         self.train_op = self.optimizer.minimize(self.loss, global_step=tf.contrib.framework.get_global_step())

   def predict(self, state, sess=None):
      # evalutes to the left most argument that is true
      sess = sess or tf.get_default_session()
      state = featurize_state(state)
      return sess.run(self.action, feed_dict={self.state: state})

   def update(self, state, target, action, sess=None):
      sess = sess or tf.get_default_session()

class ValueEstimator():
   """ Value Function approximator """
   def __init__(self, learning_rate=0.1, scope="value_estimator"):
      with tf.variable_scope(scope):
         self.state = tf.placeholder(tf.float32, [400], "state")
         self.target = tf.placeholder(tf.float32, name="target")

         self.output_layer = tf.contrib.layers.fully_connected(
               inputs=tf.expand_dims(self.state, 0),
               num_outputs=1,
               activation_fn=None,
               weights_initializer=tf.zeros_initializer())

         self.value_estimate = tf.squeeze(self.output_layer)
         self.loss = tf.squared_difference(self.value_estimate, self.target)

         self.optimzer = tf.train.AdamOptimizer(learning_rate=learning_rate)
         self.train_op = self.optimzer.minimize(
               self.loss, global_step=tf.contrib.framework.get_global_step())

   def predict(self, state, sess=None):
      sess = sess or tf.get_default_session()
      state = featurize_state(state)
      return sess.run(self.value_estimate, feed_dict={self.state: state})

   def update(self, state, target, sess=None):
      sess = sess or tf.get_default_session()
      state = featurize_state(state)
      feed_dict = {self.state: state, self.target: target}
      _, loss = sess.run([self.train_op, self.loss], feed_dict)
      return loss

def actor_critic(env, estimator_policy, estimator_value, num_episodes, discount_factor=1.0):
   """
    Actor Critic Algorithm. Optimizes the policy
    function approximator using policy gradient.

    Args:
        env: OpenAI environment.
        estimator_policy: Policy Function to be optimized
        estimator_value: Value function approximator, used as a critic
        num_episodes: Number of episodes to run for
        discount_factor: Time-discount factor

    Returns:
        An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """
   # keeps track of useful statistics
   Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

   for i_episode in range(num_episodes):
      state = env.reset()

      episode = []

      for t in itertools.count():

         env.render()

         # step
         action = estimator_policy.predict(state)
         next_state, reward, done, _ = env.step(action)

         # keep track of transition
         episode.append(Transition(state=state, action=action, reward=reward, next_state=next_state, done=done))

         # update statistics

         value_next = estimator_value.predict(next_state)
         td_target = reward + discount_factor * value_next
         td_error = td_target - value_next

         # update value estimator
         estimator_value.update(state, td_error)

         estimator_policy.update(state, td_error, action)

         # print out which step we are on
         print("\rStep {} @ Episode {}/{}".format(
               t, i_episode + 1, num_episodes), end="")

         if done:
            break
         state = next_state


tf.reset_default_graph()

global_step = tf.Variable(0, name="global_step", trainable=False)
policy_estimator = PolicyEstimator(learning_rate=0.001)
value_estimator = ValueEstimator(learning_rate=0.1)

with tf.Session() as sess:
   sess.run(tf.global_variables_initializer())
   actor_critic(env, policy_estimator, value_estimator, 50, discount_factor=0.95)