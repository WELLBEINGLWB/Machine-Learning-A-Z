import numpy as np
import random, math, gym

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop
from keras.optimizers import Adam
from keras.models import load_model

from datetime import datetime

""" ENVIRONMENT """
class Environment:
    """ creates an openai environment """
    def __init__(self, problem):
        self.problem = problem
        self.env = gym.make(problem)

    def run(self, agent, training=True):
        """ runs one epoch of the agent acting in environment and prints out total epoch reward """
        s = self.env.reset()
        R = 0
        while True:
            self.env.render()

            a = agent.act(s)

            s_, r, done, info = self.env.step(a)

            # terminal state
            if done:
                s_ = None

            if training == True:
                agent.observe( (s, a, r, s_) )
                agent.replay()

            # set next state for next step and increment reward
            s = s_
            R += r

            if done:
                break

        print("Total reward:", R)

""" AGENT """
MEMORY_CAPACITY = 100000
BATCH_SIZE = 64
GAMMA = 0.99
MAX_EPSILON = 1
MIN_EPSILON = 0.01
LAMBDA = 0.001 # speed of epsilon decay
TARGET_NETWORK_UPDATE_STEPS = 10000

class Agent:

    def __init__(self, stateCnt, actionCnt, training):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt
        self.training = training
        self.steps = 0

        if training == True:
            self.epsilon = MAX_EPSILON
        else:
           # act greedily with a trained agent
            self.epsilon = 0

        self.brain = Brain(stateCnt, actionCnt, training)
        self.memory = Memory(MEMORY_CAPACITY)

    # when loading weights this has to be changed
    def act(self, s):
        if random.random() < self.epsilon:
            return random.randint(0, self.actionCnt - 1)
        else:
            # 0 or 1 depending on Q fn value
            return np.argmax(self.brain.predictOne(s))

    # adds sample (s, a, r, s_) to memory
    def observe(self, sample):
        self.memory.add(sample)

        if self.steps % TARGET_NETWORK_UPDATE_STEPS == 0:
           self.brain.update_network()

        if self.training == True:
            self.steps += 1
            self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * self.steps)

    # replays memories and improves
    def replay(self):
        # random samples from previous trajectories
        # ( s, a, r, s_ )
        batch = self.memory.sample(BATCH_SIZE)
        batchLen = len(batch)

        no_state = np.zeros(self.stateCnt)
        # np arrays are used to improve theano computation time
        states = np.array([ o[0] for o in batch ])
        states_ = np.array([ no_state if o[3] is None else o[3] for o in batch ])

        # predicted Q function values in state s
        p = self.brain.predict(states)
        # predictions of Q function in state s_ taking action a
        p_ = self.brain.predict(states_)

        x = np.zeros((batchLen, self.stateCnt))
        y = np.zeros((batchLen, self.actionCnt))

        # create training data
        for i in range(batchLen):
            o = batch[i]
            s = o[0]; a = o[1]; r = o[2]; s_ = o[3]

            # target values in vector form
            t = p[i]
            # only the target given action a is updated in the
            if s_ is None:
                t[a] = r
            else:
                t[a] = r + GAMMA * np.amax(p_[i])

            # x consists of the states
            # y consists of the new updated targets
            x[i] = s
            y[i] = t
        #perform single supervised learning step
        self.brain.train(x, y)

    def update_network(self):
       pass


""" BRAIN """
class Brain:

    def __init__(self, stateCnt, actionCnt, training):
        self.training = training
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt

        if training == True:
            self.model  = self._createModel()
            # create two neural nets for FULL DQN
            self.model_ = self._createModel()
        else:
            self.model = load_model('cartpole-basic-2018-03-09-18-44.h5')

    def _createModel(self):
        model = Sequential()

        # try out different add version with input_shape
        model.add(Dense(units=64, activation='relu', input_dim=self.stateCnt))
#        model.add(Dense(units=64, activation='relu', input_shape=(self.stateCnt,)))
        # 2nd layer does not need input shape as it can derive it from first layer
        # output is the action
        model.add(Dense(units=self.actionCnt, activation='linear'))

#        opt = Adam()
        opt = RMSprop(lr=0.00025)
        model.compile(loss='mse', optimizer=opt)

        return model

    def train(self, x, y, epoch=1, verbose=0):
        # performs the single supervised training step
        self.model.fit(x, y, batch_size=64, epochs=1, verbose=verbose)

    def predict(self, s):
        """ predicts the ouput value with the stable target network """
        # predicts the output value = action (numpy array)
        return self.model_.predict(s)


    def predictOne(self, s):
        # question what shape does s have?
        # and what shape does predict return?
        # needs to be flattened because the argmax of the flattened array is calculated
        return self.predict(s.reshape(1, self.stateCnt)).flatten()

    def update_network(self):
       # update the target network weights
       self.model_.set_weights(self.model.get_weights())

""" MEMORY """
class Memory:
    # stores samples as ( s, a, r, s_ )
    samples_buffer = []

    def __init__(self, capacity):
        self.capacity = capacity

    def add(self, sample):
        self.samples_buffer.append(sample)
        # streaming samples
        if len(self.samples_buffer) > self.capacity:
            self.samples_buffer.pop(0)

    # returns a batch of n samples
    def sample(self, n):
        # make sure batch returns complies the availbale samples
        n = min(n, len(self.samples_buffer))
        return random.sample(self.samples_buffer, n)

    def isFull(self):
       return len(self.samples_buffer) >= self.capacity

class RandomAgent:
   memory = Memory(MEMORY_CAPACITY)

   def __init__(self, actionCnt):
      self.actionCnt = actionCnt

   def act(self, s):
      return random.randint(0, self.actionCnt - 1)

   def observe(self, sample):
      if len(self.memory.samples_buffer) % 1000 == 0:
         print("memory samples length %d" % (len(self.memory.samples_buffer)))
      self.memory.add(sample)

   def replay(self):
      pass

# %%
""" MAIN """
# TRAINING
PROBLEM = 'CartPole-v0'
env = Environment(PROBLEM)

stateCnt = env.env.observation_space.shape[0]
actionCnt = env.env.action_space.n



agent = Agent(stateCnt, actionCnt, training=True)
randomAgent = RandomAgent(actionCnt)

try:
   while randomAgent.memory.isFull() == False:
      env.run(randomAgent)

   agent.memory.samples_buffer = randomAgent.memory.samples_buffer

   while True:
        env.run(agent, training=True)
finally:
    # code is run after keyboard interrupt
    file_save_name = "cartpole-full-" + datetime.now().strftime('%Y-%m-%d-%H-%M') + ".h5"
    agent.brain.model.save(file_save_name)
    env.env.close()

# TEST TRAINED MODEL
#agent = Agent(stateCnt, actionCnt, training=False)

#episode_count = 5
#done = False
## speed of rendering is highly dependent on training
#for i in range(episode_count):
#    env.run(agent, training=False)
#env.env.close()


