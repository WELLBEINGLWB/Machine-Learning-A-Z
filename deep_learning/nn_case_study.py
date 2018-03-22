# example of toy neural network in 2 dimensions
# http://cs231n.github.io/neural-networks-case-study/
import numpy as np
import matplotlib.pyplot as plt

# GENERATING SOME DATA
N = 100 # numbers of points per class
D = 2 # dimensionality, x and y coordinate in this example
K = 3 # number of classes - three seprable data sets
X = np.zeros((N*K, D)) # data matrix (each row = single example)
y = np.zeros(N*K, dtype='uint8') # class labels
for j in range(K):
   ix = range(N*j, N*(j+1))
   r = np.linspace(0.0, 1, N) # radius
   t = np.linspace(j*4, (j+1)*4, N) + np.random.randn(N)*0.2 # theta
   X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
   y[ix] = j
# visualize the data
plt.scatter(X[:, 0], X[:, 1], c=y, s=40)
plt.show()

# normally data preprocessing would take place here, but data is already
# in range from -1 to 1

# TRAINING SOFTMAX LINEAR CLASSIFIER
# some hyperparameters
step_size = 1e-0
reg = 1e-3 # regularization strength

num_examples = X.shape[0]
W = 0.01 * np.random.randn(D, K)
b = np.zeros((1,K))

for i in range(200):
   # compute class scores
   # each row gives the class scores to the three classes (yellow, )
   scores = np.dot(X, W) + b

   # compute the loss
   # (cross entropy method - is associated with softmax classifier)

   # get unnormilized probabilities
   exp_scores = np.exp(scores)
   # normalize them for each example
   probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
   # get the log probabilities
   correct_logprobs = -np.log(probs[range(num_examples), y])
   # calculate full los
   data_loss = np.sum(correct_logprobs) / num_examples
   reg_loss = 0.5*reg*np.sum(W*W)
   loss = data_loss + reg_loss
   if i % 10 == 0:
      print("iteration %d: loss %f" % (i, loss))
   # loss is here roughly 1.1 = np.log(1.0/3) as there are three classes to classify
   # and all weights were initialized randomly - possibility to guess correct = 1/3

   # gradient of the scores
   dscores = probs
   dscores[range(num_examples), y] -= 1
   dscores /= num_examples

   dW = np.dot(X.T, dscores)
   db = np.sum(dscores, axis=0, keepdims=True)
   dW += reg*W

   # performing update parameter
   W += -step_size * dW
   b += -step_size * db

# TRAINING A NEURAL NETWORK
# initialize parameters randomly
h = 100 # size of hidden layer
W = 0.01 * np.random.randn(D,h)
b = np.zeros((1,h))
W2 = 0.01 * np.random.randn(h,K)
b2 = np.zeros((1,K))

# some hyperparameters
step_size = 1e-0
reg = 1e-3 # regularization strength

# gradient descent loop
num_examples = X.shape[0]
for i in range(10000):

  # evaluate class scores, [N x K]
  hidden_layer = np.maximum(0, np.dot(X, W) + b) # note, ReLU activation
  scores = np.dot(hidden_layer, W2) + b2

  # compute the class probabilities
  exp_scores = np.exp(scores)
  probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K]

  # compute the loss: average cross-entropy loss and regularization
  corect_logprobs = -np.log(probs[range(num_examples),y])
  data_loss = np.sum(corect_logprobs)/num_examples
  reg_loss = 0.5*reg*np.sum(W*W) + 0.5*reg*np.sum(W2*W2)
  loss = data_loss + reg_loss
  if i % 1000 == 0:
    print("iteration %d: loss %f" % (i, loss))

  # compute the gradient on scores
  dscores = probs
  dscores[range(num_examples),y] -= 1
  dscores /= num_examples

  # backpropate the gradient to the parameters
  # first backprop into parameters W2 and b2
  dW2 = np.dot(hidden_layer.T, dscores)
  db2 = np.sum(dscores, axis=0, keepdims=True)
  # next backprop into hidden layer
  dhidden = np.dot(dscores, W2.T)
  # backprop the ReLU non-linearity
  dhidden[hidden_layer <= 0] = 0
  # finally into W,b
  dW = np.dot(X.T, dhidden)
  db = np.sum(dhidden, axis=0, keepdims=True)

  # add regularization gradient contribution
  dW2 += reg * W2
  dW  += reg * W

  # perform a parameter update
  W  += -step_size * dW
  b  += -step_size * db
  W2 += -step_size * dW2
  b2 += -step_size * db2
























































