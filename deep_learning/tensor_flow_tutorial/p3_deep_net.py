import tensorflow as tf

# using MNIST data set, 60.000 training examples - 28x28 pixels images of digits
"""
feed-forward neural network
input > weight > hiddenlayer1 (activation function) > weights > hiddenlayer2
(activation function) > weights > output layer

compare output to intended output > cost or loss function (cross entropy)
optimization function (optimizer) > minimize cost (AdamOptimizer...SGD, AdaGrad)

backpropagation

feed forward + backprop = epoch

"""

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data", one_hot=True)

# 10 classes, 0-9
"""
0 = [1,0,0,0,0,0,0,0,0,0]
1 = [0,1,0,0,0,0,0,0,0,0]
2 = [0,0,1,0,0,0,0,0,0,0]
3 = [0,0,0,3,0,0,0,0,0,0]
"""
# nodes for hidden layers
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10
# data sets might too big for RAM
batch_size = 100 # goes through 100 of features and goes through network (pictures here)

# x input, y output/label
# and assign sizes: height x width, squashed mnist picture 28x28=784
x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

def neural_network_model(data):

   # initializing the weights randomly in a dict
   hidden_1_layer = {'weights': tf.Variable(tf.random_normal([784, n_nodes_hl1])),
                     'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}

   hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                     'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}

   hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                     'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}

   output_layer   = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                     'biases': tf.Variable(tf.random_normal([n_classes]))}

   # input_data * weights + biases
   l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
   # apply activation fn (relu) to layer 1
   l1 = tf.nn.relu(l1)

   # here the input is what ever passes through layer1
   l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
   # apply activation fn (relu) to layer 2
   l2 = tf.nn.relu(l2)

   # here the input is what ever passes through layer2
   l3 = tf.add(tf.matmul(l2, hidden_2_layer['weights']), hidden_3_layer['biases'])
   # apply activation fn (relu) to layer 1
   l3 = tf.nn.relu(l3)

   # output is one-hot array
   output = tf.add(tf.matmul(l3, output_layer['weights']), output_layer['biases'])

   return output

def train_neural_network(x):
   prediction = neural_network_model(x)
   # cross entropy with logits as our cost function
   cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y) )
   # for adam default learning_rate=0.0001
   optimizer = tf.train.AdamOptimizer().minimize(cost)

   # cycles of feedforward + backprop
   hm_epochs = 10

   with tf.Session() as sess:
      # initialize variables
      sess.run(tf.global_variables_initializer())

      for epoch in range(hm_epochs):
         epoch_loss = 0
         # this for loop depends on #total_number_of_samples / batch_size
         for _ in range(int(mnist.train.num_examples/batch_size)):
            epoch_x, epoch_y = mnist.train.next_batch(batch_size)
            # c ist cost
            _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
            epoch_loss += c
         print("Epoch", epoch+1, "completed out of", hm_epochs, "loss", epoch_loss)

      correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

      accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
      print("Accuracy", accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))


train_neural_network(x)













