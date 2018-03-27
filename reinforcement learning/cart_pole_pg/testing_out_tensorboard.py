import tensorflow as tf
import numpy as np

X = tf.placeholder(tf.float32, name="x")
Y = tf.placeholder(tf.float32, name="y")
addition = tf.add(X, Y, name="addition")

#logs_path = '/tmp/tensorflow/addition'

with tf.Session() as sess:
   sess.run(tf.global_variables_initializer())

   result = sess.run(addition, feed_dict={X: [4,2,1],
                                          Y: [-1,0,0]})


#   summary_writer = tf.summary.FileWriter(logs_path, sess.graph)
   print()
   print(result)
