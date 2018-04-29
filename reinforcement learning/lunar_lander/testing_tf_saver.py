import numpy as np
import tensorflow as tf

tf.reset_default_graph()

m1 = tf.get_variable(name="ones4", shape=[10], initializer=tf.initializers.ones())
m2 = tf.get_variable(name="identy4", shape=[4,4], initializer=tf.initializers.identity())

inc_m1 = m1.assign(m1+1)
dec_m2 = m2.assign(m2-1)

logs_path = "/home/andi/Documents/Machine-Learning-A-Z/reinforcement learning/lunar_lander/test"

with tf.Session() as sess:
   sess.run(tf.global_variables_initializer())
   saver = tf.train.Saver()

   sess.run(inc_m1)
   print(m1.eval())
   sess.run(dec_m2)
   print(m2.eval())

   save_path = saver.save(sess, logs_path+"/test_saver.ckpt")

