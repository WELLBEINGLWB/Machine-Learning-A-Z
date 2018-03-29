import tensorflow as tf

tf.reset_default_graph()
m1 = tf.get_variable(name="ones4", shape=[10])
m2 = tf.get_variable(name="identy4", shape=[4,4])
saver = tf.train.Saver()
logs_path = '/tmp/tensorflow/saver'


with tf.Session() as sess:

   saver.restore(sess, logs_path+"/test_saver.ckpt")

   print(m1.eval())
   print(m2.eval())