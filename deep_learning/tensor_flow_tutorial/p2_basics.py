# tensorflow basically functions on tensors (arrays)
# tensorflow operates outside python code thatswhy its fast
import tensorflow as tf

x1 = tf.constant(5)
x2 = tf.constant(6)

# works
#result = x1*x2
# official way to do it
# no process computes this multiplication until the session below is started
result = tf.multiply(x1,x2)

#print(result)

# gives sess variable
#sess = tf.Session()
#print(sess.run(result))
#sess.close()
# better
with tf.Session() as sess:
   output = sess.run(result)
   print(output)
# output is not defined outside the tensorflow session
print(output)