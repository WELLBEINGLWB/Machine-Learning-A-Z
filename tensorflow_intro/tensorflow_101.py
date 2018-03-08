import tensorflow as tf

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)

sess = tf.Session()

node3 = tf.add(node1, node2)


a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

# tensor abstraction over different types of structes
adder_node = a + b

print(sess.run(adder_node, {a:[1,3], b:[2,4]}))

print(sess.run(adder_node, {a:3, b:4.5}))

# new new depending on old node
add_and_triple = adder_node * 3
print(sess.run(add_and_triple, {a:3, b:4.5}))

# example for tensorboard
a = tf.constant(5, name="input_a")
b = tf.constant(3, name="input_b")
c = tf.multiply(a,b, name="multiply_c")
d = tf.add(a,b, name="add_d")
e = tf.add(c,d, name="add_e")
sess = tf.Session()
output = sess.run(e)
writer = tf.summary.FileWriter('./my_graph', sess.graph)
writer.close()
sess.close()

# example linear model
W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

print(sess.run(linear_model, {x:[1,2,3,4]}))

y = tf.placeholder(tf.float32)
sqared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(sqared_deltas)
print(sess.run(loss, 
               {x:[1,2,3,4], y:[0,-1,-2,-3]}))

# right values to see loss is corrected
fixW = tf.assign(W, [-1.])
fixb = tf.assign(b, [1.])
sess.run([fixW, fixb])
print(sess.run(loss, {x:[1,2,3,4],
                      y:[0,-1,-2,-3]}))

# gradient descent
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# run linear model with optimizer
sess.run(init)
for i in range(1000):
    sess.run(train, {x:[1,2,3,4], 
                     y:[0,-1,-2,-3]})
print(sess.run([W,b]))
