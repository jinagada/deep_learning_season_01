import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
tf.set_random_seed(777)
X = [1, 2, 3]
Y = [1, 2, 3]
W = tf.Variable(5.)
hypothesis = X * W
gradient = tf.reduce_mean((W * X - Y) * X) * 2
cost = tf.reduce_mean(tf.square(hypothesis - Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
gvs = optimizer.compute_gradients(cost)
spply_gradients = optimizer.apply_gradients(gvs)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for step in range(100):
    print(step, sess.run([gradient, W, gvs]))
    sess.run(spply_gradients)
