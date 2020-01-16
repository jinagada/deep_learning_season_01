import tensorflow.compat.v1 as tf
import numpy as np

tf.disable_v2_behavior()
sess = tf.Session()
matrix1 = tf.constant([[1., 2.], [3., 4.]])
matrix2 = tf.constant([[1.], [2.]])
print("Metrix 1 shap", matrix1.shape)
print("Metrix 2 shap", matrix2.shape)
print(tf.matmul(matrix1, matrix2).eval(session=sess))
matrix1 = tf.constant([[1., 2.]])
matrix2 = tf.constant(3.)
print((matrix1 + matrix2).eval(session=sess))
print(tf.reduce_mean([1, 2], axis=0).eval(session=sess))
x = [[1., 2.], [3., 4.]]
print(tf.reduce_mean(x).eval(session=sess))
print(tf.reduce_mean(x, axis=0).eval(session=sess))
print(tf.reduce_mean(x, axis=1).eval(session=sess))
print(tf.reduce_mean(x, axis=-1).eval(session=sess))
print(tf.reduce_sum(x).eval(session=sess))
print(tf.reduce_sum(x, axis=0).eval(session=sess))
print(tf.reduce_sum(x, axis=-1).eval(session=sess))
print(tf.reduce_mean(tf.reduce_sum(x, axis=-1)).eval(session=sess))
x = [[0, 1, 2], # axis0 →
     [2, 1, 0]] # axis1 ↓
print(tf.argmax(x, axis=0).eval(session=sess))
print(tf.argmax(x, axis=1).eval(session=sess))
print(tf.argmax(x, axis=-1).eval(session=sess))
t = np.array([[[0, 1, 2],
               [3, 4, 5]],
              [[6, 7, 8],
               [9, 10, 11]]])
print(t.shape)
print(tf.reshape(t, shape=[-1, 3]).eval(session=sess))
print(tf.reshape(t, shape=[-1, 1, 3]).eval(session=sess))
print(tf.squeeze([[0], [1], [2]]).eval(session=sess))
print(tf.expand_dims([0, 1, 2], axis=1).eval(session=sess))
print(tf.one_hot([[0], [1], [2], [3]], depth=3).eval(session=sess))
t = tf.one_hot([[0], [1], [2], [0]], depth=3)
print(tf.reshape(t, shape=[-1, 3]).eval(session=sess))
x = [1, 4]
y = [2, 5]
z = [3, 6]
print(tf.stack([x, y, z]).eval(session=sess))
print(tf.stack([x, y, z], axis=1).eval(session=sess))
print(tf.stack([x, y, z], axis=0).eval(session=sess))
print(tf.stack([x, y, z], axis=-1).eval(session=sess))
for x, y in zip([1, 2, 3], [4, 5, 6]):
    print(x, y)
for x, y, z in zip([1, 2, 3], [4, 5, 6], [7, 8, 9]):
    print(x, y, z)
