import tensorflow as tf
from tensorflow.python.saved_model.builder import SavedModelBuilder
from tensorflow.python.saved_model.tag_constants import TRAINING, SERVING

x = tf.placeholder(tf.float32,shape=[None,1], name='x')
y = tf.placeholder(tf.float32,shape=[None,1], name='y')

w = tf.Variable(tf.random_uniform([1], -1.0, 1.0), name='w')
b = tf.Variable(tf.zeros([1]), name='b')
y_hat = tf.add(w * x, b, name="y_hat")

loss = tf.reduce_mean(tf.square(y_hat - y), name="loss")
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss, name='train')

init = tf.variables_initializer(tf.global_variables(), name='init')

directory = 'saved-models/saved-regression-model'
builder = SavedModelBuilder(directory)

with tf.Session(graph=tf.get_default_graph()) as sess:
    sess.run(init)

    builder.add_meta_graph_and_variables(
        sess, [TRAINING, SERVING])
    builder.save(as_text=False)