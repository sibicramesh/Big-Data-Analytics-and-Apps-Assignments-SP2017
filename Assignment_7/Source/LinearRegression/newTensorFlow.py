from __future__ import print_function

import tensorflow as tf
import numpy
import matplotlib.pyplot as plt
rng = numpy.random

# Parameters
learning_rate = 0.01
training_epochs = 1000
display_step = 50

# Training Data
train_X = numpy.asarray([0.00632,0.02731,0.02729,0.03237,0.06905,0.02985,0.08829,0.14455,0.21124,0.17004,0.22489,
0.11747,0.09378,0.62976,0.63796,0.62739,1.05393,0.7842,0.80271,0.7258,1.25179,0.85204,1.23247,0.98843,0.75026,0.84054,
0.67191,0.95577,0.77299,1.00245,1.13081,1.35472,1.38799,1.15172,1.61282,0.06417,0.09744,0.08014,0.17505,0.02763,0.03359,
0.12744,0.1415,0.15936,0.12269,0.17142,0.18836,0.22927,0.25387,0.21977,0.08873,0.04337,0.0536,0.04981,0.0136,0.01311,0.02055,
0.01432,0.15445,0.10328,0.14932,0.17171,0.11027,0.1265,0.01951,0.03584,0.04379,0.05789,0.13554,0.12816,0.08826,0.15876,
0.09164,0.19539,0.07896,0.09512,0.10153,0.08707,0.05646,0.08387,0.04113,0.04462,0.03659,0.03551,0.05059,0.05735,0.05188,
0.07151,0.0566,0.05302,0.04684,0.03932,0.04203,0.02875,0.04294,0.12204,0.11504,0.12083,0.08187,0.0686,])

train_Y = numpy.asarray([0.538,0.469,0.469,0.458,0.458,0.458,0.524,0.524,0.524,0.524,0.524,0.524,0.524,0.538,0.538,
0.538,0.538,0.538,0.538,0.538,0.538,0.538,0.538,0.538,0.538,0.538,0.538,0.538,0.538,0.538,0.538,0.538,0.538,0.538,
0.538,0.499,0.499,0.499,0.499,0.428,0.428,0.448,0.448,0.448,0.448,0.448,0.448,0.448,0.448,0.448,0.439,0.439,0.439,0.439,
0.41,0.403,0.41,0.411,0.453,0.453,0.453,0.453,0.453,0.453,0.4161,0.398,0.398,0.409,0.409,0.409,0.413,0.413,0.413,0.413,
0.437,0.437,0.437,0.437,0.437,0.437,0.426,0.426,0.426,0.426,0.449,0.449,0.449,0.449,0.489,0.489,0.489,0.489,0.464,0.464,
0.464,0.445,0.445,0.445,0.445,0.445])

n_samples = train_X.shape[0]

# tf Graph Input
X = tf.placeholder("float")
Y = tf.placeholder("float")

# Set model weights
cr = tf.Variable(rng.randn(), name="CRIM")
no = tf.Variable(rng.randn(), name="NOX")

# Construct a linear model
pred = tf.add(tf.multiply(X, cr), no)

# Mean squared error
cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)
# Gradient descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Fit all training data
    for epoch in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x, Y: y})

        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            c = sess.run(cost, feed_dict={X: train_X, Y:train_Y})
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c), \
                "c=", sess.run(cr), "n=", sess.run(no))

    print("Optimization Finished!")
    training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
    print("Training cost=", training_cost, "c=", sess.run(cr), "c=", sess.run(no), '\n')

    # Graphic display
    plt.plot(train_X, train_Y, 'ro', label='Original data')
    plt.plot(train_X, sess.run(cr) * train_X + sess.run(no), label='Fitted line')
    plt.legend()
    plt.show()

    # Testing example, as requested (Issue #2)
    test_X = numpy.asarray([1.13081,1.35472,1.38799,1.15172,1.61282,0.06417,0.09744,0.08014])
    test_Y = numpy.asarray([0.538,0.538,0.538,0.538,0.538,0.538,0.538,0.538])

    print("Testing... (Mean square loss Comparison)")
    testing_cost = sess.run(
        tf.reduce_sum(tf.pow(pred - Y, 2)) / (2 * test_X.shape[0]),
        feed_dict={X: test_X, Y: test_Y})  # same function as cost above
    print("Testing cost=", testing_cost)
    print("Absolute mean square loss difference:", abs(
        training_cost - testing_cost))

    plt.plot(test_X, test_Y, 'bo', label='Testing data')
    plt.plot(train_X, sess.run(cr) * train_X + sess.run(no), label='Fitted line')
    plt.legend()
    plt.show()