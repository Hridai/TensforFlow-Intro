''' TensorFlow programs are typically split into two parts: 
    1) The construction phase (builds a computation graph representing ML model, computations 
     required to train it)
    2) The execution phase - runs a loop which evaluates a training step repeatedly (i.e. one
     step per mini-batch) gradually improving model params)
'''

''' Trivial tf use
    Creating and setting up graphs
'''

import tensorflow as tf
x = tf.Variable(3, name='x')
y = tf.Variable(4, name='y')
f = x*x*y+y+2


## can run like this:
with tf.Session() as sess:
    x.initializer.run()
    y.initializer.run()
    result = f.eval()

## or like this:
init = tf.global_variables_initializer()
with tf.Session() as sess:
    init.run()
    result = f.eval()

## or like this:
sess = tf.InteractiveSession()
init.run()
result = f.eval()
print(result)


x1 = tf.Variable(1)
x1.graph is tf.get_default_graph()

## or
graph = tf.Graph()
with graph.as_default():
    x2 = tf.Variable(2)
    x2.graph is graph
x2.graph is tf.get_default_graph() ## shows that the graph is not default if in "when" block
tf.reset_default_graph()

## or
w = tf.constant(3)
x = w + 2
y = x + 5
z = x * 3

with tf.Session() as sess:
    print(y.eval())
    print(z.eval())
# worth noting, when calculating z, even though y is a dependent, it recalculates it from scratch
# the y value previously calculated does not live in memory. multiple sessions do not share the
# the same state. Variable states are stored on the servers, not in sessions.
    
''' Implementing Batch Gradient Descent '''
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

housing = fetch_california_housing()
m,n = housing.data.shape
housing_data_plus_bias = np.c_[np.ones((m,1)),housing.data]
standardscaler = StandardScaler(with_mean=True,with_std=True)
standardscaler.fit(housing_data_plus_bias)
scaled_housing_data_plus_bias = standardscaler.transform(housing_data_plus_bias)

n_epochs = 1000
learning_rate = 0.01

X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1,1),dtype=tf.float32, name="y")
theta = tf.Variable(tf.random_uniform([n+1,1],-1.0,1.0),name='theta')
y_pred = tf.matmul(X,theta,name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error),name="mse")
##gradients = tf.gradients(mse, [theta])[0]
##training_op = tf.assign(theta, theta - learning_rate * gradients)
### The above two lines can be optimsed by the below:
#optimizer = tf.train.GradientDescentOptimizer( learning_rate = learning_rate)
optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
training_op = optimizer.minimize(mse)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    
    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print( 'Epoch', epoch, ' MSE = ', mse.eval())
        sess.run(training_op)
    
    best_theta = theta.eval()











