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
    
