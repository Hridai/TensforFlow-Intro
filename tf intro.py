''' TensorFlow programs are typically split into two parts: 
    1) The construction phase (builds a computation graph representing ML model, computations 
     required to train it)
    2) The execution phase - runs a loop which evaluates a training step repeatedly (i.e. one
     step per mini-batch) gradually improving model params)
'''

''' Trivial tf use '''

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




