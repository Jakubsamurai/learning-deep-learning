import tensorflow as tf

# from here to the line 10 we build the computation graph
# create two constants
''''
tf.constant(value, dtype=None, shape=None, name='Const', verify_shape=False)
Creates a constant tensor.
The resulting tensor is populated with values of type dtype,
as specified by arguments value and (optionally) shape (see examples below).
''''
x1 = tf.constant(5)
x2 = tf.constant(6)

# multyply them
# this is the same as result = x1 * x2
'''
tf.mul(x, y, name=None)
Returns x * y element-wise.
'''
result = tf.mul(x1, x2)

print(result)

# from here we run the session
# get the session object
# run the session that can modify tensors
# needs to be closed like a file
'''
A class for running TensorFlow operations.

A Session object encapsulates the environment
in which Operation objects are executed, 
and Tensor objects are evaluated.
'''
with tf.Session() as sess:
    output = sess.run(result)
    print(output)

# we can still access the variable
print(output)
