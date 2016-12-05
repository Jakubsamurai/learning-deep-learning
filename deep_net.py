import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

'''
we send the weighted input data to hidden layer 1
this layer has an activation function that gives new weights
and it goes to the output layer
we compare the output to the intended output with a cost
function, and we run a optimizer (AdamOptimizer). this goes
backwards and modifies the weights (backpropagation)
the cycle: feed forward + backpropagation = epoch
'''
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

'''
one_hot does that only one component is 'hot':
0 = [1,0,0,0,0,0,0,0,0,0]
1 = [0,1,0,0,0,0,0,0,0,0]
...
'''
# we define the three hidden layers
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

# we define the classes
n_classes = 10
# setting the batch of images size
batch_size = 100

'''
tf.placeholder(dtype, shape=None, name=None)
Inserts a placeholder for a tensor that will be always fed.
'''
# x is data
x = tf.placeholder('float', [None, 784])
# y are the value for tha data
y = tf.placeholder('float')

'''
When you train a model, you use variables to hold and
update parameters. Variables are in-memory buffers
containing tensors. They must be explicitly initialized
and can be saved to disk during and after training.
You can later restore saved values to exercise or analyze
the model.
'''

def neural_network_model(data):
    
    # bias exists to add a default value to fire neurons even with 0 input
    # changed biases for more acurracy to 1 from tf.Variable(tf.random_normal(n_nodes_hl3))
    hidden_1_layer = {'weights': tf.Variable(tf.random_normal([784, n_nodes_hl1])),
                      'biases': 1}
    
    hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases': 1}
    
    hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases': 1}
    
    output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                      'biases': tf.Variable(tf.random_normal(n_classes))}                                            
    
    # (input data * weights) + biases
    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']) + hidden_1_layer['biases'])
    # we apply the values in the layer one to the activation function
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']) + hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']) + hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.add(tf.matmul(l3, output_layer['weights']) + output_layer['biases'])

    return output
