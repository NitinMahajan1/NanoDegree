# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 13:32:20 2017

@author: Nitin
"""
# In[1]:
import tensorflow as tf

# Create TensorFlow object called tensor
hello_constant = tf.constant('Hello World!')

with tf.Session() as sess:
    # Run the tf.constant operation in the session
    output = sess.run(hello_constant)
    print(output)
    
# In[2]:
x = tf.placeholder(tf.string)
y = tf.placeholder(tf.int32)
z = tf.placeholder(tf.float32)

with tf.Session() as sess:
    output = sess.run(x, feed_dict={x: 'Test String', y: 123, z: 45.67})
# In[3]:
    # Quiz Solution
# Note: You can't run code in this tab
import tensorflow as tf

# TODO: Convert the following to TensorFlow:
x = tf.constant(10)
y = tf.constant(2)
z = tf.subtract(tf.divide(x,y),tf.cast(tf.constant(1), tf.float64))

# TODO: Print z from a session
with tf.Session() as sess:
    output = sess.run(z)
    print(output)

# In[4]:   
#Initialization
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

# In[5]: 
# In[6]: 