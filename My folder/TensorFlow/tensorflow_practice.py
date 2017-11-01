
# coding: utf-8

# In[ ]:


import tensorflow as tf
x = tf.Variable(3,name='x')
y = tf.Variable(4,name='y')
f = x*x*y +y + 2
sess=tf.Session()
sess.run(x.initializer)
sess.run(y.initializer)
result=sess.run(f)
print(result)
sess.close()


# In[ ]:


import tensorflow as tf
x = tf.Variable(3,name='x')
y = tf.Variable(4,name='y')
f = x*x*y +y + 2
with tf.Session() as sess:
    x.initializer.run()
    y.initializer.run()
    result=f.eval()


# In[ ]:


print(result)


# In[ ]:


import tensorflow as tf
x = tf.Variable(3)
y = tf.Variable(4)
f = x*x*y +y + 2
init=tf.global_variables_initializer()
with tf.Session() as sess:
    init.run()
    result=f.eval()
    print(result)
x.graph is tf.get_default_graph()


# In[1]:


import tensorflow as tf
graph = tf.Graph()
with graph.as_default():
    x = tf.Variable(3)
    y = tf.Variable(4)
    f=x*x*y+y+2
    init = tf.global_variables_initializer()


# In[8]:


with tf.Session(graph=graph) as sess:
    init.run()
    result=f.eval()
    print(result)
    print(sess.run(f))


# In[ ]:


tf.get_default_graph()


# In[ ]:


x.graph


# In[ ]:


y.graph


# In[ ]:


x.graph is tf.get_default_graph()


# In[9]:


import numpy as np
from sklearn.datasets import fetch_california_housing


# In[10]:


housing = fetch_california_housing()
m,n=housing.data.shape
housing_data_plus_bias = np.c_[np.ones((m,1)),housing.data]


# In[13]:


X =tf.constant(housing_data_plus_bias,dtype=tf.float32,name="X")
y=tf.constant(housing.target.reshape(-1,1),dtype=tf.float32,name="y")
XT = tf.transpose(X)
theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT,X)),XT),y)


# In[14]:


with tf.Session() as sess:
    theta_value = theta.eval()


# In[15]:


print(theta_value)


# In[ ]:




