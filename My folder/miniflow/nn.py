"""
This network feeds the output of a linear transform
to the sigmoid function.

Finish implementing the Sigmoid class in miniflow.py!

Feel free to play around with this network, too!
"""

import numpy as np
from miniflow import *

X, W, b = Input(), Input(), Input()

print ("+++++++++++Before f")
f = Linear(X, W, b)
print ("+++++++++Before g")
g = Sigmoid(f)

X_ = np.array([[-1., -2.], [-1, -2]])
W_ = np.array([[2., -3], [2., -3]])
b_ = np.array([-3., -5])

print ("++Before feed")
feed_dict = {X: X_, W: W_, b: b_}
print ("++Before graph")
graph = topological_sort(feed_dict)
print ("++Before output")
output = forward_pass(g, graph)

"""
Output should be:
[[  1.23394576e-04   9.82013790e-01]
 [  1.23394576e-04   9.82013790e-01]]
"""
print(output)
