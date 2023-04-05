'''import numpy as np
np.random.seed(0)

def create_data(points, classes):
     X = np.zeros((points*classes, 2))
     y = np.zeros(points*classes, dtype='uint8')
     for class_number in range(classes):
         ix = range(points*class_number, points*(class_number+1))
         r = np.linspace(0.0, 1, points)  # radius
         t = np.linspace(class_number*4, (class_number+1)*4, points) + np.random.randn(points)*0.2
         X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
         y[ix] = class_number
     return X, y
import matplotlib.pyplot as plt
print("here")
X,y=create_data(100,3)
plt.scatter(X[:,0],X[:,1])
plt.show()
plt.scatter(X[:,0],X[:,1],c=y,cmap="brg")
plt.show()
'''
import math
layer_outputs=[4.8,1.21,2.385]

layer_outputs=[4.8,4.79,4.25]
E=math.e
exp_values=[]
for output in layer_outputs:
    exp_values.append(E**output)
print(exp_values)
norm_base=sum(exp_values)
norm_values=[]
for value in exp_values:
    norm_values.append(value/norm_base)
print(norm_values)
print(sum(norm_values))


