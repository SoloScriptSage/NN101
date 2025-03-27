# Creating a data set

A =[0, 0, 1, 1, 0, 0,
   0, 1, 0, 0, 1, 0,
   1, 1, 1, 1, 1, 1,
   1, 0, 0, 0, 0, 1,
   1, 0, 0, 0, 0, 1]

B =[0, 1, 1, 1, 1, 0,
   0, 1, 0, 0, 1, 0,
   0, 1, 1, 1, 1, 0,
   0, 1, 0, 0, 1, 0,
   0, 1, 1, 1, 1, 0]

C =[0, 1, 1, 1, 1, 0,
   0, 1, 0, 0, 0, 0,
   0, 1, 0, 0, 0, 0,
   0, 1, 0, 0, 0, 0,
   0, 1, 1, 1, 1, 0]

# Labels creating
Y = [
   [1, 0, 0],
   [0, 1, 0],
   [0, 0, 1]
]

# Visualization of the data set

import numpy as np
import matplotlib.pyplot as plt

plt.imshow(np.array(A).reshape(5,6))
plt.show()

# Converting data and labels into a numpy arrau

X = [np.array(A).reshape(1,30),
     np.array(B).reshape(1,30),
     np.array(C).reshape(1,30)]

Y = np.array(Y)

print(X, "\n\n", Y)

# Activation function
def sigmoid(x):
   return (1/(1 + np.exp(-x)))

# Creatnig the Feed forward neural network
# 1 layes *INPUT (1,30)
# 2 layer *HIDDEN (1,5)
# 3 layer *output (3,3)

def f_forward(x, w1, w2):
   # Hidden layer
   z1 = x.dot(w1)
   a1 = sigmoid(z1)

   # =Output layer
   z2 = a1.dot(w2)
   a2 = sigmoid(z2)

   return (a2)

# Initializing the random weights


