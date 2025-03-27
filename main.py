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

