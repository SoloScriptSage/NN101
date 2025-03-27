# Creating a data set

a =[0, 0, 1, 1, 0, 0,
   0, 1, 0, 0, 1, 0,
   1, 1, 1, 1, 1, 1,
   1, 0, 0, 0, 0, 1,
   1, 0, 0, 0, 0, 1]

b =[0, 1, 1, 1, 1, 0,
   0, 1, 0, 0, 1, 0,
   0, 1, 1, 1, 1, 0,
   0, 1, 0, 0, 1, 0,
   0, 1, 1, 1, 1, 0]

c =[0, 1, 1, 1, 1, 0,
   0, 1, 0, 0, 0, 0,
   0, 1, 0, 0, 0, 0,
   0, 1, 0, 0, 0, 0,
   0, 1, 1, 1, 1, 0]

# Labels creating
y = [
   [1, 0, 0],
   [0, 1, 0],
   [0, 0, 1]
]

# Visualization of the data set

import numpy as np
import matplotlib.pyplot as plt

plt.imshow(np.array(a).reshape(5,6))
plt.show()

# Converting data and labels into a numpy arrau

x = [np.array(a).reshape(1,30),
     np.array(b).reshape(1,30),
     np.array(c).reshape(1,30)]

y = np.array(y)

print(x, "\n\n", y)

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
def generate_weights(x, y):
   l = []
   for i in range(x * y):
      l.append(np.random.randn())

   return (np.array(l).reshape(x,y))

# Using MEAN SQUARE ARROR (MSE) for losses
def loss(out, Y):
   s = (np.square(out - Y))
   s = np.sum(s) / len(y)

   return (s)

# Back propagation of error
def backPropagation(x, y, w1, w2, alpha):
   # Hidden layer
   z1 = x.dot(w1) # Input from layer 1
   a1 = sigmoid(z1) # Output of layer 2

   # Output layer
   z2 = a1.dot(w2) # INPUT OF out LAYER
   a2 = sigmoid(z2) # output of out layer

   # error in output layer
   d2 = (a2-y)
   d1 = np.multiply(
      (w2.dot(
         (d2.transpose())
      )).transpose(),
      (np.multiply(a1, 1-a1))
   )

   # Gradient for w1 and w2
   w1_adj = x.transpose().dot(d1)
   w2_adj = a1.transpose().dot(d2)

   # Updating parameters
   w1 = w1 - (alpha * (w1_adj))
   w2 = w1 - (alpha * (w2_adj))

   return (w1, w2)


def train(x, Y, w1, w2, alpha=0.01, epoch=10):
   acc = []
   losss = []
   for j in range(epoch):
      l=[]
   for i in range(len(x)):
      out = f_forward(x[i], w1, w2)
      l.append(
         (loss(out, Y[i]))
      )

      w1, w2 = backPropagation(x[i], y[i], w1, w2, alpha)

   print("epochs:", j+1, "=== acc:",
         (1-(sum(l)/len(x)))*100)
   acc.append((1 - (sum(l) / len(x))) * 100)
   losss.append(sum(l)/len(x))

   return (acc, losss, w1, w2)

def predict(x1, w1, w2):
   Out = f_forward(x, w1, w2)
   maxM = 0
   k = 0
   for i in range(len(Out[0])):
      if(maxM < Out[0][i]):
         maxM=Out[0][i]
         k=i

   if (k==0) :
      print("image is of letter A.")
   elif (k==1) :
      print("img is of letter b.")
   else:
      print("image is of letter c.")

   plt.imshow(x.reshape(5,6))
   plt.show()

w1 = generate_weights(30,5)
w2 = generate_weights(5,3)

print(w1, "\n\n", w2)

acc, losss, w1, w2 = train(x, y, w1, w2, 0.1, 100)

import matplotlib.pyplot as plt1

# plotting accuracy
plt1.plot(acc)
plt1.ylabel('Accuracy')
plt1.xlabel("Epochs:")
plt1.show()

# plotting Loss
plt1.plot(losss)
plt1.ylabel('Loss')
plt1.xlabel("Epochs:")
plt1.show()

print(w1, "\n", w2)

predict(x[1], w1, w2)
