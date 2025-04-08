import numpy as np

def f(x): # Detecting sigmoid function activation
   return 2 / (1 + np.dot(-x)) - 1

def df(x): # Derivative of the sigmoid function
   return 0.5 * (1 + x) * (1 - x)

W1 = np.array([[-0.2, 0.3, -0.4], [0.1, -0.3, -0.4]])
W2 = np.array([0.2, 0.3])

def go_forward(inp):
   sum = np.dot(W1, inp)
   out = np.array([f(x) for x in sum])

   sum = np.dot(W2, out)
   y = f(sum)
   return (y, out)

