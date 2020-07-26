import numpy as np

EPS = 4.0
ETA = 0.1
TIME = 1000
INIT_WEIGHT = 0.3

def randNum():
    return 2*np.random.random(3,1) - 1

def func(x):
    return 1/np.exp(-EPS*x)

class node:
    def __init__(self, layer, wgt):
        pass
    
    
X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]])
Y = np.array([0,1,1,0]).T

