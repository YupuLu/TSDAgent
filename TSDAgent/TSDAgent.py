#   _____                    _ _ _               ____        _               _                        ____            _     _                
#  |_   _| __ __ ___   _____| | (_)_ __   __ _  / ___|  __ _| | ___  ___  __| |_ __ ___  _ __   ___  |  _ \ _ __ ___ | |__ | | ___ _ __ ___  
#    | || '__/ _` \ \ / / _ \ | | | '_ \ / _` | \___ \ / _` | |/ _ \/ __|/ _` | '__/ _ \| '_ \ / _ \ | |_) | '__/ _ \| '_ \| |/ _ \ '_ ` _ \ 
#    | || | | (_| |\ V /  __/ | | | | | | (_| |  ___) | (_| | |  __/\__ \ (_| | | | (_) | | | |  __/ |  __/| | | (_) | |_) | |  __/ | | | | |
#    |_||_|  \__,_| \_/ \___|_|_|_|_| |_|\__, | |____/ \__,_|_|\___||___/\__,_|_|  \___/|_| |_|\___| |_|   |_|  \___/|_.__/|_|\___|_| |_| |_|
#                                        |___/                                                                                               
botName='demo-defbot'

import json
import math
import numpy as np

class SCNN:
    """
    A Noisy Chaotic Neural Network for Solving Combinatorial Optimization Problems: 
    Stochastic Chaotic Simulated Annealing
    By Lipo Wang, Sa Li, Fuyu Tian, and Xiuju Fu, 2004
    """
    
    def __init__(self, citylist):
        n = len(citylist)
        self.n = n
        self.real_dis = self.cal_distance(citylist)
        self.norm_dis = self.real_dis / self.real_dis.max()
        #print(self.real_dis)
        
        # Value Initialization
        self.z = 0.1
        self.k = 0.9
        self.epsilon = 0.002
        self.W1 = self.W2 = 1
        self.alpha = 0.015 
        self.beta1 = 0.01
        self.beta2 = 0.003
        self.A = 0.002
        self.I = 0.65
        
        # Input Initialization
        self.Y = np.random.uniform(-1, 1, (n, n))
        self.X = np.zeros((n, n))
        self.ite = 0
        self.SeqX = list(range(self.n))
        self.SeqY = list(range(self.n))
        np.random.shuffle(self.SeqX)
        np.random.shuffle(self.SeqY)
        self.E = []

    # tested successful
    def distance_one2list(self, pos, citylist):
        return np.linalg.norm(np.array(citylist) - np.array([pos]), axis = 1)

    # tested successful
    def cal_distance(self, citylist):
        dis_matrix = np.zeros((self.n, self.n))
        i = 0
        while(i < self.n):
            dis_matrix[i,:] = self.distance_one2list(citylist[i], citylist)
            i += 1
        return dis_matrix
    
    def iterate_result(self):
        self.ite = 0
        while not(self.validate()):
            self.update_neurons()
            self.z *= (1 - self.beta1)
            self.A *= (1 - self.beta2)
            self.ite += 1
            
    # tested successful
    def update_result(self, target):
        return .5 * (1 + math.tanh(0.5 * target / self.epsilon))
    
    def update_neurons(self):
        SeqX = self.SeqX
        SeqY = self.SeqY
        n = self.n
        a = 0
        while(a < n):
            b = 0
            while(b < n):
                k, i = SeqX[a], SeqY[b]
                self.update_neuron(i, k)
                b += 1
            a += 1
            
    def update_neuron(self, i, k):
        X, Y, D = self.X, self.Y, self.norm_dis
        n, z, I = self.n, self.z, self.I
        W1, W2 = self.W1, self.W2
        ns = range(n)
        u = self.k * Y[i,k] - z * (X[i,k] - I)
        v = -W1 * (np.sum(X[i,:]) + np.sum(X[:,k]) - 2*X[i,k])
        w = -W2 * (np.dot(D[i,:], (X[:, (k+1) %n] + X[:, (k-1) % n])))
        Y[i,k] = u + (v + w + W1) * self.alpha + self.A
        if(i != k):
            X[i,k] = self.update_result(Y[i,k])
                    
    def validate(self):
        n = self.n
        X = self.X
        XT = X.T
        count = 0
        len1 = [len(np.where(X[i])[0]) == 1 for i in range(n)]
        len2 = [len(np.where(XT[i])[0]) == 1 for i in range(n)]
        len1 = np.where(len1)[0]
        len2 = np.where(len2)[0]
        count += len(len1)
        count += len(len2)
        return count == 2*n
        
    def route(self):
        if not self.validate():
            raise RuntimeError("Tour is not valid!")
        
        XT = self.X.T
        return [ np.where(XT[i])[0][0] for i in range(self.n)]
        
    def route_lenth(self):
        D = self.real_dis
        
        route = self.route()
        dis = 0
        for i in range(self.n):
            dis += D[route[i], route[(i+1)%self.n]]
        return dis


# These are the only additional libraries available to you. Uncomment them
# to use them in your solution.
#
#import numpy    # Base N-dimensional array package
#import pandas   # Data structures & analysis


# =============================================================================
# This calculateMove() function is where you need to write your code. When it
# is first loaded, it will play a complete game for you using the Helper
# functions that are defined below. The Helper functions give great example
# code that shows you how to manipulate the data you receive and the move
# that you have to return.
#

def calculateMove(gameState):
    print(gameState)
    a = [i for i in range(len(gameState["CityCoords"]))]  # Produces a list of all the city indexes
    np.random.shuffle(a)  # Randomly orders the list of cities
    S = SCNN(gameState['CityCoords'])
    S.iterate_result()
    r = S.route()
    r = [int(r[i]) for i in range(len(r))]
    move = {'Path': r}  # Sets move to be the random order of cities
    print(S.route_lenth())
    print(move)
    print(a)
    return move
    
