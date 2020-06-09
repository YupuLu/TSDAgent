#   _____                    _ _ _               ____        _               _                        ____            _     _                
#  |_   _| __ __ ___   _____| | (_)_ __   __ _  / ___|  __ _| | ___  ___  __| |_ __ ___  _ __   ___  |  _ \ _ __ ___ | |__ | | ___ _ __ ___  
#    | || '__/ _` \ \ / / _ \ | | | '_ \ / _` | \___ \ / _` | |/ _ \/ __|/ _` | '__/ _ \| '_ \ / _ \ | |_) | '__/ _ \| '_ \| |/ _ \ '_ ` _ \ 
#    | || | | (_| |\ V /  __/ | | | | | | (_| |  ___) | (_| | |  __/\__ \ (_| | | | (_) | | | |  __/ |  __/| | | (_) | |_) | |  __/ | | | | |
#    |_||_|  \__,_| \_/ \___|_|_|_|_| |_|\__, | |____/ \__,_|_|\___||___/\__,_|_|  \___/|_| |_|\___| |_|   |_|  \___/|_.__/|_|\___|_| |_| |_|
#                                        |___/                                                                                               
botName='demo-defbot'

import numpy as np
import random
import time

class qCABC:
    """
        Solving Traveling Salesman Problem by Using Combinatorial Artificial Bee Colony Algorithms
        By Dervis Karaboga and Beyza Gorkemli, 2019
    """

    def __init__(self, gameState):
        n = len(gameState['CityCoords'])
        self.n = n
        self.dis = self.CalDistance(gameState['CityCoords'])
        
        if(n >= 200):
            self.cs = 80
        else:
            self.cs = (int) (n * 0.4)
            
        self.MaxNumber = 2000
        self.P_RC = 0.5
        self.P_CP = 0.8
        self.P_L = 0.2
        self.L_MIN = 2
        self.L_MAX = n / 2
        self.NL_MAX = 5
        self.r = 1
        self.L = 3 # 1 - 4
        self.limit = (int) (self.cs * self.n / self.L)
        
        self.X, self.fit, self.BestSolution, self.BestSolutionValue = self.Initialize(gameState)
        self.count = np.zeros(self.cs, dtype = int)
        self.neighbour = self.GenerateNeighbor()
        
        self.IsCABC = True
        #print('self.limit: ',self.limit)
        #print('BS: ', self.BestSolution, ' + BSV:', self.BestSolutionValue)
        
    """
    生成每个城市的邻居，在GSTM方法中使用
    """
    def GenerateNeighbor(self):
        sort = np.argsort(self.dis)
        return sort[:,1 : self.NL_MAX + 1]
        
    # verified successful
    def distance_one2list(self, pos, citylist):
        return np.linalg.norm(np.array(citylist) - np.array([pos]), axis = 1)

    # verified successful
    """
    计算距离矩阵（n*n）
    """
    def CalDistance(self, citylist):
        dis_matrix = np.zeros((self.n, self.n))
        i = 0
        while(i < self.n):
            dis_matrix[i] = self.distance_one2list(citylist[i], citylist)
            i += 1
        return dis_matrix
    
    # verified successful
    """
    计算一条路径的长度
    """
    def routeLength(self, route):
        D = self.dis
        dis = 0
        for i in range(self.n):
            dis += D[route[i], route[(i+1)%self.n]]
        return dis
    
    """
    初始化，包括每个雇佣蜂的解路径X，对应路径的长度的倒数fit，以及最优解。
    """
    def Initialize(self, gameState):
        X = np.empty([self.cs, self.n], dtype = int, order = 'C')
        if(gameState['MyBestSolution'] == None and gameState['OppBestSolution'] == None):
            X[:] = np.arange(self.n, dtype = int)
        else:
            if(gameState['IsLeading']):
                X[:] = gameState['MyBestSolution']
            else:
                X[:] = gameState['OppBestSolution']
        
        fit = np.empty(self.cs, dtype = float, order = 'C')
        #print(fit)
        #fit[0] = 1 / (1 + self.routeLength(X[0]))
        fit[0] = self.routeLength(X[0])
        i = 1
        while(i < self.cs):
            np.random.shuffle(X[i])
            #print(X[i])
            #fit[i] = 1 / (1 + self.routeLength(X[i]))
            fit[i] = self.routeLength(X[i])
            i+=1
        i = np.argmin(fit)
        return X, fit, X[i], fit[i]
    
    """
    雇佣蜂阶段
    """
    def EmployedBee(self):
        i = 0
        X = self.X
        fit =self.fit
        count = self.count
        while(i < self.cs):
            k = random.randint(0, self.cs-1)
            while(k == i):
                k = random.randint(0, self.cs-1)
            v, fitv = self.GSTM(i,k)
            if(fitv < fit[i]):
                #print('i: ', i, ' + v: ',v ,' + fitv: ', fitv)
                X[i] = v
                fit[i] = fitv
                count[i] = 0
                if(fitv < self.BestSolutionValue):
                    self.BestSolutionValue = fitv
                    self.BestSolution = v
                    #print('BS: ', self.BestSolution, ' + BSV:', self.BestSolutionValue)
            else:
                count[i] += 1
            i += 1
        
    """
    GSTM方法，在雇佣蜂阶段和跟随蜂阶段都会使用，非常重要。
    """
    # IMPORTANT!! 
    # MUST verify it for several times!!
    def GSTM(self, i, k):
        #v = np.empty([1, self.n], dtype = float, order = 'C')
        D = self.dis
        xi = self.X[i]
        xk = self.X[k]
        j = random.randint(0, self.n-1)
        phi = random.choice([True, False])
        
        # A new closed tour Ti♯ is produced with this operation.
        # As a result of this new connection, there will be an open sub tour Ti∗. 
        # This sub tour’s first city is assigned as R1 and last city is assigned as R2
        
        if(phi):
            # The city visited just before the city j in tour Ti is set as 
            # the city visited just before the city j in tour Tk.
            
            city = np.where(xi == xk[j])[0][0] # The city j in Ti
            citylink = np.where(xi == xk[j-1])[0][0] # The city visited just before the city j in Tk in Ti
            
        else:
            # The city visited immediately after the city j in tour Ti is set as 
            # the city visited immediately after the city j in tour Tk.
            
            citylink = np.where(xi == xk[j])[0][0] # The city j in Ti
            city = np.where(xi == xk[(j+1)%self.n])[0][0] # The city visited just after the city j in Tk in Ti
        
        # Actually these two cases work the same.
        
        #print('city: ', city, ' + citylink: ',citylink)
        if(citylink < city):
            # Ti* = [R1, R2] = [citylink + 1 : city),  Ti# = [city : -1] + [0 : citylink+1) 
            if(city - citylink > 1):
                v = np.concatenate((xi[city :] , xi[0 : citylink + 1]))
                R = xi[citylink + 1: city]

            else:
                # No modification
                v = xi
                fitv = self.fit[i]
                #print('return v: ', v)
                return v, fitv
        else:
            if(citylink - city + 1 != self.n):
            # Ti* = [R1, R2] = (citylink+1 : -1] + [0 : city),  Ti# = [city : citylink+1)
                v = xi[city: citylink + 1]  # It doesn't matter whether (citylink + 1) is larger than len(xi).
                R = np.concatenate((xi[citylink + 1 :] , xi[0 : city])) 
            else:
                v = xi
                fitv = self.fit[i]
                #print('return v: ', v)
                return v, fitv
                
        #print('v: ', v, ' + R: ', R)
        if( random.random() <= self.P_RC ):
            #print('case1')
            # Add Ti∗ to Ti♯ so that a minimum extension is generated.
            i = v.shape[0] - 1
            pos = i
            GR = D[v[i-1], R[0]] + D[v[i], R[-1]] - D[v[i], v[i-1]]
            GR1 = D[v[i], R[0]] + D[v[i-1], R[-1]] - D[v[i], v[i-1]]
            # print('v[i-1], R[0], D[v[i-1], R[0]]:', v[i-1], R[0], D[v[i-1], R[0]])
            if(GR > GR1):
                GR = GR1
                direction = False
            else:
                direction = True
            
            while(i >= 0):
                GR1 = D[v[i-1], R[0]] + D[v[i], R[-1]] - D[v[i], v[i-1]]
                if(GR > GR1):
                    pos = i
                    GR = GR1
                    direction = True
                GR1 = D[v[i], R[0]] + D[v[i-1], R[-1]] - D[v[i], v[i-1]]
                if(GR > GR1):
                    pos = i
                    GR = GR1
                    direction = False
                i -= 1
                
            #print('direction: ', direction, ' + pos: ', pos)
            #print('v[0:pos-1]: ', v[0:pos-1], ' + R:', R, ' + v[pos:]:', v[pos:])
            if(direction):
                if(pos > 0):
                    v = np.concatenate((v[0:pos], R, v[pos:]))
                else:
                    v = np.concatenate((R, v[pos:]))
            else:
                if(pos > 0):
                    v = np.concatenate((v[0:pos], R[::-1], v[pos:]))
                else:
                    v = np.concatenate((R[::-1], v[pos:]))
        else:
            if( random.random() <= self.P_CP ):
                #print('case2')
                # Start from the position of R1 in Ti and add each city of Ti∗ to the Ti♯ by rolling or mixing with PL probability
                i = R.shape[0]
                while(i > 0):
                    if( random.random() <= self.P_L ):
                        # mixing
                        arg = random.randint(0, i-1)
                    else:
                        # rolling 
                        arg = -1
                    v = np.append(v, R[arg])
                    R = np.delete(R, arg)
                    i -= 1
                    
            else:
                #print('case3')
                # Randomly select one neighbor from neighbor lists 
                # for the points R1 and R2 (NLR1 and NLR2). 
                #
                # (To guarantee the inversion, these selected points must not be 
                # the immediately preceding cities of R1 and R2 in tour Ti.)
                #
                # Invert the points NLR1 or NLR2 which provides maximum gain
                # by arranging these points to be the neighbors 
                # (immediately preceding cities) of the points R1 or R2.
                v = np.append(R, v)
                
                if(len(R) > 1):
                    R_node = 0
                    NLR = random.choice(self.neighbour[v[R_node]])
                    while(NLR == v[R_node-1]):
                        NLR = random.choice(self.neighbour[v[R_node]])
                    arg = np.where(v == NLR)[0][0]
                    if((D[v[R_node], v[R_node-1]] + D[v[arg-1], v[arg]] - D[v[R_node], v[arg]] - D[v[R_node-1], v[arg-1]]) > 0):
                        invv = v[arg:] # 这里只是因为 R_node = 0
                        v[arg:] = invv[::-1]

                    R_node = np.where(v == R[-1])[0][0]
                    NLR = random.choice(self.neighbour[v[R_node]])
                    while(NLR == v[R_node-1]):
                        NLR = random.choice(self.neighbour[v[R_node]])
                    arg = np.where(v == NLR)[0][0]
                    if((D[v[R_node], v[R_node-1]] + D[v[arg-1], v[arg]] - D[v[R_node], v[arg]] - D[v[R_node-1], v[arg-1]]) > 0):
                        if(arg > R_node): # 翻转[R_node: arg - 1]
                            invv = v[R_node: arg] # 这里 R_node != 0
                            v[R_node: arg] = invv[::-1]
                        else: # 翻转[arg: R_node-1]
                            invv = v[arg: R_node] # 这里 R_node != 0
                            v[arg: R_node] = invv[::-1]
                    
                else:
                    R_node = 0
                    NLR = random.choice(self.neighbour[v[R_node]])
                    while(NLR == v[R_node-1]):
                        NLR = random.choice(self.neighbour[v[R_node]])
                    arg = np.where(v == NLR)[0][0]
                    if((D[v[R_node], v[R_node-1]] + D[v[arg-1], v[arg]] - D[v[R_node], v[arg]] - D[v[R_node-1], v[arg-1]]) > 0):
                        invv = v[arg:] # 这里只是因为 R_node = 0
                        v[arg:] = invv[::-1]
                    
                    
        #print('v: ',v)
        #fitv = 1 / (1 + self.routeLength(v))
        fitv = self.routeLength(v)
        return v, fitv
    
    """
    跟随蜂阶段，搜索对应城市的邻居。该函数在qCABC方法中需要进一步优化。
    """
    def BestNeighbour(self, i):
        if(self.IsCABC):
            k = random.randint(0, self.cs-1)
            while(k == i):
                k = random.randint(0, self.cs-1)
            return k
            
    """
    跟随蜂阶段，只有随机概率高于（低于？）阈值的时候才会更新对应解。
    """
    def OnlookerBee(self):
        i = 0
        X = self.X
        fit = self.fit
        fitsum = sum(self.fit)
        count = self.count
        
        # 每个解更新
        while(i < self.cs):
            p = 0.9 * fit[i] / fitsum + 0.1
            count[i] += 1
            if(random.random() < p):
                k = self.BestNeighbour(i)
                v, fitv = self.GSTM(i,k)
                if(fitv < fit[i]):
                    X[i] = v
                    fit[i] = fitv
                    count[i] = 0 
                    if(fitv < self.BestSolutionValue):
                        self.BestSolutionValue = fitv
                        self.BestSolution = v
                        #print('BS: ', self.BestSolution, ' + BSV:', self.BestSolutionValue)
            i += 1
    
    """
    侦查蜂阶段
    """
    def ScoutBee(self):
        i = 0
        count = self.count
        l = self.limit
        while(i < self.cs):
            if(count[i] >= l):
                count[i] = 0
                np.random.shuffle(self.X[i])
            i += 1
    
    """
    求解函数
    """
    def Go(self):
        j = 0
        #best = [self.BestSolutionValue]
        a = self.BestSolutionValue
        clock = 0
        while(j < self.MaxNumber):
            #print('self.count: ', self.count)
            self.EmployedBee()
            self.OnlookerBee()
            self.ScoutBee()
            #best.append(self.BestSolutionValue)
            j+=1
            if(a > self.BestSolutionValue):
                clock = 0
                a = self.BestSolutionValue
            else:
                clock+=1
            if(clock == 100):
                break
            
        #plt.plot(range(j+1), best)
        #plt.xlabel("Iteration Number")
        #plt.ylabel("Value")
            
# =============================================================================
# This calculateMove() function is where you need to write your code. When it
# is first loaded, it will play a complete game for you using the Helper
# functions that are defined below. The Helper functions give great example
# code that shows you how to manipulate the data you receive and the move
# that you have to return.
#

def calculateMove(gameState):
    print(gameState)
    start = time.perf_counter()
    q = qCABC(gameState)
    q.Go()
    print('time: ', time.perf_counter()-start)
    print('BS:', q.BestSolution, ' + BSV:', q.BestSolutionValue)
    
    r = q.BestSolution
    r = [int(r[i]) for i in range(len(r))]
    move = {'Path': r}  # Sets move to be the random order of cities
    return move
    
