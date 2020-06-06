#   _____                    _ _ _               ____        _               _                        ____            _     _                
#  |_   _| __ __ ___   _____| | (_)_ __   __ _  / ___|  __ _| | ___  ___  __| |_ __ ___  _ __   ___  |  _ \ _ __ ___ | |__ | | ___ _ __ ___  
#    | || '__/ _` \ \ / / _ \ | | | '_ \ / _` | \___ \ / _` | |/ _ \/ __|/ _` | '__/ _ \| '_ \ / _ \ | |_) | '__/ _ \| '_ \| |/ _ \ '_ ` _ \ 
#    | || | | (_| |\ V /  __/ | | | | | | (_| |  ___) | (_| | |  __/\__ \ (_| | | | (_) | | | |  __/ |  __/| | | (_) | |_) | |  __/ | | | | |
#    |_||_|  \__,_| \_/ \___|_|_|_|_| |_|\__, | |____/ \__,_|_|\___||___/\__,_|_|  \___/|_| |_|\___| |_|   |_|  \___/|_.__/|_|\___|_| |_| |_|
#                                        |___/                                                                                               
botName='davidtang1-defbot'

import random
import json
from random import randint, choice, shuffle
from math import hypot

# These are the only additional libraries available to you. Uncomment them
# to use them in your solution.
#
import numpy    # Base N-dimensional array package
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
    
    city_list = gameState["CityCoords"]
    city_num = len(city_list)

    if (gameState["MyBestSolution"]==None):
        unvisited = [i for i in range(city_num)]
        list_ini = [0]
        del unvisited[0]
        
        for i in range(city_num-1):
            next_p = findClosestCity(city_list, list_ini[i], unvisited)
            list_ini.append(next_p)
            del unvisited[unvisited.index(next_p)]

    else:
        if (gameState["MyBestDistance"]<=gameState["OppBestDistance"]):
            list_ini = gameState["MyBestSolution"]
        else: list_ini = gameState["OppBestSolution"]
    
    s = []
    for i in range(city_num): s.append(list_ini[i])  
    
    T = 10
    L = 20
    
    for k in range(L):
        cost = 0
        for i in range(city_num-1): 
            cost = cost + getDistance(city_list[s[i]], city_list[s[i+1]])
        cost = cost + getDistance(city_list[s[city_num-1]], city_list[s[0]])
        
        print(cost)
        
        s_new = []
        for i in range(city_num): s_new.append(s[i])   
        
        pos_i = numpy.random.randint(low=0, high=city_num-1)
        pos_f = numpy.random.randint(low=0, high=city_num-1)
        
        if (pos_f == pos_i): pos_f = pos_i + 1
        elif (pos_f < pos_i): pos_i, pos_f = pos_f, pos_i
        
        while (pos_i < pos_f):
            s_new[pos_i], s_new[pos_f] = s_new[pos_f], s_new[pos_i]
            pos_i = pos_i + 1
            pos_f = pos_f - 1
        
        cost_new = 0
        for i in range(city_num-1): 
            cost_new = cost_new + getDistance(city_list[s_new[i]], city_list[s_new[i+1]])
        cost_new = cost_new + getDistance(city_list[s_new[city_num-1]], city_list[s_new[0]])
        
        if (cost_new < cost): 
            s = s_new
        else:
            e = numpy.exp(-(cost_new-cost)/T)
            T = T * 0.9
            p = numpy.random.rand()
            if (p <= e): s = s_new
            
    move = {"Path":s}
    return move
    

# Given two city coordinates of the form [x, y] returns the distance between them
def getDistance(origin, destination):
    distance = hypot(abs(origin[0]-destination[0]), abs(origin[1]-destination[1]))
    return distance


# Given the list of city coordinates, the current city index, and a list of available cities indexes to choose from
# calculates the closest city (from the list of available cities) to the current city and returns it
def findClosestCity(coords, cur_city, available_cities):
    closest_city = available_cities[0]  # initialise closest city so far to the first city in the list
    closest_distance = getDistance(coords[cur_city], coords[closest_city])  # Initialise the distance to the closest city as the distance to the first city in the list
    
    for next_city in available_cities[1:]:  # For all remaining cities
        next_distance = getDistance(coords[cur_city], coords[next_city])  # Calculate the distance to it from our current city
        if next_distance < closest_distance:  # If this distance is our new shortest
            closest_distance = next_distance  # Update closest_distance
            closest_city = next_city  # Update closest_city
            
    return closest_city  # Return the closest city we found