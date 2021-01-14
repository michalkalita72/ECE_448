# geometry.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Jongdeog Lee (jlee700@illinois.edu) on 09/12/2018

"""
This file contains geometry functions that relate with Part1 in MP2.
"""

#Used for geometry calculations
#https://codereview.stackexchange.com/questions/86421/line-segment-to-circle-collision-algorithm

import math
import numpy as np
from const import *

def computeCoordinate(start, length, angle):
    """Compute the end cooridinate based on the given start position, length and angle.

        Args:
            start (tuple): base of the arm link. (x-coordinate, y-coordinate)
            length (int): length of the arm link
            angle (int): degree of the arm link from x-axis to couter-clockwise

        Return:
            End position (int,int):of the arm link, (x-coordinate, y-coordinate)
    """
    return ( start[0]+int(length*math.cos(math.pi*angle/180) ), 
    start[1]-int(length*math.sin(math.pi*angle/180) ) )
    #return (0, 0)

def doesArmTouchObjects(armPosDist, objects, isGoal=False):
    """Determine whether the given arm links touch any obstacle or goal

        Args:
            armPosDist (list): start and end position and padding distance of all
             arm links [(start, end, padding_distance)]
            objects (list): x-, y- coordinate and radius of object (obstacles or 
            goals) [(x, y, r)]
            isGoal (bool): 
             True if the object is a goal and False if the object is an obstacle.
             When the object is an obstacle, consider padding distance.
             When the object is a goal, no need to consider padding distance.
        Return:
            True if touched. False if not.
    """
    # print("PRINT armPosDist: ",armPosDist)
    temp = armPosDist # Get the second arm link(tip) 
    V = subtract_tuple_index_2( temp[0][1], temp[0][0] ) # diff of p1 coordinates to p2 coordinates
    p1_t = temp[0][0] # start of arm link_2
    p1 = list(p1_t) # list of coordinates for link_2

    if isGoal == False:
        for object in objects:
            radius_pad = object[2] + temp[0][2] # radius of circle plus padding distance
            #radius_ = object[2]
            center_t = (object[0], object[1])
            Center = list(center_t)

            a = np.dot( V, V )

            # b = @ * V.dot(P1 - Q) where Q is center of circle
            diff_Base_Center = []
            diff_Base_Center.append(p1[0] - Center[0])
            diff_Base_Center.append(p1[1] - Center[1])
            b = 2 * ( np.dot( V, diff_Base_Center ) )

            c = np.dot( p1,p1 ) + np.dot(Center, Center) - 2 * np.dot(p1,Center) - radius_pad**2
            # calculate the discriminant b2−4ac  
            disc = b**2 - 4 * a * c
            if disc < 0:
                continue

            # two solutions to quadratic equation aka
            sqrt_disc = math.sqrt(disc)
            t1 = (-b + sqrt_disc) / (2 * a)
            t2 = (-b - sqrt_disc) / (2 * a)

            if not (0 <= t1 <= 1 or 0 <= t2 <= 1):
                continue
            else:
                return True
    # else, isGoal = True
    else: 
        for goal in objects:
            #radius_pad = object[2] + temp[0][2] # radius of circle plus padding distance
            radius_ = goal[2]
            center_t = (goal[0], goal[1])
            Center = list(center_t)

            a = np.dot( V, V )

            # b = @ * V.dot(P1 - Q) where Q is center of circle
            diff_Base_Center = []
            diff_Base_Center.append(p1[0] - Center[0])
            diff_Base_Center.append(p1[1] - Center[1])
            b = 2 * ( np.dot( V, diff_Base_Center ) )

            c = np.dot( p1,p1 ) + np.dot(Center, Center) - 2 * np.dot(p1,Center) - radius_**2
            # calculate the discriminant b2−4ac  
            disc = b**2 -4*a*c
            if disc < 0:
                continue

            # two solutions to quadratic equation aka
            sqrt_disc = math.sqrt(disc)
            t1 = (-b + sqrt_disc) / (2 * a)
            t2 = (-b - sqrt_disc) / (2 * a)

            if not (0 <= t1 <= 1 or 0 <= t2 <= 1):
                continue
            else:
                return True

    return False

# Vector Subtraction on 2 tuples of size two and returns a list
# tuple_1: The first operand tuple
# tuple_2: The second operand tuple
# Ret: A list containing a vector from the difference of two parameters
def subtract_tuple_index_2( tuple_1, tuple_2 ):
    tuple_ret = (tuple_1[0] - tuple_2[0], tuple_1[1] - tuple_2[1])
    return list(tuple_ret)


def doesArmTipTouchGoals(armEnd, goals):
    """Determine whether the given arm tip touch goals

        Args:
            armEnd (tuple): the arm tip position, (x-coordinate, y-coordinate)
            goals (list): x-, y- coordinate and radius of goals [(x, y, r)]. 
            There can be more than one goal.
            Note: length of arm tip is 50
        Return:
            True if arm tick touches any goal. False if not.
    """
    for goal in goals:
        position_tip = (armEnd[0] - goal[0])**2 + (armEnd[1] - goal[1])**2
        if(position_tip <= goal[2]**2):
            return True
        else:
            continue
        
    return False


def isArmWithinWindow(armPos, window):
    """Determine whether the given arm stays in the window

        Args:
            armPos (list): start and end positions of all arm links [(start, end)]
            window (tuple): (width, height) of the window

        Return:
            True if all parts are in the window. False if not.
    """
    temp = armPos
    # eg: temp[0][1][0] is the second item in list and the first item of second item
    if not ( 0 <= temp[0][0][0] <= window[0] and 0 <= temp[0][0][1] <= window[1]):
        return False
    elif not ( 0 <= temp[0][1][0] <= window[0] and 0 <= temp[0][1][1] <= window[1]):
        return False
    
    else: 
        return True


if __name__ == '__main__':
    computeCoordinateParameters = [((150, 190),100,20), ((150, 190),100,40), ((150, 190),100,60), ((150, 190),100,160)]
    resultComputeCoordinate = [(243, 156), (226, 126), (200, 104), (57, 156)]
    testRestuls = [computeCoordinate(start, length, angle) for start, length, angle in computeCoordinateParameters]
    print(testRestuls)
    assert testRestuls == resultComputeCoordinate

    testArmPosDists = [((100,100), (135, 110), 4), ((135, 110), (150, 150), 5)]
    testObstacles = [[(120, 100, 5)], [(110, 110, 20)], [(160, 160, 5)], [(130, 105, 10)]]
    resultDoesArmTouchObjects = [
        True, True, False, True, False, True, False, True,
        False, True, False, True, False, False, False, True
    ]

    testResults = []
    for testArmPosDist in testArmPosDists:
        for testObstacle in testObstacles:
            testResults.append(doesArmTouchObjects([testArmPosDist], testObstacle))
            # print(testArmPosDist)
            # print(doesArmTouchObjects([testArmPosDist], testObstacle))

    print("\n")
    for testArmPosDist in testArmPosDists:
        for testObstacle in testObstacles:
            testResults.append(doesArmTouchObjects([testArmPosDist], testObstacle, isGoal=True))
            # print(testArmPosDist)
            # print(doesArmTouchObjects([testArmPosDist], testObstacle, isGoal=True))

    print("MYRESULT: ", testResults)
    print("Solution: ", resultDoesArmTouchObjects)
    assert resultDoesArmTouchObjects == testResults

    testArmEnds = [(100, 100), (95, 95), (90, 90)]
    testGoal = [(100, 100, 10)]
    resultDoesArmTouchGoals = [True, True, False]

    testResults = [doesArmTipTouchGoals(testArmEnd, testGoal) for testArmEnd in testArmEnds]
    assert resultDoesArmTouchGoals == testResults

    testArmPoss = [( (100,100), (135, 110) ), ( (135, 110), (150, 150) )]
    testWindows = [(160, 130), (130, 170), (200, 200)]
    resultIsArmWithinWindow = [True, False, True, False, False, True]
    testResults = []
    for testArmPos in testArmPoss:
        for testWindow in testWindows:
            testResults.append(isArmWithinWindow([testArmPos], testWindow))
    assert resultIsArmWithinWindow == testResults

    print("Test passed\n")
