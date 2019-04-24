# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 20:06:11 2019

@author: Xuan Li
"""

import numpy as np
from numpy.random import rand
import logging

# define a Uniform Distribution
U = lambda MIN, MAX, SAMPLES: rand(*SAMPLES.shape) * (MAX - MIN) + MIN

# define a Truncated Power Law Distribution
P = lambda ALPHA, MIN, MAX, SAMPLES: ((MAX ** (ALPHA+1.) - 1.) * rand(*SAMPLES.shape) + 1.) ** (1./(ALPHA+1.))

# define an Exponential Distribution
E = lambda SCALE, SAMPLES: -SCALE*np.log(rand(*SAMPLES.shape))

# *************** Palm state probability **********************
def pause_probability_init(pause_low, pause_high, speed_low, speed_high, dimensions):
    alpha1 = ((pause_high+pause_low)*(speed_high-speed_low))/(2*np.log(speed_high/speed_low))
    delta1 = np.sqrt(np.sum(np.square(dimensions)))
    return alpha1/(alpha1+delta1)

# *************** Palm residual ******************************
def residual_time(mean, delta, shape=(1,)):
    t1 = mean - delta;
    t2 = mean + delta;
    u = rand(*shape);
    residual = np.zeros(shape)
    if delta != 0.0:
        case_1_u = u < (2.*t1/(t1+t2))
        residual[case_1_u] = u[case_1_u]*(t1+t2)/2.
        residual[np.logical_not(case_1_u)] = t2-np.sqrt((1.-u[np.logical_not(case_1_u)])*(t2*t2 - t1*t1))
    else:
        residual=u*mean  
    return residual

# *********** Initial speed ***************************
def initial_speed(speed_mean, speed_delta, shape=(1,)):
    v0 = speed_mean - speed_delta
    v1 = speed_mean + speed_delta
    u = rand(*shape)
    return pow(v1, u) / pow(v0, u - 1)

def init_random_waypoint(nr_nodes, dimensions,
                         speed_low, speed_high, pause_low, pause_high):

    ndim = len(dimensions)
    positions = np.empty((nr_nodes, ndim))
    waypoints = np.empty((nr_nodes, ndim))
    speed = np.empty(nr_nodes)
    pause_time = np.empty(nr_nodes)

    speed_low = float(speed_low)
    speed_high = float(speed_high)

    moving = np.ones(nr_nodes)
    speed_mean, speed_delta = (speed_low+speed_high)/2., (speed_high-speed_low)/2.
    pause_mean, pause_delta = (pause_low+pause_high)/2., (pause_high-pause_low)/2.

    # steady-state pause probability for Random Waypoint
    q0 = pause_probability_init(pause_low, pause_high, speed_low, speed_high, dimensions)
    
    for i in range(nr_nodes):
        
        while True:

            z1 = rand(ndim) * np.array(dimensions)
            z2 = rand(ndim) * np.array(dimensions)

            if rand() < q0:
                moving[i] = 0.
                break
            else:
                #r is a ratio of the length of the randomly chosen path over
                # the length of a diagonal across the simulation area
                r = np.sqrt(np.sum((z2 - z1) ** 2) / np.sum(np.array(dimensions) ** 2))
                if rand() < r:
                    moving[i] = 1.
                    break

        positions[i] = z1
        waypoints[i] = z2

    # steady-state positions
    # initially the node has traveled a proportion u2 of the path from (x1,y1) to (x2,y2)
    u2 = rand(*positions.shape)
    positions = u2*positions + (1 - u2)*waypoints

    # steady-state speed and pause time
    paused_bool = moving==0.
    paused_idx = np.where(paused_bool)[0]
    pause_time[paused_idx] = residual_time(pause_mean, pause_delta, paused_idx.shape)
    speed[paused_idx] = 0.0

    moving_bool = np.logical_not(paused_bool)
    moving_idx = np.where(moving_bool)[0]
    pause_time[moving_idx] = 0.0
    speed[moving_idx] = initial_speed(speed_mean,speed_delta, moving_idx.shape)

    return positions, waypoints, speed, pause_time

class RandomWaypoint(object):
    
    def __init__(self, nr_nodes, dimensions, velocity=(0.1, 1.), wt_max=None):
        
        '''
        Random Waypoint model.
        
        Required arguments:
        
          *nr_nodes*:
            Integer, the number of nodes.
          
          *dimensions*:
            Tuple of Integers, the x and y dimensions of the simulation area.
          
        keyword arguments:
        
          *velocity*:
            Tuple of Float, the minimum and maximum values for node velocity.
          
          *wt_max*:
            The maximum wait time for node pauses.
            If wt_max is 0 or None, there is no pause time.
        '''
        
        self.nr_nodes = nr_nodes
        self.dimensions = dimensions
        self.velocity = velocity
        self.wt_max = wt_max
        self.init_stationary = True
    
    def __iter__(self):
        
        ndim = len(self.dimensions)
        MIN_V, MAX_V = self.velocity
        
        wt_min = 0.
        
        if self.init_stationary:

            positions, waypoints, velocity, wt = \
                init_random_waypoint(self.nr_nodes, self.dimensions, MIN_V, MAX_V, wt_min, 
                             (self.wt_max if self.wt_max is not None else 0.))
        else:

            NODES = np.arange(self.nr_nodes)
            positions = U(np.zeros(ndim), np.array(self.dimensions), np.dstack((NODES,)*ndim)[0])
            waypoints = U(np.zeros(ndim), np.array(self.dimensions), np.dstack((NODES,)*ndim)[0])
            wt = np.zeros(self.nr_nodes)
            velocity = U(MIN_V, MAX_V, NODES)

        # assign nodes' movements (direction * node velocity)
        direction = waypoints - positions
        direction /= np.linalg.norm(direction, axis=1)[:, np.newaxis]
        
        while True:
            # update node position
            positions += direction * velocity[:, np.newaxis]
            # calculate distance to waypoint
            d = np.sqrt(np.sum(np.square(waypoints - positions), axis=1))
            # update info for arrived nodes
            arrived = np.where(np.logical_and(d<=velocity, wt<=0.))[0]
            
            # step back for nodes that surpassed waypoint
            positions[arrived] = waypoints[arrived]
            
            if self.wt_max:
                velocity[arrived] = 0.
                wt[arrived] = U(0, self.wt_max, arrived)
                # update info for paused nodes
                wt[np.where(velocity==0.)[0]] -= 1.
                # update info for moving nodes
                arrived = np.where(np.logical_and(velocity==0., wt<0.))[0]
            
            if arrived.size > 0:
                waypoints[arrived] = U(np.zeros(ndim), np.array(self.dimensions), np.zeros((arrived.size, ndim)))
                velocity[arrived] = U(MIN_V, MAX_V, arrived)

                new_direction = waypoints[arrived] - positions[arrived]
                direction[arrived] = new_direction / np.linalg.norm(new_direction, axis=1)[:, np.newaxis]
            
            self.velocity = velocity
            self.wt = wt
            yield positions

def random_waypoint(*args, **kwargs):
    return iter(RandomWaypoint(*args, **kwargs))


if __name__ == '__main__':
    # number of steps to ignore before start plotting
    STEPS_TO_IGNORE = 1000
    logging.basicConfig(format='%(asctime)-15s - %(message)s', level=logging.INFO)
    logger = logging.getLogger("simulation")
    step = 0
    np.random.seed(0xffff)
    ## Random Walk model
    delta_t = 0.1
    rwp = random_waypoint(10, dimensions=(100, 100), velocity=(0.1, 2), wt_max=0)
    f = open('workfile.txt', 'w')
    for xy in rwp:
   
        step += 1
        for i in range(len(xy)):
            f.write(str(xy[i,0]))
            f.write(' ')
            f.write(str(xy[i,1]))
            f.write('\n')
        f.write('\n')
        if step%10000==0: logger.info('Step %s'% step)
        if step < STEPS_TO_IGNORE: continue
        if step==2*STEPS_TO_IGNORE:
            break
    f.close()


