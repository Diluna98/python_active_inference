import numpy as np
from PyAIF import utils, ActiveInfAgent
from environment import GridEnvironment
import matplotlib.pyplot as plt
import time

"""
## Hidden state factors ##

Current_Resolution: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
Resource_Demand: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
"""

"""
## observation modalities ##

model_complexity: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
model_accuracy: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
computaional_cost: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
"""

"""
## Controllable hidden state factors ##

Current_Resolution: stay (Exit), decrease, increase
"""

def create_generative_modelx():

    num_states = [10, 10]
    num_obs = [10, 10, 10]
    num_controls = [3, 1]
    control_fac_idx = [0]
    Temp_horizon = 1
    
    A = utils.uniform_A_matrix(num_obs, num_states)

    ################# - negative_model_evidence: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 ################
    #Current_Resolution:          0,    1,  2,   3,   4,   5,   6,   7,   8,   9
    for i in range(num_states[1]):
        A[0][:,:,i] = np.array([[1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], #0
                                [0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], #1 
                                [0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], #2
                                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], #3    
                                [0.0, 0.0, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], #4
                                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], #5
                                [0.0, 0.0, 0.0, 0.1, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0], #6
                                [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.1, 0.1, 0.1, 0.0], #7
                                [0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 1.0, 1.0, 1.0, 0.0], #8
                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.1, 1.0]])#9 
        
    ################# - Inference latency: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 ################
    #Current_Resolution:          0,    1,  2,   3,   4,   5,   6,   7,   8,   9
    for i in range(num_states[1]):
        A[1][:,:,i] = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], #0
                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], #1 
                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], #2
                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], #3    
                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], #4
                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], #5
                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], #6
                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], #7
                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], #8
                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])#9 
