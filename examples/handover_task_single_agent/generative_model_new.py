import numpy as np
#from pymdp import utils
from PyAIF import utils
import copy

# Factorized state space
"""
Full state space is factorized into:
- team_task:        ['sort_safe', 'sort_hazardous']
- slot1_status:     ['empty', 'occupied']
- slot2_status:     ['empty', 'occupied']
- slot3_status:     ['empty', 'occupied']
- end_effector_position : ['slot1', 'slot2', 'slot3', ideal]
- human_command_memory:   ['slot1', 'slot2', 'slot3', 'wait']
- h_command: ['slot1', 'slot2', 'slot3', 'wait']
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
"""
num_states = [2, 2, 2, 2, 4, 4, 4]

# Multimodal observation space
"""
Observations are coming from different sensors such as vision and proprioception:
- (fxed_camera) objs on picking_slot: ['safe', 'hazardous', 'not_clear']
- (fxed_camera) objs on slot1: ['safe', 'hazardous', 'not_clear', 'empty']
- (fxed_camera) objs on slot2: ['safe', 'hazardous', 'not_clear', 'empty']
- (fxed_camera) objs on slot3: ['safe', 'hazardous', 'not_clear', 'empty']
- (voice)       h_commands: ['slot1', 'slot2', 'slot3', 'wait', 'not_clear'] 
- (voice)       h_feedback_commands: ['positive', 'negative', 'not_clear']
- (3D-pose)     end_effector_position: ['slot1', 'slot2', 'slot3', 'ideal']
"""
num_obs = [3, 4, 4, 4, 5, 3, 4]

# Factorized control space
"""
- end_effector_position : ['slot1', 'slot2', 'slot3', 'do_nothing']
"""
num_controls = [1, 4, 4, 4, 4, 1, 1]

control_fac_idx = [4]
Temp_horizon = 4

def create_generative_model():
    
    A = utils.uniform_A_matrix(num_obs, num_states)
    #A_aif = AIF_utils.uniform_A_matrix(num_obs, num_states)

    ################# - (fxed_camera) objs on picking_slot: ['safe', 'hazardous', 'not_clear'] ################
    #(sort_safe:sort_hazardous, slot1_status:, slot2_status:, slot3_status:, end_eff_position-:, h_command_memory-:)
    for i in range(num_states[1]):
        for j in range(num_states[2]):
            for k in range(num_states[3]):
                for l in range(num_states[4]):
                    for m in range(num_states[5]):
                        for n in range(num_states[6]):
                            A[0][:,:,i,j,k,l,m,n] = np.array([[1.0, 0.0], #safe
                                                            [0.0, 1.0], #hazardous
                                                            [0.0, 0.0]]) #not_clear
                        
    ################# - (fxed_camera) objs on slot123: ['safe', 'hazardous', 'not_clear', 'empty'] ################
    #(sort_safe:sort_hazardous, slot1_status:, slot2_status:, slot3_status:, end_eff_position-:, h_command_memory-:)
    
    # slot1 status empty
    for j in range(num_states[2]):
        for k in range(num_states[3]):
            for l in range(num_states[4]):
                for m in range(num_states[5]):
                    for n in range(num_states[6]):
                        A[1][:,:,0,j,k,l,m,n] = np.array([[0.1, 0.1], #safe
                                                        [0.1, 0.1], #hazardous
                                                        [0.1, 0.1], #not_clear
                                                        [0.9, 0.9]])#empty
                    
    # slot status occupied
    for j in range(num_states[2]):
        for k in range(num_states[3]):
            for l in range(num_states[4]):
                for m in range(num_states[5]):
                    for n in range(num_states[6]):
                        A[1][:,:,1,j,k,l,m,n] = np.array([[0.9, 0.9], #safe
                                                        [0.9, 0.9], #hazardous
                                                        [0.1, 0.1], #not_clear
                                                        [0.1, 0.1]])#empty
                    
    # slot2 status empty
    for i in range(num_states[1]):
        for k in range(num_states[3]):
            for l in range(num_states[4]):
                for m in range(num_states[5]):
                    for n in range(num_states[6]):
                        A[2][:,:,i,0,k,l,m,n] = np.array([[0.1, 0.1], #safe
                                                        [0.1, 0.1], #hazardous
                                                        [0.1, 0.1], #not_clear
                                                        [0.9, 0.9]])#empty
                    
    # slot2 status occupied
    for i in range(num_states[1]):
        for k in range(num_states[3]):
            for l in range(num_states[4]):
                for m in range(num_states[5]):
                    for n in range(num_states[6]):
                        A[2][:,:,i,1,k,l,m,n] = np.array([[0.9, 0.9], #safe
                                                        [0.9, 0.9], #hazardous
                                                        [0.1, 0.1], #not_clear
                                                        [0.1, 0.1]])#empty
                    
    # slot3 status empty
    for i in range(num_states[1]):
        for j in range(num_states[2]):
            for l in range(num_states[4]):
                for m in range(num_states[5]):
                    for n in range(num_states[6]):
                        A[3][:,:,i,j,0,l,m,n] = np.array([[0.1, 0.1], #safe
                                                        [0.1, 0.1], #hazardous
                                                        [0.1, 0.1], #not_clear
                                                        [0.9, 0.9]])#empty
                    
    # slot3 status occupied
    for i in range(num_states[1]):
        for j in range(num_states[2]):
            for l in range(num_states[4]):
                for m in range(num_states[5]):
                    for n in range(num_states[6]):
                        A[3][:,:,i,j,1,l,m,n] = np.array([[0.9, 0.9], #safe
                                                        [0.9, 0.9], #hazardous
                                                        [0.1, 0.1], #not_clear
                                                        [0.1, 0.1]])#empty
                    
    ################# - (voice)       h_commands: ['slot1', 'slot2', 'slot3', 'wait', 'not_clear']  ################
    #(sort_safe:sort_hazardous, slot1_status:, slot2_status:, slot3_status:, end_eff_position-:, h_command_memory-:)
    
    for i in range(num_states[1]):
        for j in range(num_states[2]):
            for k in range(num_states[3]):    
                for l in range(num_states[4]):
                    for m in range(num_states[5]):
                        A[4][:,:,i,j,k,l,m,0] = np.array([[0.9, 0.9], #slot1
                                                          [0.1, 0.1], #slot2
                                                          [0.1, 0.1], #slot3
                                                          [0.1, 0.1], #wait
                                                          [0.1, 0.1]])#not_clear    
                        
    for i in range(num_states[1]):
        for j in range(num_states[2]):
            for k in range(num_states[3]):    
                for l in range(num_states[4]):
                    for m in range(num_states[5]):
                        A[4][:,:,i,j,k,l,m,1] = np.array([[0.1, 0.1], #slot1
                                                          [0.9, 0.9], #slot2
                                                          [0.1, 0.1], #slot3
                                                          [0.1, 0.1], #wait
                                                          [0.1, 0.1]])#not_clear
                        
    for i in range(num_states[1]):
        for j in range(num_states[2]):
            for k in range(num_states[3]):    
                for l in range(num_states[4]):
                    for m in range(num_states[5]):
                        A[4][:,:,i,j,k,l,m,2] = np.array([[0.1, 0.1], #slot1
                                                          [0.1, 0.1], #slot2
                                                          [0.9, 0.9], #slot3
                                                          [0.1, 0.1], #wait
                                                          [0.1, 0.1]])#not_clear
                        
    for i in range(num_states[1]):
        for j in range(num_states[2]):
            for k in range(num_states[3]):    
                for l in range(num_states[4]):
                    for m in range(num_states[5]):
                        A[4][:,:,i,j,k,l,m,3] = np.array([[0.1, 0.1], #slot1
                                                          [0.1, 0.1], #slot2
                                                          [0.1, 0.1], #slot3
                                                          [0.9, 0.9], #wait
                                                          [0.1, 0.1]])#not_clear
    
    """
    for l in range(num_states[4]):
        for m in range(num_states[5]):
            A[4][:,:,0,0,0,l,m,0] = np.array([[0.9, 0.1], #slot1
                                              [0.1, 0.1], #slot2
                                              [0.1, 0.1], #slot3
                                              [0.1, 0.1], #not_clear
                                              [0.1, 0.1]])#wait
            
            A[4][:,:,0,0,0,l,m,1] = np.array([[0.1, 0.1], #slot1
                                              [0.1, 0.9], #slot2
                                              [0.1, 0.1], #slot3
                                              [0.1, 0.1], #not_clear
                                              [0.1, 0.1]])#wait
            
            A[4][:,:,0,0,0,l,m,2] = np.array([[0.1, 0.1], #slot1
                                              [0.1, 0.1], #slot2
                                              [0.9, 0.9], #slot3
                                              [0.1, 0.1], #not_clear
                                              [0.1, 0.1]])#wait
            
            A[4][:,:,0,0,0,l,m,3] = np.array([[0.1, 0.1], #slot1
                                              [0.1, 0.1], #slot2
                                              [0.1, 0.1], #slot3
                                              [0.1, 0.1], #not_clear
                                              [0.9, 0.9]])#wait
    
    for l in range(num_states[4]):
        for m in range(num_states[5]):
            A[4][:,:,0,0,1,l,m,0] = np.array([[0.9, 0.1], #slot1
                                              [0.1, 0.1], #slot2
                                              [0.1, 0.1], #slot3
                                              [0.1, 0.1], #not_clear
                                              [0.1, 0.1]])#wait
            
            A[4][:,:,0,0,1,l,m,1] = np.array([[0.1, 0.1], #slot1
                                              [0.1, 0.9], #slot2
                                              [0.1, 0.1], #slot3
                                              [0.1, 0.1], #not_clear
                                              [0.1, 0.1]])#wait
            
            A[4][:,:,0,0,1,l,m,2] = np.array([[0.1, 0.1], #slot1
                                              [0.1, 0.1], #slot2
                                              [0.1, 0.1], #slot3
                                              [0.1, 0.1], #not_clear
                                              [0.1, 0.1]])#wait
            
            A[4][:,:,0,0,1,l,m,3] = np.array([[0.1, 0.1], #slot1
                                              [0.1, 0.9], #slot2
                                              [0.1, 0.1], #slot3
                                              [0.1, 0.1], #not_clear
                                              [0.9, 0.9]])#wait
                        
    for l in range(num_states[4]):
        for m in range(num_states[5]):
            A[4][:,:,0,1,0,l,m] = np.array([[0.9, 0.1], #slot1
                                            [0.1, 0.1], #slot2
                                            [0.9, 0.9], #slot3
                                            [0.1, 0.1]])#wait

    for l in range(num_states[4]):
        for m in range(num_states[5]):
            A[4][:,:,0,1,1,l,m] = np.array([[0.9, 0.1], #slot1
                                            [0.1, 0.1], #slot2
                                            [0.1, 0.1], #slot3
                                            [0.1, 0.9]])#wait
                        
    for l in range(num_states[4]):
        for m in range(num_states[5]):
            A[4][:,:,1,0,0,l,m] = np.array([[0.1, 0.1], #slot1
                                            [0.1, 0.9], #slot2
                                            [0.9, 0.9], #slot3
                                            [0.1, 0.1]])#wait

    for l in range(num_states[4]):
        for m in range(num_states[5]):
            A[4][:,:,1,0,1,l,m] = np.array([[0.1, 0.1], #slot1
                                            [0.1, 0.9], #slot2
                                            [0.1, 0.1], #slot3
                                            [0.9, 0.1]])#wait
                        
    for l in range(num_states[4]):
        for m in range(num_states[5]):
            A[4][:,:,1,1,0,l,m] = np.array([[0.1, 0.1], #slot1
                                            [0.1, 0.1], #slot2
                                            [0.9, 0.9], #slot3
                                            [0.1, 0.1]])#wait
                        
    for l in range(num_states[4]):
        for m in range(num_states[5]):
            A[4][:,:,1,1,1,l,m] = np.array([[0.1, 0.1], #slot1
                                            [0.1, 0.1], #slot2
                                            [0.1, 0.1], #slot3
                                            [0.9, 0.9]])#wait
    """                    
    ################# - (voice)       h_feedback_commands: ['positive', 'negative', 'not_clear'] ################
    #(sort_safe:sort_hazardous, slot1_status:, slot2_status:, slot3_status:, end_eff_position-:, h_command_memory-:)

    for i in range(num_states[1]):
        for j in range(num_states[2]):
            for k in range(num_states[3]):
                for l in range(num_states[4]):
                    for m in range(num_states[5]):
                        for n in range(num_states[6]):
                            A[5][:,:,i,j,k,l,m,n] = np.array([[0.1, 0.1], #positive
                                                            [0.9, 0.9], #negative
                                                            [0.1, 0.1]])
    for i in range(num_states[1]):
        for j in range(num_states[2]):
            for k in range(num_states[3]):
                for n in range(num_states[6]):
                    A[5][:,:,i,j,k,0,0,n] = np.array([[0.9, 0.9], #positive
                                                    [0.1, 0.1], #negative
                                                    [0.1, 0.1]]) #not_clear
    for i in range(num_states[1]):
        for j in range(num_states[2]): 
            for k in range(num_states[3]):
                for n in range(num_states[6]):
                    A[5][:,:,i,j,k,1,1,n] = np.array([[0.9, 0.9], #positive
                                                    [0.1, 0.1], #negative
                                                    [0.1, 0.1]]) #not_clear
                
    for i in range(num_states[1]):
        for j in range(num_states[2]): 
            for k in range(num_states[3]):
                for n in range(num_states[6]):
                    A[5][:,:,i,j,k,2,2,n] = np.array([[0.9, 0.9], #positive
                                                    [0.1, 0.1], #negative
                                                    [0.1, 0.1]]) #not_clear
                
    for i in range(num_states[1]):
        for j in range(num_states[2]): 
            for k in range(num_states[3]):
                for n in range(num_states[6]):
                    A[5][:,:,i,j,k,3,3,n] = np.array([[0.9, 0.9], #positive
                                                    [0.1, 0.1], #negative
                                                    [0.1, 0.1]]) #not_clear
                    
    ################# - (3D-pose)     end_effector_position: ['slot1', 'slot2', 'slot3', 'ideal'] ################
    #(sort_safe:sort_hazardous, slot1_status:, slot2_status:, slot3_status:, end_eff_position-:, h_command_memory-:)

    for i in range(num_states[1]):
        for j in range(num_states[2]):
            for k in range(num_states[3]):
                for m in range(num_states[5]):
                    for n in range(num_states[6]):
                        A[6][:,:,i,j,k,0,m,n] = np.array([[0.9, 0.9], #slot1
                                                        [0.1, 0.1], #slot2
                                                        [0.1, 0.1], #slot3
                                                        [0.1, 0.1]])#ideal 

    for i in range(num_states[1]):
        for j in range(num_states[2]):
            for k in range(num_states[3]):
                for m in range(num_states[5]):
                    for n in range(num_states[6]):
                        A[6][:,:,i,j,k,1,m,n] = np.array([[0.1, 0.1], #slot1
                                                        [0.9, 0.9], #slot2
                                                        [0.1, 0.1], #slot3
                                                        [0.1, 0.1]])#ideal                    
    for i in range(num_states[1]):
        for j in range(num_states[2]):
            for k in range(num_states[3]):
                for m in range(num_states[5]):
                    for n in range(num_states[6]):
                        A[6][:,:,i,j,k,2,m,n] = np.array([[0.1, 0.1], #slot1
                                                        [0.1, 0.1], #slot2
                                                        [0.9, 0.9], #slot3
                                                        [0.1, 0.1]])#ideal

    for i in range(num_states[1]):
        for j in range(num_states[2]):
            for k in range(num_states[3]):
                for m in range(num_states[5]):
                    for n in range(num_states[6]):
                        A[6][:,:,i,j,k,3,m,n] = np.array([[0.1, 0.1], #slot1
                                                        [0.1, 0.1], #slot2
                                                        [0.1, 0.1], #slot3
                                                        [0.9, 0.9]])#ideal
                    
############ - (metacognition) human_command_memory: ['slot1', 'slot2', 'slot3', 'wait'] ############

    """
    for i in range(num_states[1]):
        for j in range(num_states[2]):
            for k in range(num_states[3]):
                for l in range(num_states[4]):
                    A[7][:,:,i,j,k,l,0] = np.array([[0.9, 0.9], #slot1
                                                    [0.1, 0.1], #slot2
                                                    [0.1, 0.1], #slot3
                                                    [0.1, 0.1]])#ideal 

    for i in range(num_states[1]):
        for j in range(num_states[2]):
            for k in range(num_states[3]):
                for l in range(num_states[4]):
                    A[7][:,:,i,j,k,l,1] = np.array([[0.1, 0.1], #slot1
                                                    [0.9, 0.9], #slot2
                                                    [0.1, 0.1], #slot3
                                                    [0.1, 0.1]])#ideal                    
    for i in range(num_states[1]):
        for j in range(num_states[2]):
            for k in range(num_states[3]):
                for l in range(num_states[4]):
                    A[7][:,:,i,j,k,l,2] = np.array([[0.1, 0.1], #slot1
                                                    [0.1, 0.1], #slot2
                                                    [0.9, 0.9], #slot3
                                                    [0.1, 0.1]])#ideal

    for i in range(num_states[1]):
        for j in range(num_states[2]):
            for k in range(num_states[3]):
                for l in range(num_states[4]):
                    A[7][:,:,i,j,k,l,3] = np.array([[0.1, 0.1], #slot1
                                                    [0.1, 0.1], #slot2
                                                    [0.1, 0.1], #slot3
                                                    [0.9, 0.9]])#ideal                    
    """
    B = utils.uniform_B_matrix(num_states, num_controls)
    #B_aif = AIF_utils.uniform_B_matrix(num_states, num_controls)
    ############ control actions - team_task:  ['sort_safe', 'sort_hazardous'] ############
    matrix_22 = np.array([[0.1, 0.1],
                         [0.1, 0.1]])
    matrix_33 = np.array([[0.1, 0.1, 0.1],
                          [0.1, 0.1, 0.1],
                          [0.1, 0.1, 0.1]])
    
    B[0][:,:,0] = matrix_22

    ############ control actions - slot123_status:['empty', 'occupied'] ############
    #action_mappings = {0:'slot1', 1:'slot2', 2:'slot3', 3:'ideal'}
    B[1][:,:,3] = np.array([[1, 0],
                            [0, 1]])
    
    B[1][:,:,0] = np.array([[0, 0],
                            [1, 1]])
    B[1][:,:,1] = np.array([[1, 0],
                            [0, 1]])
    B[1][:,:,2] = np.array([[1, 0],
                            [0, 1]])
        
    B[2][:,:,3] = np.array([[1, 0],
                            [0, 1]])
    B[2][:,:,0] = np.array([[1, 0],
                            [0, 1]])
    B[2][:,:,1] = np.array([[0, 0],
                            [1, 1]])
    B[2][:,:,2] = np.array([[1, 0],
                            [0, 1]])
    
    B[3][:,:,3] = np.array([[1, 0],
                            [0, 1]])
    B[3][:,:,0] = np.array([[1, 0],
                            [0, 1]])
    B[3][:,:,1] = np.array([[1, 0],
                            [0, 1]])
    B[3][:,:,2] = np.array([[0, 0],
                            [1, 1]])

    ############ control actions - end_effector_position : ['slot1', 'slot2', 'slot3', 'ideal'] ############
    # action 0 is 'slot1'
    B[4][:,:,0] = np.array([[1.0, 1.0, 1.0, 1.0],
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0]])

    # action 1 is 'slot2'
    B[4][:,:,1] = np.array([[0.0, 0.0, 0.0, 0.0],
                            [1.0, 1.0, 1.0, 1.0],
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0]])

    # action 2 is 'slot3'
    B[4][:,:,2] = np.array([[0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                            [1.0, 1.0, 1.0, 1.0],
                            [0.0, 0.0, 0.0, 0.0]])

    # action 3 is 'do_nothing'
    B[4][:,:,3] = np.array([[0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                            [1.0, 1.0, 1.0, 1.0]])

    ############ control actions - (metacognition) human_command_memory: ['slot1', 'slot2', 'slot3', 'wait'] ############
    # causal dependency for memory_states
    # columns - human_command_(t-1)
    # rows - human_command_memory(t)
    # action 0 is 'slot1'

    B[5][:,:,0] = np.eye(4)
    B[6][:,:,0] = np.eye(4)
    """
    # action 1 is 'slot2'
    B[5][:,:,1] = np.array([[0.1, 0.1, 0.1, 0.1],
                            [0.15, 0.15, 0.15, 0.15],
                            [0.1, 0.1, 0.1, 0.1],
                            [0.1, 0.1, 0.1, 0.1]])

    # action 2 is 'slot3'
    B[5][:,:,2] = np.array([[0.1, 0.1, 0.1, 0.1],
                            [0.1, 0.1, 0.1, 0.1],
                            [0.15, 0.15, 0.15, 0.15],
                            [0.1, 0.1, 0.1, 0.1]])

    # action 3 is 'do_nothing'
    B[5][:,:,3] = np.array([[0.1, 0.1, 0.1, 0.1],
                            [0.1, 0.1, 0.1, 0.1],
                            [0.1, 0.1, 0.1, 0.1],
                            [0.15, 0.15, 0.15, 0.15]])

    """
                                             #####################

    D = utils.uniform_D_matrix(num_states)
    #D_aif = AIF_utils.uniform_D_matrix(num_states)

    ############ team_task:  ['sort_safe', 'sort_hazardous'] ############
    D[0] = np.array([0.1, 0.1])

    ############ slot123_status: ['empty', 'occupied'] ############
    D[1] = np.array([0.9, 0.1])
    D[2] = np.array([0.9, 0.1])
    D[3] = np.array([0.9, 0.1])

    ############ end_effector_position : ['ideal', 'slot1', 'slot2', 'slot3'] ############
    D[4] = np.array([0.1, 0.1, 0.1, 0.1])

    ############ - (metacognition) human_command_memory: ['slot1', 'slot2', 'slot3', 'wait'] ############
    D[5] = np.array([0.1, 0.1, 0.1, 0.1])
    
                                ##########################

    C = utils.zero_C_matrix(num_obs, Temp_horizon)
    #C_aif = AIF_utils.zero_C_matrix(num_obs, 4)

    ################# - (fxed_camera) objs on picking_slot: ['safe', 'hazardous', 'not_clear'] ################

    C[0] = np.array([[0.0, 0.0, 0.0, 0.0], #safe
                     [0.0, 0.0, 0.0, 0.0], #hazardous
                     [0.0, 0.0, 0.0, 0.0]])#not_clear

    ################# - (fxed_camera) objs on slot1: ['safe', 'hazardous', 'not_clear', 'empty'] ################

    C[1] = np.array([[0.0, 0.0, 0.0, 0.0], #safe
                     [0.0, 0.0, 0.0, 0.0], #hazardous
                     [0.0, 0.0, 0.0, 0.0], #not_clear
                     [0.0, 0.0, 0.0, 0.0]])#empty

    ################# - (fxed_camera) objs on slot2: ['safe', 'hazardous', 'not_clear', 'empty'] ################

    C[2] = np.array([[0.0, 0.0, 0.0, 0.0], #safe
                     [0.0, 0.0, 0.0, 0.0], #hazardous
                     [0.0, 0.0, 0.0, 0.0], #not_clear
                     [0.0, 0.0, 0.0, 0.0]])#empty

    ################# - (fxed_camera) objs on slot3: ['safe', 'hazardous', 'not_clear', 'empty'] ################

    C[3] = np.array([[0.0, 0.0, 0.0, 0.0], #safe
                     [0.0, 0.0, 0.0, 0.0], #hazardous
                     [0.0, 0.0, 0.0, 0.0], #not_clear
                     [0.0, 0.0, 0.0, 0.0]])#empty

    ################# - (voice) h_commands: ['slot1', 'slot2', 'slot3', 'slot4', 'wait'] ################

    C[4] = np.array([[0.0, 0.0, 0.0, 0.0], #slot1
                     [0.0, 0.0, 0.0, 0.0], #slot2
                     [0.0, 0.0, 0.0, 0.0], #slot3
                     [0.0, 0.0, 0.0, 0.0], #wait
                     [0.0, 0.0, 0.0, 0.0]])#not_clear

    ################# - (voice) h_feedback_commands: ['positive', 'negative', 'not_clear'] ################

    C[5] = np.array([[10.0, 10.0, 10.0, 10.0], #positive
                     [-5.0, -5.0, -5.0, -5.0], #negative
                     [0.5, 0.5, 0.5, 0.5]])#not_clear

    ################# - (3D-pose)     end_effector_position: ['ideal', 'slot1', 'slot2', 'slot3', 'slot4'] ################

    C[6] = np.array([[0.0, 0.0, 0.0, 0.0], #ideal
                     [0.0, 0.0, 0.0, 0.0], #slot1
                     [0.0, 0.0, 0.0, 0.0], #slot2
                     [0.0, 0.0, 0.0, 0.0]])#slot3
    
    ################# - (metacognition) human_command_memory: ['slot1', 'slot2', 'slot3', 'wait'] ################
    """
    C[7] = np.array([[0.0, 0.0, 0.0, 0.0], #slot1
                     [0.0, 0.0, 0.0, 0.0], #slot2
                     [0.0, 0.0, 0.0, 0.0], #slot3
                     [0.0, 0.0, 0.0, 0.0]])#wait
    """

    return A, B, C, D, num_states, num_obs, num_controls, control_fac_idx, Temp_horizon
                    
