import numpy as np
from pymdp import utils
from AIF
import copy

"""
sort_safe - slot1, slot3
sort_hazardous - slot2, slot3
"""

# Factorized state space
"""
Full state space is factorized into:
- team_task:        ['sort_safe', 'sort_hazardous']
- safe_slots:       ['slot12', 'slot13', 'slot23']
- hazardous_slots:  ['slot12', 'slot13', 'slot23']
- slot1_status:     ['empty', 'occupied']
- slot2_status:     ['empty', 'occupied']
- slot3_status:     ['empty', 'occupied']
- h_trustworthines:['reliable', 'unreliable']
- end_effector_position : [ideal, 'slot1', 'slot2', 'slot3']
- action_outcome:   ['correct_choice', 'incorrect_mapping_choice', 'occupied_slot_choice']
- obeyed: ['obeyed', 'not_obeyed']                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
"""
num_states = [2, 3, 3, 2, 2, 2, 2, 4, 3, 2]

# Multimodal observation space
"""
Observations are coming from different sensors such as vision and proprioception:
- (fxed_camera) objs on picking_slot: ['safe', 'hazardous', 'not_clear']
- (fxed_camera) objs on slot1: ['safe', 'hazardous', 'not_clear', 'empty']
- (fxed_camera) objs on slot2: ['safe', 'hazardous', 'not_clear', 'empty']
- (fxed_camera) objs on slot3: ['safe', 'hazardous', 'not_clear', 'empty']
- (voice)       h_commands: ['slot1', 'slot2', 'slot3', 'wait'] 
- (voice)       h_feedback_commands: ['positive', 'negative', 'not_clear']
- (3D-pose)     end_effector_position: ['ideal','slot1', 'slot2', 'slot3']
- (metacognition) obedience: ['obyed', 'not_obeyed']
"""
num_obs = [3, 4, 4, 4, 4, 3, 4, 2]

# Factorized control space
"""
- end_effector_position : ['do_nothing', 'slot1', 'slot2', 'slot3']
"""
num_controls = [1, 1, 1, 1, 1, 1, 1, 4, 1, 1]

def _likelihoods_for_h_reliability(indices):
    # Define slot pairs and their order
    slot_pairs = ['12', '13', '23']
    
    # Initialize a 4x2 matrix with 0.1
    matrix = np.full((4, 2), 0.1)
    
    # Get the slot occupancy status from the input indices
    slot_status = indices[2:5] # [0 for empty 1 for occupied for slot1, slot2, slot3]

    # Fill the matrix column-wise based on indices
    for col_idx, pair_idx in enumerate(indices[:2]):
        pair = slot_pairs[pair_idx]  # Get the slot pair (e.g., '12')
        slots = [int(slot) - 1 for slot in pair]  # Convert to 0-based row indices
        
        # Get the occupancy status for the two slots in the current pair
        slot1_occupied = slot_status[slots[0]] == 1
        slot2_occupied = slot_status[slots[1]] == 1
        
        # Check if a slot is empty and assign 0.9
        if not slot1_occupied:
            matrix[slots[0], col_idx] = 0.9
        if not slot2_occupied:
            matrix[slots[1], col_idx] = 0.9
        
        # If both slots in the pair are occupied, assign 0.9 to the overflow slot
        if slot1_occupied and slot2_occupied:
            matrix[3, col_idx] = 0.9
            
    return matrix

def _likelihoods_for_h_unreliability(indices):
    # Define slot pairs and their order
    slot_pairs = ['12', '13', '23']
    
    # Initialize a 4x2 matrix with 0.1
    matrix = np.full((4, 2), 0.9)
    
    # Get the slot occupancy status from the input indices
    slot_status = indices[2:5] # [0 for empty 1 for occupied for slot1, slot2, slot3]

    # Fill the matrix column-wise based on indices
    for col_idx, pair_idx in enumerate(indices[:2]):
        pair = slot_pairs[pair_idx]  # Get the slot pair (e.g., '12')
        slots = [int(slot) - 1 for slot in pair]  # Convert to 0-based row indices
        
        # Get the occupancy status for the two slots in the current pair
        slot1_occupied = slot_status[slots[0]] == 1
        slot2_occupied = slot_status[slots[1]] == 1
        
        # Check if a slot is empty and assign 0.1
        if not slot1_occupied:
            matrix[slots[0], col_idx] = 0.1
        if not slot2_occupied:
            matrix[slots[1], col_idx] = 0.1
        
        # If both slots in the pair are occupied, assign 0.1 to the overflow slot
        if slot1_occupied and slot2_occupied:
            matrix[3, col_idx] = 0.1
            
    return matrix

def create_generative_model():
    
    A = utils.uniform_A_matrix(num_obs, num_states)

    ################# - (fxed_camera) objs on picking_slot: ['safe', 'hazardous', 'not_clear'] ################
    #(sort_safe:sort_hazardous, s_slots-:, h_slots-:, slot1_status:, slot2_status:, slot3_status:, h_trustworthines-:, end_eff_position-:, action_outcome-:)
    for i in range(num_states[1]):
        for j in range(num_states[2]):
            for k in range(num_states[3]):
                for l in range(num_states[4]):
                    for m in range(num_states[5]):
                        for n in range(num_states[6]):
                            for o in range(num_states[7]):
                                for p in range(num_states[8]):
                                    for q in range(num_states[9]):
                                        A[0][:,:,i,j,k,l,m,n,o,p,q] = np.array([[1.0, 0.0], #safe
                                                                            [0.0, 1.0], #hazardous
                                                                            [0.0, 0.0]]) #not_clear
                                
    ################# - (fxed_camera) objs on slot1: ['safe', 'hazardous', 'not_clear', 'empty'] ################
    #(sort_safe:sort_hazardous, s_slots-:, h_slots-:, slot1_status:, slot2_status:, slot3_status:, h_trustworthines-:, end_eff_position-:, action_outcome-:)
    # slot1 status empty
    for i in range(num_states[1]):
        for j in range(num_states[2]):
            for l in range(num_states[4]):
                for m in range(num_states[5]):
                    for n in range(num_states[6]):
                        for o in range(num_states[7]):
                            for p in range(num_states[8]):
                                for q in range(num_states[9]):
                                    A[1][:,:,i,j,0,l,m,n,o,p,q] = np.array([[0.1, 0.1], #safe
                                                                        [0.1, 0.1], #hazardous
                                                                        [0.1, 0.1], #not_clear
                                                                        [0.9, 0.9]])#empty
                            
    #(sort_safe:sort_hazardous, s_slots-:, h_slots-:, slot1_status:, slot2_status:, slot3_status:, h_trustworthines-:, end_eff_position-:, action_outcome-:)
    # *slot combinations with slot1 and slot1 status occupied
    for i in [0, 1]:
        for j in [0, 1]:
            for l in range(num_states[4]):
                for m in range(num_states[5]):
                    for n in range(num_states[6]):
                        for o in range(num_states[7]):
                            for p in range(num_states[8]):
                                for q in range(num_states[9]):
                                    A[1][:,:,i,j,1,l,m,n,o,p,q] = np.array([[0.9, 0.9], #safe
                                                                        [0.9, 0.9], #hazardous
                                                                        [0.1, 0.1], #not_clear
                                                                        [0.1, 0.1]])#empty
                                
    # *slot combinations with slot1 and slot1 status occupied
    for i in [2]:
        for j in [0, 1]:
            for l in range(num_states[4]):
                for m in range(num_states[5]):
                    for n in range(num_states[6]):
                        for o in range(num_states[7]):
                            for p in range(num_states[8]):
                                for q in range(num_states[9]):
                                    A[1][:,:,i,j,1,l,m,n,o,p,q] = np.array([[0.1, 0.1], #safe
                                                                        [0.9, 0.9], #hazardous
                                                                        [0.1, 0.1], #not_clear
                                                                        [0.1, 0.1]])#empty
                                
    # *slot combinations with slot1 and slot1 status occupied
    for i in [0,1]:
        for j in [2]:
            for l in range(num_states[4]):
                for m in range(num_states[5]):
                    for n in range(num_states[6]):
                        for o in range(num_states[7]):
                            for p in range(num_states[8]):
                                for q in range(num_states[9]):
                                    A[1][:,:,i,j,1,l,m,n,o,p,q] = np.array([[0.9, 0.9], #safe
                                                                        [0.1, 0.1], #hazardous
                                                                        [0.1, 0.1], #not_clear
                                                                        [0.1, 0.1]])#empty
                                
    for i in [2]:
        for j in [2]:
            for l in range(num_states[4]):
                for m in range(num_states[5]):
                    for n in range(num_states[6]):
                        for o in range(num_states[7]):
                            for p in range(num_states[8]):
                                for q in range(num_states[9]):
                                    A[1][:,:,i,j,1,l,m,n,o,p,q] = np.array([[0.1, 0.1], #safe
                                                                        [0.1, 0.1], #hazardous
                                                                        [0.1, 0.1], #not_clear
                                                                        [0.1, 0.1]])#empty
                                
    ################# - (fxed_camera) objs on slot2: ['safe', 'hazardous', 'not_clear', 'empty'] ################
    #(sort_safe:sort_hazardous, s_slots-:, h_slots-:, slot1_status:, slot2_status:, slot3_status:, h_trustworthines-:, end_eff_position-:, action_outcome-:)
    # slot2 status empty
    for i in range(num_states[1]):
        for j in range(num_states[2]):
            for k in range(num_states[3]):
                for m in range(num_states[5]):
                    for n in range(num_states[6]):
                        for o in range(num_states[7]):
                            for p in range(num_states[8]):
                                for q in range(num_states[9]):
                                    A[2][:,:,i,j,k,0,m,n,o,p,q] = np.array([[0.1, 0.1], #safe
                                                                        [0.1, 0.1], #hazardous
                                                                        [0.1, 0.1], #not_clear
                                                                        [0.9, 0.9]])#empty    
    """
    # *slot combinations with slot2 and slot2 status empty
    for i in [0, 2]:
        for j in [0, 2]:
            for k in range(num_states[3]):
                for m in range(num_states[5]):
                    for n in range(num_states[6]):
                        for o in range(num_states[7]):
                            for p in range(num_states[8]):
                                A[2][:,:,i,j,k,0,m,n,o,p] = np.array([[0.1, 0.1], #safe
                                                                      [0.1, 0.1], #hazardous
                                                                      [0.1, 0.1], #not_clear
                                                                      [0.9, 0.9]])#empty
    """                            
    #(sort_safe:sort_hazardous, s_slots-:, h_slots-:, slot1_status:, slot2_status:, slot3_status:, h_trustworthines-:, end_eff_position-:, action_outcome-:)
    # *slot combinations with slot2 and slot2 status occupied
    for i in [0, 2]:
        for j in [0, 2]:
            for k in range(num_states[3]):
                for m in range(num_states[5]):
                    for n in range(num_states[6]):
                        for o in range(num_states[7]):
                            for p in range(num_states[8]):
                                for q in range(num_states[9]):
                                    A[2][:,:,i,j,k,1,m,n,o,p,q] = np.array([[0.9, 0.9], #safe
                                                                        [0.9, 0.9], #hazardous
                                                                        [0.1, 0.1], #not_clear
                                                                        [0.1, 0.1]])#empty
                                
    # *slot combinations with slot2 and slot2 status occupied
    for i in [0, 2]:
        for j in [1]:
            for k in range(num_states[3]):
                for m in range(num_states[5]):
                    for n in range(num_states[6]):
                        for o in range(num_states[7]):
                            for p in range(num_states[8]):
                                for q in range(num_states[9]):
                                    A[2][:,:,i,j,k,1,m,n,o,p,q] = np.array([[0.9, 0.9], #safe
                                                                        [0.1, 0.1], #hazardous
                                                                        [0.1, 0.1], #not_clear
                                                                        [0.1, 0.1]])#empty
                                
    # *slot combinations with slot2 and slot2 status occupied
    for i in [1]:
        for j in [0, 2]:
            for k in range(num_states[3]):
                for m in range(num_states[5]):
                    for n in range(num_states[6]):
                        for o in range(num_states[7]):
                            for p in range(num_states[8]):
                                for q in range(num_states[9]):
                                    A[2][:,:,i,j,k,1,m,n,o,p,q] = np.array([[0.1, 0.1], #safe
                                                                        [0.9, 0.9], #hazardous
                                                                        [0.1, 0.1], #not_clear
                                                                        [0.1, 0.1]])#empty
                                
    ################# - (fxed_camera) objs on slot3: ['safe', 'hazardous', 'not_clear', 'empty'] ################
    #(sort_safe:sort_hazardous, s_slots-:, h_slots-:, slot1_status:, slot2_status:, slot3_status:, h_trustworthines-:, end_eff_position-:, action_outcome-:)
    # slot3 status empty
    for i in range(num_states[1]):
        for j in range(num_states[2]):
            for k in range(num_states[3]):
                for l in range(num_states[4]):
                    for n in range(num_states[6]):
                        for o in range(num_states[7]):
                            for p in range(num_states[8]):
                                for q in range(num_states[9]):
                                    A[3][:,:,i,j,k,l,0,n,o,p,q] = np.array([[0.1, 0.1], #safe
                                                                        [0.1, 0.1], #hazardous
                                                                        [0.1, 0.1], #not_clear
                                                                        [0.9, 0.9]])#empty    
    """
    # *slot combinations with slot3 and slot3 status empty
    for i in [1, 2]:
        for j in [1, 2]:
            for k in range(num_states[3]):
                for l in range(num_states[4]):
                    for n in range(num_states[6]):
                        for o in range(num_states[7]):
                            for p in range(num_states[8]):
                                A[3][:,:,i,j,k,l,0,n,o,p] = np.array([[0.1, 0.1], #safe
                                                                      [0.1, 0.1], #hazardous
                                                                      [0.1, 0.1], #not_clear
                                                                      [0.9, 0.9]])#empty
    """                            
    #(sort_safe:sort_hazardous, s_slots-:, h_slots-:, slot1_status:, slot2_status:, slot3_status:, h_trustworthines-:, end_eff_position-:, action_outcome-:)
    # *slot combinations with slot3 and slot3 status occupied
    for i in [1, 2]:
        for j in [1, 2]:
            for k in range(num_states[3]):
                for l in range(num_states[4]):
                    for n in range(num_states[6]):
                        for o in range(num_states[7]):
                            for p in range(num_states[8]):
                                for q in range(num_states[9]):
                                    A[3][:,:,i,j,k,l,1,n,o,p,q] = np.array([[0.9, 0.9], #safe
                                                                        [0.9, 0.9], #hazardous
                                                                        [0.1, 0.1], #not_clear
                                                                        [0.1, 0.1]])#empty
                                
    # *slot combinations with slot3 and slot3 status occupied
    for i in [0]:
        for j in [1, 2]:
            for k in range(num_states[3]):
                for l in range(num_states[4]):
                    for n in range(num_states[6]):
                        for o in range(num_states[7]):
                            for p in range(num_states[8]):
                                for q in range(num_states[9]):
                                    A[3][:,:,i,j,k,l,1,n,o,p,q] = np.array([[0.1, 0.1], #safe
                                                                        [0.9, 0.9], #hazardous
                                                                        [0.1, 0.1], #not_clear
                                                                        [0.1, 0.1]])#empty
                                
    # *slot combinations with slot3 and slot3 status occupied
    for i in [1,2]:
        for j in [0]:
            for k in range(num_states[3]):
                for l in range(num_states[4]):
                    for n in range(num_states[6]):
                        for o in range(num_states[7]):
                            for p in range(num_states[8]):
                                for q in range(num_states[9]):
                                    A[3][:,:,i,j,k,l,1,n,o,p,q] = np.array([[0.9, 0.9], #safe
                                                                        [0.1, 0.1], #hazardous
                                                                        [0.1, 0.1], #not_clear
                                                                        [0.1, 0.1]])#empty

    ################# - (voice)       h_commands: ['slot1', 'slot2', 'slot3'] ################                        
    #(sort_safe:sort_hazardous, s_slots-:, h_slots-:, slot1_status:, slot2_status:, slot3_status:, h_trustworthines-:, end_eff_position-:, action_outcome-:)
    for i in range(num_states[1]):
        for j in range(num_states[2]):
            for k in range(num_states[3]):
                for l in range(num_states[4]):
                    for m in range(num_states[5]):
                        matrix = _likelihoods_for_h_reliability((i,j,k,l,m))
                        for o in range(num_states[7]):
                            for p in range(num_states[8]):
                                for q in range(num_states[9]):
                                    A[4][:,:,i,j,k,l,m,0,o,p,q] = matrix.copy()    

    #(sort_safe:sort_hazardous, s_slots-:, h_slots-:, slot1_status:, slot2_status:, slot3_status:, h_trustworthines-:, end_eff_position-:, action_outcome-:)
    for i in range(num_states[1]):
        for j in range(num_states[2]):
            for k in range(num_states[3]):
                for l in range(num_states[4]):
                    for m in range(num_states[5]):
                        matrix = _likelihoods_for_h_unreliability((i,j,k,l,m))
                        for o in range(num_states[7]):
                            for p in range(num_states[8]):
                                for q in range(num_states[9]):
                                    A[4][:,:,i,j,k,l,m,1,o,p,q] = matrix.copy()

                              
           
    ################# - (voice)       h_feedback_commands: ['positive', 'negative', 'not_clear'] ################
    ################## action-outcome: correct_choice ##################

    #(sort_safe:sort_hazardous, s_slots-:, h_slots-:, slot1_status:, slot2_status:, slot3_status:, h_trustworthines-:, end_eff_position-:, action_outcome-:)
    for i in range(num_states[1]):
        for j in range(num_states[2]):
            for k in range(num_states[3]):
                for l in range(num_states[4]):
                    for m in range(num_states[5]):
                        for n in range(num_states[6]):
                            for o in range(num_states[7]):
                                for p in range(num_states[8]):
                                    for q in range(num_states[9]):
                                        A[5][:,:,i,j,k,l,m,n,o,0,q] = np.array([[0.2, 0.2], #positive
                                                                            [0.1, 0.1], #negative
                                                                            [0.1, 0.1]])#not_clear

    #(sort_safe:sort_hazardous, s_slots-0, h_slots-0, slot1_status-1, slot2_status-1, slot3_status-1, h_trustworthines-:, end_eff_position-0, action_outcome-0)
    for i in [0,1]:
        for j in [0,1]:
            for l in range(num_states[4]):
                for m in range(num_states[5]):
                    for n in range(num_states[6]):
                        for q in range(num_states[9]):
                            A[5][:,:,i,j,1,l,m,n,0,0,q] = np.array([[0.9, 0.9], #positiv
                                                                [0.1, 0.1], #negative
                                                                [0.1, 0.1]])#not_clear 
                        
    #(sort_safe:sort_hazardous, s_slots-0, h_slots-0, slot1_status-1, slot2_status-1, slot3_status-1, h_trustworthines-:, end_eff_position-0, action_outcome-0)
    for i in [0,2]:
        for j in [0,2]:
            for k in range(num_states[3]):
                for m in range(num_states[5]):
                    for n in range(num_states[6]):
                        for q in range(num_states[9]):
                            A[5][:,:,i,j,k,1,m,n,1,0,q] = np.array([[0.9, 0.9], #positiv
                                                                [0.1, 0.1], #negative
                                                                [0.1, 0.1]])#not_clear
                        
    #(sort_safe:sort_hazardous, s_slots-0, h_slots-0, slot1_status-1, slot2_status-1, slot3_status-1, h_trustworthines-:, end_eff_position-0, action_outcome-0)
    for i in [1,2]:
        for j in [1,2]:
            for k in range(num_states[3]):
                for l in range(num_states[4]):
                    for n in range(num_states[6]):
                        for q in range(num_states[9]):
                            A[5][:,:,i,j,k,l,1,n,2,0,q] = np.array([[0.9, 0.9], #positiv
                                                                [0.1, 0.1], #negative
                                                                [0.1, 0.1]])#not_clear
                        
    #(sort_safe:sort_hazardous, s_slots-0, h_slots-0, slot1_status-1, slot2_status-1, slot3_status-1, h_trustworthines-:, end_eff_position-ideal, action_outcome-0)
    for i in range(num_states[1]):
        for j in range(num_states[2]):
            for n in range(num_states[6]):
                for q in range(num_states[9]):
                    A[5][:,:,i,j,1,1,1,n,3,0,q] = np.array([[0.9, 0.9], #positive
                                                            [0.1, 0.1], #negative
                                                            [0.1, 0.1]])#not_clear
                    
    #(sort_safe:sort_hazardous, s_slots-0, h_slots-0, slot1_status-1, slot2_status-1, slot3_status-1, h_trustworthines-:, end_eff_position-ideal, action_outcome-0)
    for n in range(num_states[6]):
        for q in range(num_states[9]):
            A[5][:,:,0,0,1,1,0,n,3,0,q] = np.array([[0.9, 0.9], #positive
                                                    [0.1, 0.1], #negative
                                                    [0.1, 0.1]])#not_clear        
    for i in [1,2]:
        for j in [1,2]:
            for n in range(num_states[6]):
                for q in range(num_states[9]):
                    A[5][:,:,i,j,1,1,0,n,3,0,q] = np.array([[0.2, 0.2], #positive
                                                            [0.1, 0.1], #negative
                                                            [0.1, 0.1]])#not_clear

    for n in range(num_states[6]):
        for q in range(num_states[9]):
            A[5][:,:,1,1,1,0,1,n,3,0,q] = np.array([[0.9, 0.9], #positive
                                                    [0.1, 0.1], #negative
                                                    [0.1, 0.1]])#not_clear
        
    for i in [0,2]:
        for j in [0,2]:
            for n in range(num_states[6]):
                for q in range(num_states[9]):
                    A[5][:,:,i,j,1,1,0,n,3,0,q] = np.array([[0.2, 0.2], #positive
                                                            [0.1, 0.1], #negative
                                                            [0.1, 0.1]])#not_clear
        
    for n in range(num_states[6]):
        for q in range(num_states[9]):
            A[5][:,:,2,2,0,1,1,n,3,0,q] = np.array([[0.9, 0.9], #positive
                                                [0.1, 0.1], #negative
                                                [0.1, 0.1]])#not_clear

    for i in [0,1]:
        for j in [0,1]:
            for n in range(num_states[6]):
                for q in range(num_states[9]):
                    A[5][:,:,i,j,1,1,0,n,3,0,q] = np.array([[0.2, 0.2], #positive
                                                            [0.1, 0.1], #negative
                                                            [0.1, 0.1]])#not_clear                        

                          ################## action-outcome: incorrect_mapping_choice ##################
    #(sort_safe:sort_hazardous, s_slots-:, h_slots-:, slot1_status:, slot2_status:, slot3_status:, h_trustworthines-:, end_eff_position-:, action_outcome-:)
    for i in range(num_states[1]):
        for j in range(num_states[2]):
            for k in range(num_states[3]):
                for l in range(num_states[4]):
                    for m in range(num_states[5]):
                        for n in range(num_states[6]):
                            for o in range(num_states[7]):
                                for p in range(num_states[8]):
                                    for q in range(num_states[9]):
                                        A[5][:,:,i,j,k,l,m,n,o,1,q] = np.array([[0.1, 0.1], #positive
                                                                            [0.2, 0.2], #negative
                                                                            [0.1, 0.1]])#not_clear

    #(sort_safe:sort_hazardous, s_slots-0, h_slots-0, slot1_status-1, slot2_status-1, slot3_status-1, h_trustworthines-:, end_eff_position-0, action_outcome-0)
    for i in range(num_states[1]):
        for j in range(num_states[2]):
            for l in range(num_states[4]):
                for m in range(num_states[5]):
                    for n in range(num_states[6]):
                        for q in range(num_states[9]):
                            A[5][:,:,i,j,0,l,m,n,0,1,q] = np.array([[0.1, 0.1], #positiv
                                                                [0.9, 0.9], #negative
                                                                [0.1, 0.1]])#not_clear
    for i in range(num_states[1]):
        for j in range(num_states[2]):
            for l in range(num_states[4]):
                for m in range(num_states[5]):
                    for n in range(num_states[6]):
                        for q in range(num_states[9]):
                            A[5][:,:,i,j,0,l,m,n,1,1,q] = np.array([[0.1, 0.1], #positiv
                                                                [0.9, 0.9], #negative
                                                                [0.1, 0.1]])#not_clear
    
    #(sort_safe:sort_hazardous, s_slots-:, h_slots-:, slot1_status:, slot2_status:, slot3_status:, h_trustworthines-:, end_eff_position-:, action_outcome-incorrect_mapping_choice)
    for i in [0,1]:
        for j in [0,1]:
            for l in range(num_states[4]):
                for m in range(num_states[5]):
                    for n in range(num_states[6]):
                        for q in range(num_states[9]):
                            A[5][:,:,i,j,0,l,m,n,0,1,q] = np.array([[0.1, 0.1], #positiv
                                                                [0.9, 0.9], #negative
                                                                [0.1, 0.1]])#not_clear
    for i in range(num_states[1]):
        for j in range(num_states[2]):
            for k in range(num_states[3]):
                for m in range(num_states[5]):
                    for n in range(num_states[6]):
                        for q in range(num_states[9]):
                            A[5][:,:,i,j,k,0,m,n,1,1,q] = np.array([[0.1, 0.1], #positiv
                                                                [0.2, 0.2], #negative
                                                                [0.1, 0.1]])#not_clear 
                        
    #(sort_safe:sort_hazardous, s_slots-0, h_slots-0, slot1_status-1, slot2_status-1, slot3_status-1, h_trustworthines-:, end_eff_position-0, action_outcome-0)
    for i in [0,2]:
        for j in [0,2]:
            for k in range(num_states[3]):
                for m in range(num_states[5]):
                    for n in range(num_states[6]):
                        for q in range(num_states[9]):
                            A[5][:,:,i,j,k,0,m,n,1,1,q] = np.array([[0.1, 0.1], #positiv
                                                                [0.9, 0.9], #negative
                                                                [0.1, 0.1]])#not_clear
                        
    for i in range(num_states[1]):
        for j in range(num_states[2]):
            for k in range(num_states[3]):
                for l in range(num_states[4]):
                    for n in range(num_states[6]):
                        for q in range(num_states[9]):
                            A[5][:,:,i,j,k,l,0,n,2,1,q] = np.array([[0.1, 0.1], #positiv
                                                                [0.2, 0.2], #negative
                                                                [0.1, 0.1]])#not_clear
                        
    #(sort_safe:sort_hazardous, s_slots-0, h_slots-0, slot1_status-1, slot2_status-1, slot3_status-1, h_trustworthines-:, end_eff_position-0, action_outcome-0)
    for i in [1,2]:
        for j in [1,2]:
            for k in range(num_states[3]):
                for l in range(num_states[4]):
                    for n in range(num_states[6]):
                        for q in range(num_states[9]):
                            A[5][:,:,i,j,k,l,0,n,2,1,q] = np.array([[0.1, 0.1], #positiv
                                                                [0.9, 0.9], #negative
                                                                [0.1, 0.1]])#not_clear
                        
    #(sort_safe:sort_hazardous, s_slots-0, h_slots-0, slot1_status-1, slot2_status-1, slot3_status-1, h_trustworthines-:, end_eff_position-ideal, action_outcome-0)
    for i in range(num_states[1]):
        for j in range(num_states[2]):
            for n in range(num_states[6]):
                for q in range(num_states[9]):
                    A[5][:,:,i,j,0,0,0,n,3,1,q] = np.array([[0.1, 0.1], #positive
                                                            [0.9, 0.9], #negative
                                                            [0.1, 0.1]])#not_clear
                    
    #(sort_safe:sort_hazardous, s_slots-0, h_slots-0, slot1_status-1, slot2_status-1, slot3_status-1, h_trustworthines-:, end_eff_position-ideal, action_outcome-0)
    for n in range(num_states[6]):
        for q in range(num_states[9]):
            A[5][:,:,0,0,0,0,1,n,3,1,q] = np.array([[0.1, 0.1], #positive
                                                    [0.9, 0.9], #negative
                                                    [0.1, 0.1]])#not_clear        
    for i in [1,2]:
        for j in [1,2]:
            for n in range(num_states[6]):
                for q in range(num_states[9]):
                    A[5][:,:,i,j,0,0,1,n,3,1,q] = np.array([[0.1, 0.1], #positive
                                                            [0.2, 0.2], #negative
                                                            [0.1, 0.1]])#not_clear

    for n in range(num_states[6]):
        for q in range(num_states[9]):
            A[5][:,:,1,1,0,1,0,n,3,1,q] = np.array([[0.1, 0.1], #positive
                                                    [0.9, 0.9], #negative
                                                    [0.1, 0.1]])#not_clear
        
    for i in [0,2]:
        for j in [0,2]:
            for n in range(num_states[6]):
                for q in range(num_states[9]):
                    A[5][:,:,i,j,0,0,1,n,3,1,q] = np.array([[0.1, 0.1], #positive
                                                            [0.2, 0.2], #negative
                                                            [0.1, 0.1]])#not_clear
        
    for n in range(num_states[6]):
        for q in range(num_states[9]):
            A[5][:,:,2,2,1,0,0,n,3,1,q] = np.array([[0.1, 0.1], #positive
                                                [0.9, 0.9], #negative
                                                [0.1, 0.1]])#not_clear

    for i in [0,1]:
        for j in [0,1]:
            for n in range(num_states[6]):
                for q in range(num_states[9]):
                    A[5][:,:,i,j,0,0,1,n,3,1,q] = np.array([[0.1, 0.1], #positive
                                                            [0.2, 0.2], #negative
                                                            [0.1, 0.1]])#not_clear 

                              ################## action-outcome: occupied_slot_choice ##################

    #(sort_safe:sort_hazardous, s_slots-:, h_slots-:, slot1_status:, slot2_status:, slot3_status:, h_trustworthines-:, end_eff_position-:, action_outcome-:)
    for i in range(num_states[1]):
        for j in range(num_states[2]):
            for k in range(num_states[3]):
                for l in range(num_states[4]):
                    for m in range(num_states[5]):
                        for n in range(num_states[6]):
                            for o in range(num_states[7]):
                                for p in range(num_states[8]):
                                    for q in range(num_states[9]):
                                        A[5][:,:,i,j,k,l,m,n,o,2,q] = np.array([[0.1, 0.1], #positive
                                                                            [0.2, 0.2], #negative
                                                                            [0.1, 0.1]])#not_clear

    #(sort_safe:sort_hazardous, s_slots-0, h_slots-0, slot1_status-1, slot2_status-1, slot3_status-1, h_trustworthines-:, end_eff_position-0, action_outcome-0)
    for i in range(num_states[1]):
        for j in range(num_states[2]):
            for l in range(num_states[4]):
                for m in range(num_states[5]):
                    for n in range(num_states[6]):
                        for q in range(num_states[9]):
                            A[5][:,:,i,j,1,l,m,n,0,2,q] = np.array([[0.1, 0.1], #positiv
                                                                [0.2, 0.2], #negative
                                                                [0.1, 0.1]])#not_clear

    #(sort_safe:sort_hazardous, s_slots-0, h_slots-0, slot1_status-1, slot2_status-1, slot3_status-1, h_trustworthines-:, end_eff_position-0, action_outcome-0)
    for i in [0,1]:
        for j in [0,1]:
            for l in range(num_states[4]):
                for m in range(num_states[5]):
                    for n in range(num_states[6]):
                        for q in range(num_states[9]):
                            A[5][:,:,i,j,1,l,m,n,0,2,q] = np.array([[0.1, 0.1], #positiv
                                                                [0.9, 0.9], #negative
                                                                [0.1, 0.1]])#not_clear

    #(sort_safe:sort_hazardous, s_slots-0, h_slots-0, slot1_status-1, slot2_status-1, slot3_status-1, h_trustworthines-:, end_eff_position-0, action_outcome-0)
    for i in range(num_states[1]):
        for j in range(num_states[2]):
            for k in range(num_states[3]):
                for m in range(num_states[5]):
                    for n in range(num_states[6]):
                        for q in range(num_states[9]):
                            A[5][:,:,i,j,k,1,m,n,1,2,q] = np.array([[0.1, 0.1], #positiv
                                                                [0.2, 0.2], #negative
                                                                [0.1, 0.1]])#not_clear 
                        
    #(sort_safe:sort_hazardous, s_slots-0, h_slots-0, slot1_status-1, slot2_status-1, slot3_status-1, h_trustworthines-:, end_eff_position-0, action_outcome-0)
    for i in [0,2]:
        for j in [0,2]:
            for k in range(num_states[3]):
                for m in range(num_states[5]):
                    for n in range(num_states[6]):
                        for q in range(num_states[9]):
                            A[5][:,:,i,j,k,1,m,n,1,2,q] = np.array([[0.1, 0.1], #positiv
                                                                [0.9, 0.9], #negative
                                                                [0.1, 0.1]])#not_clear
                        
    #(sort_safe:sort_hazardous, s_slots-0, h_slots-0, slot1_status-1, slot2_status-1, slot3_status-1, h_trustworthines-:, end_eff_position-0, action_outcome-0)
    for i in range(num_states[1]):
        for j in range(num_states[2]):
            for k in range(num_states[3]):
                for l in range(num_states[4]):
                    for n in range(num_states[6]):
                        for q in range(num_states[9]):
                            A[5][:,:,i,j,k,l,1,n,2,2,q] = np.array([[0.1, 0.1], #positiv
                                                                [0.2, 0.2], #negative
                                                                [0.1, 0.1]])#not_clear
                        
    #(sort_safe:sort_hazardous, s_slots-0, h_slots-0, slot1_status-1, slot2_status-1, slot3_status-1, h_trustworthines-:, end_eff_position-0, action_outcome-0)
    for i in [1,2]:
        for j in [1,2]:
            for k in range(num_states[3]):
                for l in range(num_states[4]):
                    for n in range(num_states[6]):
                        for q in range(num_states[9]):
                            A[5][:,:,i,j,k,l,1,n,2,2,q] = np.array([[0.1, 0.1], #positiv
                                                                [0.9, 0.9], #negative
                                                                [0.1, 0.1]])#not_clear

                              
                            
    ################# - (3D-pose)     end_effector_position: ['ideal', 'slot1', 'slot2', 'slot3', 'slot4'] ################
    #(sort_safe:sort_hazardous, s_slots-0, h_slots-0, slot1_status-1, slot2_status-1, slot3_status-1, h_trustworthines-:, end_eff_position-0, action_outcome-0)
    for i in range(num_states[1]):
        for j in range(num_states[2]):
            for k in range(num_states[3]):
                for l in range(num_states[4]):
                    for m in range(num_states[5]):
                        for n in range(num_states[6]):
                            for p in range(num_states[8]):
                                for q in range(num_states[9]):
                                    A[6][:,:,i,j,k,l,m,n,0,p,q] = np.array([[0.9, 0.9], #ideal
                                                                        [0.1, 0.1], #slot1
                                                                        [0.1, 0.1], #slot2
                                                                        [0.1, 0.1]])#slot3
    for i in range(num_states[1]):
        for j in range(num_states[2]):
            for k in range(num_states[3]):
                for l in range(num_states[4]):
                    for m in range(num_states[5]):
                        for n in range(num_states[6]):
                            for p in range(num_states[8]):
                                for q in range(num_states[9]):
                                    A[6][:,:,i,j,k,l,m,n,1,p,q] = np.array([[0.1, 0.1], #ideal
                                                                        [0.9, 0.9], #slot1
                                                                        [0.1, 0.1], #slot2
                                                                        [0.1, 0.1]])#slot3
                                
    for i in range(num_states[1]):
        for j in range(num_states[2]):
            for k in range(num_states[3]):
                for l in range(num_states[4]):
                    for m in range(num_states[5]):
                        for n in range(num_states[6]):
                            for p in range(num_states[8]):
                                for q in range(num_states[9]):
                                    A[6][:,:,i,j,k,l,m,n,2,p,q] = np.array([[0.1, 0.1], #ideal
                                                                        [0.1, 0.1], #slot1
                                                                        [0.9, 0.9], #slot2
                                                                        [0.1, 0.1]])#slot3
                                
    for i in range(num_states[1]):
        for j in range(num_states[2]):
            for k in range(num_states[3]):
                for l in range(num_states[4]):
                    for m in range(num_states[5]):
                        for n in range(num_states[6]):
                            for p in range(num_states[8]):
                                for q in range(num_states[9]):
                                    A[6][:,:,i,j,k,l,m,n,3,p,q] = np.array([[0.1, 0.1], #ideal
                                                                        [0.1, 0.1], #slot1
                                                                        [0.1, 0.1], #slot2
                                                                        [0.9, 0.9]])#slot3
                            
                                
    ################# - (metacognition) obedience: ['obeied', 'not_obeyed'] ################
    #(sort_safe:sort_hazardous, s_slots-0, h_slots-0, slot1_status-1, slot2_status-1, slot3_status-1, h_trustworthines-:, end_eff_position-0, action_outcome-0)                                
                                
    for i in range(num_states[1]):
        for j in range(num_states[2]):
            for k in range(num_states[3]):
                for l in range(num_states[4]):
                    for m in range(num_states[5]):
                        for n in range(num_states[6]):
                            for o in range(num_states[7]):
                                for p in range(num_states[8]):
                                    A[7][:,:,i,j,k,l,m,n,o,p,0] = np.array([[1, 1], #obeyed
                                                                            [0, 0]]) #not_obeyed

    for i in range(num_states[1]):
        for j in range(num_states[2]):
            for k in range(num_states[3]):
                for l in range(num_states[4]):
                    for m in range(num_states[5]):
                        for n in range(num_states[6]):
                            for o in range(num_states[7]):
                                for p in range(num_states[8]):
                                    A[7][:,:,i,j,k,l,m,n,o,p,1] = np.array([[0, 0], #obeyed
                                                                            [1, 1]]) #not_obeyed                                
                                
                                ##############

    B = utils.random_B_matrix(num_states, num_controls)
    ############ control actions - team_task:  ['sort_safe', 'sort_hazardous'] ############
    matrix_22 = np.array([[0.1, 0.1],
                         [0.1, 0.1]])
    matrix_33 = np.array([[0.1, 0.1, 0.1],
                          [0.1, 0.1, 0.1],
                          [0.1, 0.1, 0.1]])
    B[0][:,:,0] = np.eye(2)

    ############ control actions - safe_slots: ['slot12', 'slot13','slot23'] ############

    B[1][:,:,0] = np.eye(3)

    ############ control actions - hazardous_slots: ['slot12', 'slot13', 'slot14', 'slot23', 'slot24', 'slot34'] ############

    B[2][:,:,0] = np.eye(3)

    ############ control actions - slot123_status:['empty', 'occupied'] ############

    B[3][:,:,0] = copy.deepcopy(matrix_22)
    B[4][:,:,0] = copy.deepcopy(matrix_22)
    B[5][:,:,0] = np.eye(2)

    ############ control actions - h_trustworthines:['reliable', 'unreliable'] ############

    B[6][:,:,0] = np.eye(2)


    ############ control actions - end_effector_position : ['ideal', 'slot1', 'slot2', 'slot3'] ############
    # action 0 is 'do_nothing'
    B[7][:,:,0] = np.array([[1.0, 1.0, 1.0, 1.0],
                            [0.1, 0.1, 0.1, 0.1],
                            [0.1, 0.1, 0.1, 0.1],
                            [0.1, 0.1, 0.1, 0.1]])

    # action 1 is 'slot1'
    B[7][:,:,1] = np.array([[0.1, 0.1, 0.1, 0.1],
                            [1.0, 1.0, 1.0, 1.0],
                            [0.1, 0.1, 0.1, 0.1],
                            [0.1, 0.1, 0.1, 0.1]])

    # action 2 is 'slot2'
    B[7][:,:,2] = np.array([[0.1, 0.1, 0.1, 0.1],
                            [0.1, 0.1, 0.1, 0.1],
                            [1.0, 1.0, 1.0, 1.0],
                            [0.1, 0.1, 0.1, 0.1]])

    # action 3 is 'slot3'
    B[7][:,:,3] = np.array([[0.1, 0.1, 0.1, 0.1],
                            [0.1, 0.1, 0.1, 0.1],
                            [0.1, 0.1, 0.1, 0.1],
                            [1.0, 1.0, 1.0, 1.0]])

    ############ control actions - action_outcome:   ['correct_choice', 'incorrect_mapping_choice', 'occupied_slot_choice'] ############

    B[8][:,:,0] = np.eye(3)
    B[9][:,:,0] = copy.deepcopy(matrix_22)


                                             #####################

    D = utils.obj_array_uniform(num_states)

    ############ team_task:  ['sort_safe', 'sort_hazardous'] ############
    D[0] = np.array([0.1, 0.1])

    ############ safe_slots: ['slot12', 'slot13','slot23'] ############
    D[1] = np.array([0.1, 1000.1, 0.1])

    ############ hazardous_slots: ['slot12', 'slot13','slot23'] ############
    D[2] = np.array([0.1, 0.1, 1000.1])

    ############ slot123_status: ['emprty', 'occupied'] ############
    D[3] = np.array([0.1, 0.1])
    D[4] = np.array([0.1, 0.1])
    D[5] = np.array([0.1, 0.1])

    ############ h_trustworthines:['reliable', 'unreliable'] ############
    D[6] = np.array([1.1, 0.1])

    ############ end_effector_position : ['ideal', 'slot1', 'slot2', 'slot3'] ############
    D[7] = np.array([0.1, 0.1, 0.1, 0.1])

    ############ action_outcome:   ['correct_choice', 'incorrect_mapping_choice', 'occupied_slot_choice'] ############
    D[8] = np.array([0.1, 0.1, 0.1])
    
                                ##########################



    C = utils.obj_array_uniform(num_obs)

    ################# - (fxed_camera) objs on picking_slot: ['safe', 'hazardous', 'not_clear'] ################

    C[0] = np.array([[0.0, 0.0, 0.0, 0.0], #safe
                     [0.0, 0.0, 0.0, 0.0], #hazardous
                     [0.0, 0.0, 0.0, 0.0]])#not_clear

    ################# - (fxed_camera) objs on slot1: ['safe', 'hazardous', 'not_clear', 'empty'] ################

    C[1] = np.array([[10.0, 10.0, 10.0, 10.0], #safe
                     [-15.0, -15.0, -15.0, -15.0], #hazardous
                     [0.0, 0.0, 0.0, 0.0], #not_clear
                     [0.0, 0.0, 0.0, 0.0]])#empty

    ################# - (fxed_camera) objs on slot2: ['safe', 'hazardous', 'not_clear', 'empty'] ################

    C[2] = np.array([[-15.0, -15.0, -15.0, -15.0], #safe
                     [10.0, 10.0, 10.0, 10.0], #hazardous
                     [0.0, 0.0, 0.0, 0.0], #not_clear
                     [0.0, 0.0, 0.0, 0.0]])#empty

    ################# - (fxed_camera) objs on slot3: ['safe', 'hazardous', 'not_clear', 'empty'] ################

    C[3] = np.array([[20.0, 20.0, 20.0, 20.0], #safe
                     [20.0, 20.0, 20.0, 20.0], #hazardous
                     [0.0, 0.0, 0.0, 0.0], #not_clear
                     [0.0, 0.0, 0.0, 0.0]])#empty

    ################# - (voice) h_commands: ['slot1', 'slot2', 'slot3', 'slot4', 'wait'] ################

    C[4] = np.array([[0.0, 0.0, 0.0, 0.0], #slot1
                     [0.0, 0.0, 0.0, 0.0], #slot2
                     [0.0, 0.0, 0.0, 0.0], #slot3
                     [0.0, 0.0, 0.0, 0.0]])#wait

    ################# - (voice) h_feedback_commands: ['positive', 'negative', 'not_clear'] ################

    C[5] = np.array([[10.0, 10.0, 10.0, 10.0], #positive
                     [-10.0, -10.0, -10.0, -10.0], #negative
                     [0.0, 0.0, 0.0, 0.0]])#not_clear

    ################# - (3D-pose)     end_effector_position: ['ideal', 'slot1', 'slot2', 'slot3', 'slot4'] ################

    C[6] = np.array([[0.0, 0.0, 0.0, 0.0], #ideal
                     [0.0, 0.0, 0.0, 0.0], #slot1
                     [0.0, 0.0, 0.0, 0.0], #slot2
                     [0.0, 0.0, 0.0, 0.0]])#slot3
    
    ################# - (metacognition) obedience: ['obeyed', 'not_obeyed'] ################

    C[7] = np.array([[1.0, 1.0, 1.0, 1.0], #obeyed
                     [-1.0, -1.0, -1.0, -1.0]]) #not_obeyed
    
    control_fac_idx = [7]
    Temp_horizon = 4

    return A, B, C, D, num_states, num_obs, num_controls, control_fac_idx, Temp_horizon