import numpy as np
from PyAIF import utils, ActiveInfAgent
from environment import GridEnvironment
from BMR_module import BMRModule
import matplotlib.pyplot as plt
import time

# state factors
"""
x_pos: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13
y_pos: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
marked: 0, 1
"""

# obs modalities
"""
x_pos: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13
y_pos: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
feedback: 0, 1, 2, 3
"""

# actions
"""
x_pos: do_nothing, -x, +x
y_pos: do_nothing, -y, +y
marked: do_nothing, marked
"""
def fill_modalities(num_states, num_obs, A, goal_x, goal_y):

    #### Fill modality 0

    for i in range(num_states[1]):
        for j in range(num_states[2]):
            A[0][:,:,i,j] = np.eye(num_states[0], num_obs[0])

    #### Fill modality 1

    for j in range(num_states[2]):
        for i in range(num_states[1]):
            A[1][:,:,i,j][i] = np.ones((1, num_states[0]))

    #### Fill modality 2
    for i in range(num_states[1]):
        A[2][:,:,i,0][0] = np.ones((1, num_states[0])) #all the x and y positions get nothing if marking is false.

    # 1. Set the Perfect Goal Location
    # We set Observation 1 to 1.0 for the specific goal_y and goal_x
    A[2][1, goal_x, goal_y, 1] = 100.0

    # 2. Set the "Good" Neighbors
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dy == 0 and dx == 0:
                continue # Skip the goal itself (already set to perfect)
                
            ny, nx = goal_y + dy, goal_x + dx
            
            # Boundary checks
            if 0 <= ny < num_states[0] and 0 <= nx < num_states[1]:
                # Set Observation 2 to 1.0 for these neighboring states
                A[2][2, nx, ny, 1] = 1.0

    # 1. Sum across the observation axis (axis 0) to find unassigned states
    # A[2].sum(axis=0) will have a 0 wherever no observation has been assigned yet
    unassigned_states = (A[2].sum(axis=0) == 0)

    # 2. This uses to assign bad feedback for all the other cases where marking is true
    A[2][3, unassigned_states] = 1.0


def fill_transitions(n_states, B, dtype=np.float32):

    # fill action 0 transistions for factor 0 / do_nothing
    B[0][:,:,0] = np.eye(n_states[0])

    # fill action 1 transistions for factor 0 / -x
    for col in range(n_states[0]):
        row = col - 1
        if row < 0:
            row = 0
        B[0][row, col, 1] = 1.0

    # fill action 1 transistions for factor 0 / +x
    for col in range(n_states[0]):
        row = col + 1
        if row > n_states[0] -1:
            row = n_states[0] -1
        B[0][row, col, 2] = 1.0

    # fill action 0 transistions for factor 1 / do_nothing
    B[1][:,:,0] = np.eye(n_states[1])

    # fill action 1 transistions for factor 1 / -y
    for col in range(n_states[1]):
        row = col - 1
        if row < 0:
            row = 0
        B[1][row, col, 1] = 1.0

    # fill action 1 transistions for factor 1 / +y
    for col in range(n_states[1]):
        row = col + 1
        if row > n_states[1] -1:
            row = n_states[1] -1
        B[1][row, col, 2] = 1.0

    # fill action 0 transistions for factor 2 / do_nothing
    B[2][:,:,0] = np.eye(n_states[2])
    
    # fill action 0 transistions for factor 2 / mark
    B[2][:,:,1][1] = np.ones((1, n_states[2]))


def fill_C_exponential(C, modality, target_idx, peak=5.0, lam=0.05, floor=0.0):
    """
    Exponential decay around target_idx.
    """
    n_obs, T = C[modality].shape

    x = np.arange(n_obs)
    dist = np.abs(x - target_idx)
    profile = peak * np.exp(-lam * dist)

    if floor > 0.0:
        profile[profile < floor] = floor

    C[modality][:, :] = profile[:, None]




def create_generative_model():

    num_obs = [50, 50, 4]
    num_states = [50, 50, 2]
    num_controls = [3, 3, 2]
    control_fac_idx = [0, 1, 2]
    Temp_horizon = 3

    goal_x = 45 #45
    goal_y = 2 #2
    
    A = utils.zeros_A_matrix(num_obs, num_states)
    fill_modalities(num_states, num_obs, A, goal_x, goal_y)

    B = utils.zeros_B_matrix(num_states, num_controls)
    fill_transitions(num_states, B)
    

    D = utils.uniform_D_matrix(num_states)

    C = utils.zero_C_matrix(num_obs, Temp_horizon)
    fill_C_exponential(C, modality=0, target_idx=goal_x)

    fill_C_exponential(C, modality=1, target_idx=goal_y)
    
    C[2] = np.array([[0.0, 0.0, 0.0], #nothing
                    [5.0, 5.0, 5.0], #perfect
                    [1.0, 1.0, 1.0], #good
                    [-1.0, -1.0, -1.0]])#bad
    

    return A, B, C, D, num_states, num_obs, num_controls, control_fac_idx, Temp_horizon 

def filter_policies(data):

    def keep_array(arr):
        first = arr[:, 0]
        second = arr[:, 1]
        third = arr[:, 2]

        # rule 1: reject rows with exactly one 3 in first two positions
        bad_rows = (first == 3) ^ (second == 3)
        if np.any(bad_rows):
            return False

        # rule 2: reject if all rows are [3,3,*]
        if np.all((first == 3) & (second == 3)):
            return False

        # rule 3: reject any [3,3,1]
        if np.any((first == 3) & (second == 3) & (third == 1)):
            return False

        # rule 4: temporal consistency between first and second rows
        if arr.shape[0] >= 2:
            first_row_third = arr[0, 2]
            second_row = arr[1]

            # if first third is 1, second must be [3,3,0]
            if first_row_third == 1:
                if not (second_row[0] == 3 and second_row[1] == 3 and second_row[2] == 0):
                    return False

            # if second is [3,3,0], first third must be 1
            if (second_row[0] == 3 and second_row[1] == 3 and second_row[2] == 0):
                if first_row_third != 1:
                    return False

        return True

    return [arr for arr in data if keep_array(arr)]



if __name__ == "__main__":

    TRIALS = 10
    NUM_SIMULATIONS = 1
    entropies = []
    latencies = []
    model_size = 55    

    for sim_id in range(1, NUM_SIMULATIONS + 1):
        print(f"Running simulation {sim_id}...")
        A, B, C, D, num_states, num_obs, num_controls, control_fac_idx, Temp_horizon = create_generative_model()

        #controllable_factors = [-1, -1, 2]
        #controllable_modalities = [-1, -1, 4]
        #bmr = BMRModule(num_states, num_obs, num_controls, Temp_horizon, controllable_factors,  controllable_modalities, A, B, C, D, minimum_dim = 5, E=None)

        #A, B, C, D, num_states, num_obs = bmr.decrease_resolution(curren_dim=model_size)
        #fill_transitions(num_states, B)
        #waa, wbb, wcc, wdd = bmr.increase_resolution(curren_dim=50)

        #print(f"Reduced Shape: {reduced_5x5.shape}")
        #print(f"Top-left corner of reduced matrix:\n{reduced_5x5}")

        policies_to_filter = utils.construct_policies(num_states, num_controls, Temp_horizon-1, control_fac_idx)
        policies = filter_policies(policies_to_filter)
        ainf_agent = ActiveInfAgent(A=A, B=B, states_dim=num_states, obs_dim=num_obs, controls_dim=num_controls,
                                    controlable_states=control_fac_idx, planning_depth=Temp_horizon,
                                    number_of_msg_passing=30, trials=TRIALS, D=D, C=C,
                                    policies=policies, policy_pruning=False, learning_A=False, learning_D=False, learning_B=False, learning_C=False)
        
        env = GridEnvironment(size=model_size-5)
        for trial in range(TRIALS):
            obs, done = env.reset()
            ainf_agent.store_parameters()
            ainf_agent.normalize_columns()
            ainf_agent.initialize_variables()
            t = 0
            while not done:
                t_start = time.perf_counter()
                ainf_agent.observations[t%Temp_horizon, :] = np.array(obs)
                ainf_agent.infer_states(trial, t)
                ainf_agent.infer_policies(trial, t)
                mod_averages = ainf_agent.perform_modal_average(trial, t)
                chosen_action, action_list = ainf_agent.choose_action(trial, t)
                t_end = time.perf_counter()
                latencies.append(t_end - t_start)
                if chosen_action is not None:
                    a_action = tuple(int(x) for x in chosen_action)
                    obs, done = env.step(a_action)
                    if done:
                        ainf_agent.perform_learning(trial)
                t += 1