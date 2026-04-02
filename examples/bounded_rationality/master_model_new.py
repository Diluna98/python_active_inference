import numpy as np
import pandas as pd
from PyAIF import utils, ActiveInfAgent
from environment import GridEnvironment
from meta_gen_mod import MetaAgent
import matplotlib.pyplot as plt
import time

# state factors
"""
current_x_pos: 0, 1, 2, 3, 4, ......, 19
curren_y_pos: 0, 1, 2, 3, 4, ......., 19
goal_x_pos: 0, 1, 2, 3, 4, ......, 19
goal_y_pos: 0, 1, 2, 3, 4, ......., 19
"""

# obs modalities
"""
cell_id: 0, 1, 2, 3, 4, ......, 399
signal_strength: 0, 1, 2, 3, 4, ......, 50
"""

# actions
"""
x_pos: do_nothing, -x, +x
y_pos: do_nothing, -y, +y
"""

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

    # fill action 0 transistions for factor 2
    B[2][:,:,0] = np.eye(n_states[2])
    
    B[3][:,:,0] = np.eye(n_states[3])

def create_generative_model(model_size):

    num_obs = [0, 0, 0]
    num_states = [model_size, model_size, model_size, model_size]
    num_controls = [3, 3, 1, 1]
    control_fac_idx = [0, 1]
    Temp_horizon = 3
    
    B = utils.zeros_B_matrix(num_states, num_controls)
    fill_transitions(num_states, B)

    D = utils.uniform_D_matrix(num_states)
    

    return B, D, num_states, num_obs, num_controls, control_fac_idx, Temp_horizon 

def filter_policies(data):

    def keep_array(arr):
        first = arr[:, 0]
        second = arr[:, 1]
        third = arr[:, 2]

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

def filter_diagonal_policies(policies):
    filtered = []

    for p in policies:
        diagonal = np.any((p[:, 0] != 0) & (p[:, 1] != 0))
        others_zero = True if p.shape[1] <= 2 else np.all(p[:, 2:] == 0)

        if not diagonal and others_zero:
            filtered.append(p)

    return filtered

def get_signal_preference(obs_val, preferences_dict, SIGNAL_MIN=0, SIGNAL_MAX=30, NUM_POINTS=100):
    # 1. Ensure the observation is within bounds
    obs_val = np.clip(obs_val, SIGNAL_MIN, SIGNAL_MAX)
    
    # 2. Calculate the discrete index (0 to 99)
    # Using (NUM_POINTS - 1) because indices are 0-based
    idx = int(np.round(((obs_val - SIGNAL_MIN) / (SIGNAL_MAX - SIGNAL_MIN)) * (NUM_POINTS - 1)))
    
    # 3. Lookup in your preference dictionary
    # Assuming the key is 'signal'
    pref_value = preferences_dict[2][idx]
    
    return pref_value



if __name__ == "__main__":

    TRIALS_PER_RES = 20  # Number of random starts per resolution
    STEPS_PER_TRIAL = 30000 # Limit steps so we don't get stuck in one spot
    #NUM_SIMULATIONS = 1
    MODEL_SIZES = [15] 
    profile_data = []   

    for model_size in MODEL_SIZES:
    #for sim_id in range(1, NUM_SIMULATIONS + 1):
        print(f"\n--- PROFILING RESOLUTION: {model_size}x{model_size} ---")
        B, D, num_states, num_obs, num_controls, control_fac_idx, Temp_horizon = create_generative_model(model_size)

        env = GridEnvironment(size=model_size)
        obs_limits = env.get_obs_limits()

        policies = utils.construct_policies(num_states, num_controls, Temp_horizon-1, control_fac_idx)
        policies = filter_diagonal_policies(policies)
        ainf_agent = ActiveInfAgent(states_dim=num_states, obs_dim=num_obs, controls_dim=num_controls,
                                    controlable_states=control_fac_idx, planning_depth=Temp_horizon,
                                    number_of_msg_passing=30, trials=TRIALS_PER_RES, B=B, D=D,
                                    policies=policies, policy_pruning=False, learning_A=False, learning_D=False, learning_B=False, learning_C=False,
                                    continous_obs=True, lm_name = "task", mod_dependency=[[0], [1], [0, 1, 2, 3]], pref_dep=[[0, 1]], obs_limits=obs_limits, learning_rate=0.1, forgeting_rate=0.95,
                                    obstacles_dic=env.obstacles)
        
        
        #meta_agent = MetaAgent()
        risk = []
        pref = []
        for trial in range(TRIALS_PER_RES):
            obs, done = env.reset(random_start=False)
            ainf_agent.store_parameters()
            ainf_agent.normalize_columns()
            ainf_agent.initialize_variables()
            #t = 0
            for t in range(STEPS_PER_TRIAL):
                ainf_agent.observations[t] = np.array(obs)

                # Start Timer for Wall-clock observation
                start_time = time.perf_counter()
                ainf_agent.infer_states(trial, t)
                # End Timer
                end_time = time.perf_counter()
                inference_duration = (end_time - start_time) * 1000 # Convert to ms

                ainf_agent.infer_policies(trial, t)

                chosen_action, action_list = ainf_agent.choose_action(trial, t)
                if chosen_action is not None:
                    # Get Stats (Risk and Ambiguity)
                    # Note: We take the MAX (worst-case)
                    """                    stats = ainf_agent.get_stats(t)
                    max_risk = np.min(stats.get('risk', [0]))
                    preferences = ainf_agent.log_preferences_dict
                    pref_val = get_signal_preference(obs[2], preferences)
                    risk.append(max_risk)
                    pref.append(pref_val)
                    max_ambiguity = np.min(stats.get('ambiguity', [0]))
                    entropy = np.mean(stats.get('posterior_entropy', [0]))
                    #meta_agent.run_meta_inference([max_risk, max_ambiguity, inference_duration])  # Example: provide the middle x position as input
                    # Log the data for Level 2 Likelihood (A-matrix)
                    
                    profile_data.append({
                        "resolution": model_size,
                        "trial": trial,
                        "step": t,
                        "max_risk": max_risk,
                        "max_ambiguity": max_ambiguity,
                        "entropy": entropy,
                        "inference_time_ms": inference_duration
                    })
                    #stats = ainf_agent.get_stats(t)
                    print(f"Time {t}, " + ", ".join([f"{k.capitalize()}: {np.min(v):.4f}" for k, v in stats.items()]))
                    
                    """

                    a_action = tuple(int(x) for x in chosen_action[:2])
                    obs, done = env.step(a_action)
                if t%Temp_horizon == Temp_horizon - 1:
                    #ainf_agent.perform_learning(trial)
                    #ainf_agent.normalize_columns()
                    ainf_agent.initialize_variables()
                ainf_agent.step_time(t)
                #t += 1

        # Convert to DataFrame and Calculate Signatures
        #df = pd.DataFrame(profile_data)
        #df.to_csv("resolution_signatures_new.csv", index=False)

        # Summary Stats for L2 A-Matrix
        #signatures = df.groupby('resolution').agg(['mean', 'std']).round(4)
        #print("\n--- GENERATED RESOLUTION SIGNATURES (L2 A-MATRIX) ---")
        #print(signatures)