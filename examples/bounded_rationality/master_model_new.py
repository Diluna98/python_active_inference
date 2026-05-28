import os
import json
import numpy as np
from numpy.ma import copy
import pandas as pd
from PyAIF import utils, ActiveInfAgent
from environment import GridEnvironment
from meta_gen_mod import MetaAgent
import matplotlib.pyplot as plt
import time
import copy
import random
import psutil

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

def cpu_monitor(res_size, inference_duration):
    baseline_duration = {
        20: 2241.80495083,
        10: 250.69090593,
        5: 123.4742106,
        2: 89.86553494
    }

    mu = baseline_duration[res_size]

    availability = 100.0 * (mu / max(inference_duration, 1e-16))

    # clamp
    availability = max(0.0, min(100.0, availability))

    return availability

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
    
    #B[3][:,:,0] = np.eye(n_states[3])

def create_generative_model(model_size, res_size):

    num_obs = [0, 0, 0]
    num_states = [model_size, model_size, res_size*res_size]
    num_controls = [3, 3, 1]
    control_fac_idx = [0, 1]
    Temp_horizon = 3
    
    B = utils.zeros_B_matrix(num_states, num_controls)
    fill_transitions(num_states, B)

    D = utils.uniform_D_matrix(num_states)
    
    
    

    return B, D, num_states, num_obs, num_controls, control_fac_idx, Temp_horizon 

def get_new_BD_matrices(new_dims, num_controls):
    
    B = utils.zeros_B_matrix(new_dims, num_controls)
    fill_transitions(new_dims, B)

    D = utils.uniform_D_matrix(new_dims)

    return B, D
    
    
    

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

def remap_distribution(p_old, k_new):
    """
    Map a categorical distribution to new cardinality.
    Assumes ordered states.
    """

    k_old = len(p_old)

    old_grid = np.linspace(0, 1, k_old)
    new_grid = np.linspace(0, 1, k_new)

    p_new = np.interp(new_grid, old_grid, p_old)

    p_new = np.clip(p_new, 1e-16, None)
    p_new /= p_new.sum()

    return p_new

def change_model(agent, states_dim=None, num_controls=None, meta_action=None):
    new_dims = copy.deepcopy(states_dim)
    new_dims[2] = RES_SIZES[meta_action]**2

    P, T, F = agent.policy_dep_posteriors.shape

    new_policy_dep_posteriors = copy.deepcopy(agent.policy_dep_posteriors)
    new_bayesian_mod_avg = copy.deepcopy(agent.bayesian_mod_avg)

    for p in range(P):
        for t in range(T):

                p_old = agent.policy_dep_posteriors[p, t, 2]

                k_new = new_dims[2]

                new_policy_dep_posteriors[p, t, 2] = remap_distribution(
                    p_old,
                    k_new
                )

    for t in range(T):

        p_old = agent.bayesian_mod_avg[t, 2]

        k_new = new_dims[2]

        new_bayesian_mod_avg[t, 2] = remap_distribution(
            p_old,
            k_new
        )

    if getattr(agent, "previous_qs_T", None) is not None:

        new_previous_qs_T = copy.deepcopy(agent.previous_qs_T)

        p_old = agent.previous_qs_T[2]

        k_new = new_dims[2]

        new_previous_qs_T[2] = remap_distribution(
            p_old,
            k_new
        )

        agent.previous_qs_T = new_previous_qs_T
    agent.states_dim = new_dims
    agent.update_external_likelihood_model(new_dims)
    agent.policy_dep_posteriors = new_policy_dep_posteriors
    agent.bayesian_mod_avg = new_bayesian_mod_avg

    B, D = get_new_BD_matrices(new_dims, num_controls)
    agent.pB = B
    agent.B = copy.deepcopy(agent.pB)
    agent.transposed_B = agent._transpose_B_matrix()
    agent.pD = D
    agent.D = copy.deepcopy(agent.pD)



if __name__ == "__main__":
    # Create a folder to keep things tidy
    output_dir = "profiling_results"
    os.makedirs(output_dir, exist_ok=True)

    TRIALS_PER_RES = 1  # Number of random starts per resolution
    STEPS_PER_TRIAL = 700 # Limit steps so we don't get stuck in one spot
    #NUM_SIMULATIONS = 1
    MODEL_SIZES = [20]
    RES_SIZES = [2, 5, 10, 20]  # Different resolutions to test
    profile_data = []   

    # Constants
    TOTAL_SPACE = 500
    GRID_SIZE = MODEL_SIZES[0] # e.g., 10
    STEP = TOTAL_SPACE / GRID_SIZE
    OFFSET = STEP / 2
    # 1. Generate all possible cell centers
    # This creates a list of coordinates like [25.0, 75.0, 125.0, ...]
    #possible_centers = [OFFSET + (i * STEP) for i in range(GRID_SIZE)]

    # 2. Create all possible (x, y) pairs
    #all_locations = [(x, y) for x in possible_centers for y in possible_centers]

    sample_locations = []# random.sample(all_locations, 399)
    sample_locations = list(set(sample_locations) | {(12.5, 12.5)})

    for res_size in [10]:
        profile_data = [] # Reset list for each resolution
        print(f"Profiling {res_size}...")
         #for sim_id in range(1, NUM_SIMULATIONS + 1):
        print(f"\n--- PROFILING RESOLUTION: {res_size}x{res_size} ---")
        B, D, num_states, num_obs, num_controls, control_fac_idx, Temp_horizon = create_generative_model(MODEL_SIZES[0], res_size)
        current_res = res_size
        for loc_x, loc_y in sample_locations:
            print(f"Testing location: ({loc_x}, {loc_y})")
        
            env = GridEnvironment(size=MODEL_SIZES[0], s_x=loc_x, s_y=loc_y)
            obs_limits = env.get_obs_limits()

            policies = utils.construct_policies(num_states, num_controls, Temp_horizon-1, control_fac_idx)
            policies = filter_diagonal_policies(policies)
            ainf_agent = ActiveInfAgent(states_dim=num_states, obs_dim=num_obs, controls_dim=num_controls,
                                        controlable_states=control_fac_idx, planning_depth=Temp_horizon,
                                        number_of_msg_passing=10, trials=TRIALS_PER_RES, B=B, D=D,
                                        policies=policies, policy_pruning=False, learning_A=False, learning_D=False, learning_B=False, learning_C=False,
                                        continous_obs=True, lm_name = "task", mod_dependency=[[0], [1], [0, 1, 2]], pref_dep=[[0, 1]], obs_limits=obs_limits, learning_rate=0.1, forgeting_rate=0.99,
                                        obstacles_dic=env.obstacles, action_selection="deterministic")
            
            
            meta_agent = MetaAgent()
            risk = []
            pref = []
            for trial in range(TRIALS_PER_RES):
                obs, done = env.reset(random_start=False)
                #ainf_agent.store_parameters()
                #ainf_agent.normalize_columns()
                ainf_agent.initialize_variables()
                for t in range(STEPS_PER_TRIAL):
                    try:
                        ainf_agent.observations[t] = np.array(obs)
                        #ainf_agent.external_lm.plot_goal_likelihood_inference(np.array(obs))

                        # Start Timer for Wall-clock observation
                        start_time = time.perf_counter()
                        ainf_agent.infer_states(trial, t)
                        # End Timer
                        end_time = time.perf_counter()
                        inference_duration = (end_time - start_time) * 1000 # Convert to ms

                        ainf_agent.infer_policies_parallel(trial, t)

                        chosen_action, action_list = ainf_agent.choose_action(trial, t)

                        if chosen_action is not None:
                            #chosen_action[0] = np.random.choice([0, 1, 2])
                            #chosen_action[1] = np.random.choice([0, 1, 2])
                        
                            if t % 3 == 0:
                                # Get Stats (Risk and Ambiguity)
                                # Note: We take the MAX (worst-case)
                                stats = ainf_agent.get_stats(t)
                                pred_divergence = stats.get('pred_divergence', [0])
                                mean_surprise = stats.get('mean_surprise', [0])
                                #context = int(0 if pred_divergence <= np.float64(546) else (1 if pred_divergence <= np.float64(699) else (2 if pred_divergence < np.float64(737) else 3)))
                                """
                                profile_data.append({
                                    "resolution": res_size,
                                    "trial": trial,
                                    "step": t,
                                    "context": context,
                                    "pred_divergence": pred_divergence,
                                    "mean_surprise": mean_surprise,
                                    "inference_time_ms": inference_duration
                                })
                                """
                                cpu_availability = cpu_monitor(current_res, inference_duration)
                                #print(cpu_availability)
                                meta_action = meta_agent.run_meta_inference((pred_divergence, mean_surprise, inference_duration, cpu_availability))
                                print(f"context: {pred_divergence}, Mean surprise: {mean_surprise}, latency:{inference_duration}, CPU availability: {cpu_availability}")
                                #print(f"mean_surprise: {mean_surprise}, pred_div: {pred_divergence}")
                                #meta_action = np.random.choice([0, 1, 2, 3, 4], p=[0.15, 0.15, 0.15, 0.15, 0.4])
                                #print(f"chosen action", meta_action)
                                if not meta_action == 4:
                                    if not current_res == RES_SIZES[meta_action]:
                                        #change_model(ainf_agent, num_states, num_controls, meta_action)
                                        #current_res = RES_SIZES[meta_action]
                                        continue
                                #print(f"Time {t}, " + ", ".join([f"{k.capitalize()}: {np.min(v):.4f}" for k, v in stats.items()]))
                                #print(f"inference time: {inference_duration:.2f} ms")
                                
                            
                            a_action = tuple(int(x) for x in chosen_action[:2])
                            obs, done = env.step(a_action)
                            if done:
                                print(f"Goal reached at time {t} with resolution {res_size}x{res_size}!")
                                break
                        if t%Temp_horizon == Temp_horizon - 1:
                            ainf_agent.perform_learning(trial)
                            #ainf_agent.store_parameters()
                            ainf_agent.initialize_variables()
                        ainf_agent.step_time(t)
                        #t += 1

                    except KeyboardInterrupt:
                        print(f"Stopping at t={t}")
                        break
        
        print(f"mu_div: {meta_agent.meta_agent.external_lm.mu_div.tolist()}, sigma_div: {meta_agent.meta_agent.external_lm.sigma_div.tolist()}, mu_err: {meta_agent.meta_agent.external_lm.mu_err.tolist()}, sigma_err: {meta_agent.meta_agent.external_lm.sigma_err.tolist()}, mu_lat: {meta_agent.meta_agent.external_lm.mu_lat.tolist()}, sigma_lat: {meta_agent.meta_agent.external_lm.sigma_lat.tolist()}")
        """
        data = {
            "mu_div": meta_agent.meta_agent.external_lm.mu_div.tolist(),
            "sigma_div": meta_agent.meta_agent.external_lm.sigma_div.tolist(),
            "mu_err": meta_agent.meta_agent.external_lm.mu_err.tolist(),
            "sigma_err": meta_agent.meta_agent.external_lm.sigma_err.tolist(),
            "mu_lat": meta_agent.meta_agent.external_lm.mu_lat.tolist(),
            "sigma_lat": meta_agent.meta_agent.external_lm.sigma_lat.tolist(),
            "mu_cpu": meta_agent.meta_agent.external_lm.mu_cpu.tolist(),
            "sigma_cpu": meta_agent.meta_agent.external_lm.sigma_cpu.tolist()
        }

        with open("external_lm_params.json", "w") as f:
            json.dump(data, f)

        with open("external_lm_params.json", "r") as f:
            data = json.load(f)

        mu_div = np.array(data["mu_div"])
        mu_err = np.array(data["mu_err"])
        mu_lat = np.array(data["mu_lat"])
        mu_cpu = np.array(data["mu_cpu"])
        sigma_div = np.array(data["sigma_div"])
        sigma_err = np.array(data["sigma_err"])
        sigma_lat = np.array(data["sigma_lat"])
        sigma_cpu = np.array(data["sigma_cpu"])

        
        # Save this resolution to its own file
        df = pd.DataFrame(profile_data)
        filename = os.path.join(output_dir, f"results_res_260526_{res_size}.csv")
        df.to_csv(filename, index=False)
        print(f"Saved: {filename}")
        """