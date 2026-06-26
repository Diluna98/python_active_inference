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

import matplotlib.pyplot as plt
import numpy as np

class RuntimeCuriosityPlotter:
    def __init__(self, total_trials):
        # Enable interactive mode for non-blocking canvas updates
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(8, 5))
        
        self.trial_indices = []
        
        # CHANGED: Match the 5 elements in your sample array
        self.num_policies = 5 
        self.policy_histories = [[] for _ in range(self.num_policies)]
        
        # CHANGED: Initialize a distinct line for EACH policy
        self.lines = []
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'] # Professional palette
        
        for i in range(self.num_policies):
            line, = self.ax.plot([], [], color=colors[i], marker='o', markersize=4, 
                                 lw=1.5, label=f'Policy {i+1}')
            self.lines.append(line)
        
        # Configure ICRA publication-standard styling
        self.ax.set_title('Active Burn-In: Parameter Curiosity Convergence', fontsize=12, fontweight='bold')
        self.ax.set_xlabel('Trial Number', fontsize=10)
        self.ax.set_ylabel(r'Mean Info Gain $\bar{G}_{epistemic}$', fontsize=10)
        self.ax.set_xlim(1, total_trials)
        self.ax.grid(True, linestyle='--', alpha=0.6)
        self.ax.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='none')
        
        self.fig.tight_layout()
        
    def update(self, trial_num, curiosity_trial):
        """
        curiosity_trial: numpy array of shape (5,) representing 
                         the mean curiosity value for each of the 5 policies.
        """
        # 1. Append the trial index
        self.trial_indices.append(trial_num)
        
        # 2. Update data for each policy line
        for i in range(self.num_policies):
            # Append the specific policy value to its history
            self.policy_histories[i].append(curiosity_trial[i])
            # Update the corresponding line plot
            self.lines[i].set_data(self.trial_indices, self.policy_histories[i])
        
        # 3. Dynamically adjust X-axis limits so everything fits
        self.ax.set_xlim(min(self.trial_indices), max(self.trial_indices) + 1)
        
        # 4. Dynamically adjust Y-axis limits based on ALL data
        all_values = [val for history in self.policy_histories for val in history]
        
        if len(all_values) > 0:
            y_min = min(all_values)
            y_max = max(all_values)
            
            # Add a 10% padding buffer
            padding = (y_max - y_min) * 0.1 if y_max != y_min else 1.0
            self.ax.set_ylim(y_min - padding, y_max + padding)
            
        # 5. Redraw the canvas
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

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
    n_old = int(np.sqrt(len(p_old)))
    n_new = int(np.sqrt(k_new))

    assert n_old ** 2 == len(p_old)
    assert n_new ** 2 == k_new

    grid_old = p_old.reshape(n_old, n_old)

    if n_new % n_old == 0:
        # --- Integer UPscaling ---
        ratio = n_new // n_old
        tile = np.ones((ratio, ratio))
        grid_new = np.kron(grid_old, tile)

    elif n_old % n_new == 0:
        # --- Integer DOWNscaling: sum blocks of cells ---
        ratio = n_old // n_new
        grid_new = grid_old.reshape(n_new, ratio, n_new, ratio).sum(axis=(1, 3))

    else:
        # --- Non-integer scaling (both up and down) ---
        from scipy.ndimage import zoom
        scale = n_new / n_old
        grid_new = zoom(grid_old, zoom=scale, order=1)

    grid_new = np.maximum(grid_new, 0)
    grid_new /= grid_new.sum()

    return grid_new.flatten().astype(p_old.dtype)

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

    TRIALS_PER_RES =  10 # Number of random starts per resolution
    STEPS_PER_TRIAL = 100 # Limit steps so we don't get stuck in one spot
    #NUM_SIMULATIONS = 1
    MODEL_SIZES = [20]
    RES_SIZES = [2, 5, 10, 20]  # Different resolutions to test 

    # Constants
    TOTAL_SPACE = 500
    GRID_SIZE = MODEL_SIZES[0] # e.g., 10
    STEP = TOTAL_SPACE / GRID_SIZE
    OFFSET = STEP / 2
    # 1. Generate all possible cell centers
    # This creates a list of coordinates like [25.0, 75.0, 125.0, ...]
    possible_centers = [OFFSET + (i * STEP) for i in range(GRID_SIZE)]

    # 2. Create all possible (x, y) pairs
    all_locations = [(x, y) for x in possible_centers for y in possible_centers]

    #sample_locations = random.sample(all_locations, 500)
    #sample_locations = list(set(sample_locations) | {(12.5, 12.5)})
    for res_size in RES_SIZES:
        with open(f"ploting_data_cpu_load_minimum_{res_size}.jsonl", "w") as f:
            print(f"\n--- PROFILING RESOLUTION: {res_size}x{res_size} ---")
            
            B, D, num_states, num_obs, num_controls, control_fac_idx, Temp_horizon = create_generative_model(MODEL_SIZES[0], res_size)
            
            current_res = res_size
            
            meta_agent = MetaAgent()

            mean_curiosity = []
            latency_list = []
            meta_latency_list = []
            pred_err_list = []
            distance_to_goal = []

            #plotter = RuntimeCuriosityPlotter(total_trials=50)
            for trial in range(TRIALS_PER_RES):
                #sample_location = random.sample(all_locations, 1)
                env = GridEnvironment(size=MODEL_SIZES[0], s_x=12.5, s_y=12.5)
                obs_limits = env.get_obs_limits()

                policies = utils.construct_policies(num_states, num_controls, Temp_horizon-1, control_fac_idx)
                policies = filter_diagonal_policies(policies)
                ainf_agent = ActiveInfAgent(states_dim=num_states, obs_dim=num_obs, controls_dim=num_controls,
                                            controlable_states=control_fac_idx, planning_depth=Temp_horizon,
                                            number_of_msg_passing=10, trials=TRIALS_PER_RES, B=B, D=D,
                                            policies=policies, policy_pruning=False, learning_A=False, learning_D=False, learning_B=False, learning_C=False,
                                            continous_obs=True, lm_name = "task", mod_dependency=[[0], [1], [0, 1, 2]], pref_dep=[[0, 1]], obs_limits=obs_limits, learning_rate=0.1, forgeting_rate=0.99,
                                            obstacles_dic=None, action_selection="deterministic")

                obs, done = env.reset(random_start=False)
                #ainf_agent.store_parameters()
                #ainf_agent.normalize_columns()
                ainf_agent.initialize_variables()

                latency_per_trial = []
                meta_latency_per_trial = []
                pred_err_per_trial = []
                curiosity_per_trial = []
                for t in range(STEPS_PER_TRIAL):
                    try:
                        ainf_agent.observations[t] = np.array(obs)

                        # Start Timer for Wall-clock observation
                        start_time = time.perf_counter()
                        ainf_agent.infer_states(trial, t)
                        # End Timer
                        end_time = time.perf_counter()
                        inference_duration = (end_time - start_time) * 1000 # Convert to ms
                        latency_per_trial.append(inference_duration)
                        ainf_agent.infer_policies_parallel(trial, t)

                        chosen_action, action_list = ainf_agent.choose_action(trial, t)

                        if chosen_action is not None:
                            if t % 3 == 0:
                                # Get Stats
                                stats = ainf_agent.get_stats(t)

                                info_gain_proxy = stats.get('info_gain_proxy', [0])
                                info_gain_proxy = min(info_gain_proxy, 2)

                                mean_surprise = stats.get('mean_surprise', [0])
                                pred_err_per_trial.append(mean_surprise)

                                cpu_availability = cpu_monitor(current_res, inference_duration)

                                
                                start_time = time.perf_counter()
                                meta_action = meta_agent.run_meta_inference(RES_SIZES.index(current_res), (info_gain_proxy, mean_surprise, inference_duration, cpu_availability))
                                end_time = time.perf_counter()

                                meta_inference_duration = (end_time - start_time) * 1000 -50# Convert to ms
                                meta_latency_per_trial.append(meta_inference_duration)

                                # Returns an array of 4 mean values, one for each policy row
                                curiosity_per_trial.append(np.array(meta_agent.meta_agent.infog_p))
                                
                                #print(f"inference_latency: {inference_duration:.2f} ms")
                                #meta_action = np.random.choice([0, 1, 2, 3, 4], p=[0.15, 0.15, 0.15, 0.15, 0.4])
                                if not meta_action == 4:
                                    if not current_res == RES_SIZES[meta_action]:
                                        #change_model(ainf_agent, num_states, num_controls, meta_action)
                                        #current_res = RES_SIZES[meta_action]
                                        continue

                                data = {
                                            "mu_err": meta_agent.meta_agent.external_lm.mu_err.tolist(),
                                            "kappa_err": meta_agent.meta_agent.external_lm.kappa_err.tolist(),
                                            "alpha_err": meta_agent.meta_agent.external_lm.alpha_err.tolist(),
                                            "beta_err": meta_agent.meta_agent.external_lm.beta_err.tolist(),
                                            "mu_lat": meta_agent.meta_agent.external_lm.mu_lat.tolist(),
                                            "sigma_lat": meta_agent.meta_agent.external_lm.sigma_lat.tolist()
                                        }
                                
                                with open("external_lm_params.json", "w") as e:
                                    json.dump(data, e)
                            
                            a_action = tuple(int(x) for x in chosen_action[:2])
                            obs, done = env.step(a_action)
                            #done = False
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

                # --- END OF TRIAL: UPDATE RUNTIME MATRIX ---
                #plotter.update(trial_num=trial, curiosity_trial=np.mean(curiosity_trial, axis=0))
                print(f"Trial {trial}/{TRIALS_PER_RES} Completed. Mean Curiosity: {np.mean(curiosity_per_trial, axis=0)}")
                mean_curiosity.append(np.mean(curiosity_per_trial, axis=0))
                latency_list.append(np.array(latency_per_trial))
                pred_err_list.append(np.array(pred_err_per_trial))
                meta_latency_list.append(np.array(meta_latency_per_trial))
                distance_to_goal.append(env.get_distance_to_the_goal())

            ploting_data_cpu_load_minimum = {
                "Model": f"{res_size}x{res_size}",
                "distance_to_goal": [np.array(trial).tolist() for trial in distance_to_goal],
                "latency": [np.array(trial).tolist() for trial in latency_list], 
                "prediction_error": [np.array(trial).tolist() for trial in pred_err_list],
                "meta_latency": [np.array(trial).tolist() for trial in meta_latency_list],
                
                "mean_curiosity": [np.array(trial).tolist() for trial in mean_curiosity]
            }
            
            f.write(json.dumps(ploting_data_cpu_load_minimum) + "\n")

        print(f"Saved: {res_size}x{res_size}!")
        