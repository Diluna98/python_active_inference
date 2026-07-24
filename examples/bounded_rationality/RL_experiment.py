from __future__ import annotations


from pathlib import Path
import os
import json
import sys
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
sys.path.append(r"C:\Users\dawarn\Documents\matrices_for_ICRA_paper")
from q_learning_meta_baseline import ACTION_LABELS, QLearningMetaController

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

res_sizes = [2, 5, 10, 20]

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
        20: 2655.80495083,
        10: 350.69090593,
        5: 140.4742106,
        2: 80.86553494
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
    new_dims[2] = res_sizes[meta_action]**2

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

def one_hot_confidence(action: int, n_actions: int = 5) -> list[float]:
    confidence = [0.0] * n_actions
    confidence[int(action)] = 1.0
    return confidence


def compute_rl_reward(latency_ms, prediction_error, info_gain_proxy):
    return float(
        -(latency_ms / 100.0)
        -prediction_error * info_gain_proxy
    )

def load_or_create_controller(path):
    if path.exists():
        return QLearningMetaController.load(path)
    return QLearningMetaController()

def run_experiment() -> None:
    output_dir = "profiling_results"
    os.makedirs(output_dir, exist_ok=True)

    trials_per_res = 50
    steps_per_trial = 100
    model_sizes = [20]
    res_sizes = [2, 5, 10, 20]
    goal_locations = [(212.5, 312.5)]
    START_LOCATIONS = [(487.5, 487.5), (12.5, 487.5), (487.5, 12.5)]#[(12.5, 12.5), (487.5, 487.5), (12.5, 487.5), (487.5, 12.5)]
    cpu_load_name = "high"  # Options: "low", "medium", "high"

    total_space = 500
    grid_size = model_sizes[0]
    step = total_space / grid_size
    offset = step / 2
    possible_centers = [offset + (i * step) for i in range(grid_size)]
    _all_locations = [(x, y) for x in possible_centers for y in possible_centers]

    policy_path = Path("profiling_results/q_learning_policy_latest.json")

    controller = load_or_create_controller(policy_path)

    for start in START_LOCATIONS:
        for goal in goal_locations:
            initial_res_size = 2
            out_path = f"Artifacts_cpu_load_{cpu_load_name}_{start[0]}_{start[1]}_{goal[0]}_{goal[1]}_Qlearning.jsonl"
            with open(out_path, "w", encoding="utf-8") as f:
                print(f"\n--- Q-LEARNING META-CONTROLLER, INITIAL {initial_res_size}x{initial_res_size} ---")

                latency_list = []
                meta_latency_list = []
                pred_err_list = []
                distance_to_goal_list = []
                meta_obs_list = []
                task_obs_list = []
                meta_action_confidence_list = []
                task_minimum_vfe_list = []
                meta_risk_list = []
                meta_ambiguity_list = []
                meta_info_gain_list = []
                meta_agent_ghost = MetaAgent()

                for trial in range(trials_per_res):
                    b_matrix, d_prior, num_states, num_obs, num_controls, control_fac_idx, temp_horizon = (
                        create_generative_model(model_sizes[0], initial_res_size)
                    )
                    current_res = initial_res_size

                    env = GridEnvironment(
                        size=model_sizes[0],
                        s_x=start[0],
                        s_y=start[1],
                        g_x=goal[0],
                        g_y=goal[1],
                        visualize=False,
                    )
                    obs_limits = env.get_obs_limits()
                    policies = utils.construct_policies(num_states, num_controls, temp_horizon - 1, control_fac_idx)
                    policies = filter_diagonal_policies(policies)
                    ainf_agent = ActiveInfAgent(
                        states_dim=num_states,
                        obs_dim=num_obs,
                        controls_dim=num_controls,
                        controlable_states=control_fac_idx,
                        planning_depth=temp_horizon,
                        number_of_msg_passing=10,
                        trials=trials_per_res,
                        B=b_matrix,
                        D=d_prior,
                        policies=policies,
                        policy_pruning=False,
                        learning_A=False,
                        learning_D=False,
                        learning_B=False,
                        learning_C=False,
                        continous_obs=True,
                        lm_name="task",
                        mod_dependency=[[0], [1], [0, 1, 2]],
                        pref_dep=[[0, 1]],
                        obs_limits=obs_limits,
                        learning_rate=0.1,
                        forgeting_rate=0.99,
                        obstacles_dic=None,
                        action_selection="deterministic",
                    )

                    obs, done = env.reset(random_start=False)
                    ainf_agent.initialize_variables()

                    latency_per_trial = []
                    meta_latency_per_trial = []
                    pred_err_per_trial = []
                    meta_obs_per_trial = []
                    task_obs_per_trial = []
                    meta_action_confidence_per_trial = []
                    task_minimum_vfe_per_trial = []
                    meta_risk_policies_per_trial = []
                    meta_ambiguity_policies_per_trial = []
                    meta_info_gain_policies_per_trial = []
                    distance_to_goal_per_trial = []

                    last_rl_obs = None
                    last_rl_action = None
                    last_distance = env.get_distance_to_the_goal()
                    cummalative_reward = 0.0
                    
                    for t in range(steps_per_trial):
                        try:
                            distance_before = env.get_distance_to_the_goal()
                            distance_to_goal_per_trial.append(distance_before)
                            ainf_agent.observations[t] = np.array(obs)

                            start_time = time.perf_counter()
                            ainf_agent.infer_states(trial, t)
                            inference_duration = (time.perf_counter() - start_time) * 1000.0
                            latency_per_trial.append(inference_duration)

                            ainf_agent.infer_policies_parallel(trial, t)
                            chosen_action, _action_list = ainf_agent.choose_action(trial, t)
                            if chosen_action is None:
                                ainf_agent.step_time(t)
                                continue

                            if t % 3 == 0:
                                stats = ainf_agent.get_stats(t)
                                info_gain_proxy = min(float(stats.get("info_gain_proxy", 0.0)), 2.0)
                                mean_surprise = float(stats.get("mean_surprise", 0.0))
                                cpu_availability = float(cpu_monitor(current_res, inference_duration))
                                print(f"inference_duration: {inference_duration:.2f} ms, cpu_availability: {cpu_availability:.2f}%")
                                rl_obs = (
                                    info_gain_proxy,
                                    mean_surprise,
                                    inference_duration,
                                    cpu_availability,
                                    res_sizes.index(current_res),
                                )

                                # First update previous meta action using current meta observation.
                                """
                                if last_rl_obs is not None and last_rl_action is not None:
                                    reward = compute_rl_reward(
                                        latency_ms=inference_duration,
                                        prediction_error=mean_surprise,
                                        info_gain_proxy=info_gain_proxy,
                                    )

                                    controller.learn(
                                        obs=last_rl_obs,
                                        action=last_rl_action,
                                        reward=reward,
                                        next_obs=rl_obs,
                                        done=done,
                                    )

                                    cummalative_reward += reward
                                """
                                #unused = meta_agent_ghost.run_meta_inference(res_sizes.index(current_res), (info_gain_proxy, mean_surprise, inference_duration, cpu_availability))
                                start_meta = time.perf_counter()
                                rl_action = int(controller.act(rl_obs, training=False))
                                meta_inference_duration = (time.perf_counter() - start_meta) * 1000.0
                                state = controller.discretizer.encode(controller.parse_obs(rl_obs))
                                q = controller._values(state)
                                print(f"Context: {info_gain_proxy:.4f}, state: {state}, q: {q}, action: {np.argmax(q)}")
                                #print(f"Context: {info_gain_proxy:.4f}, chosen action: {rl_action}")
                                selected_res = current_res if rl_action == 4 else res_sizes[rl_action]
                                switched = selected_res != current_res
                                if switched:
                                    change_model(ainf_agent, num_states, num_controls, rl_action)
                                    current_res = selected_res

                                meta_latency_per_trial.append(meta_inference_duration)
                                pred_err_per_trial.append(mean_surprise)
                                meta_obs_per_trial.append(list(rl_obs))
                                meta_action_confidence_per_trial.append(one_hot_confidence(rl_action))
                                meta_risk_policies_per_trial.append([np.nan] * len(ACTION_LABELS))
                                meta_ambiguity_policies_per_trial.append([np.nan] * len(ACTION_LABELS))
                                meta_info_gain_policies_per_trial.append([np.nan] * len(ACTION_LABELS))
                                task_obs_per_trial.append(np.array(obs).tolist())
                                task_minimum_vfe_per_trial.append(float(np.min(ainf_agent.F)) if hasattr(ainf_agent, "F") else np.nan)

                                last_rl_obs = rl_obs
                                last_rl_action = rl_action

                            a_action = tuple(int(x) for x in chosen_action[:2])
                            obs, done = env.step(a_action)
                            distance_after = env.get_distance_to_the_goal()

                            if done:
                                distance_to_goal_per_trial.append(distance_after)
                                print(f"Goal reached at time {t} with resolution {current_res}x{current_res}.")
                                print(f"Trial {trial}/{trials_per_res} completed with cumulative reward: {cummalative_reward:.2f}")
                                
                                break

                            if t % temp_horizon == temp_horizon - 1:
                                ainf_agent.perform_learning(trial)
                                ainf_agent.initialize_variables()

                            ainf_agent.step_time(t)

                        except KeyboardInterrupt:
                            print(f"Stopping at t={t}")
                            break
                    
                    #controller.decay_epsilon()
                    print(f"Trial {trial}/{trials_per_res} completed with cumulative reward: {cummalative_reward:.2f}, epsilon: {controller.epsilon}")
                    latency_list.append(np.array(latency_per_trial))
                    pred_err_list.append(np.array(pred_err_per_trial))
                    meta_latency_list.append(np.array(meta_latency_per_trial))
                    distance_to_goal_list.append(np.array(distance_to_goal_per_trial))
                    meta_obs_list.append(np.array(meta_obs_per_trial))
                    task_obs_list.append(np.array(task_obs_per_trial))
                    meta_action_confidence_list.append(np.array(meta_action_confidence_per_trial))
                    task_minimum_vfe_list.append(np.array(task_minimum_vfe_per_trial))
                    meta_risk_list.append(np.array(meta_risk_policies_per_trial))
                    meta_ambiguity_list.append(np.array(meta_ambiguity_policies_per_trial))
                    meta_info_gain_list.append(np.array(meta_info_gain_policies_per_trial))
                    controller.save(Path(output_dir) / "q_learning_policy_latest.json")

                artifacts = {
                    "distance_to_goal": [np.array(trial).tolist() for trial in distance_to_goal_list],
                    "latency": [np.array(trial).tolist() for trial in latency_list],
                    "prediction_error": [np.array(trial).tolist() for trial in pred_err_list],
                    "meta_latency": [np.array(trial).tolist() for trial in meta_latency_list],
                    "meta_obs": [np.array(trial).tolist() for trial in meta_obs_list],
                    "task_obs": [np.array(trial).tolist() for trial in task_obs_list],
                    "meta_action_confidance": [np.array(trial).tolist() for trial in meta_action_confidence_list],
                    "task_minimum_vfe": [np.array(trial).tolist() for trial in task_minimum_vfe_list],
                    "meta_risk_policies": [np.array(trial).tolist() for trial in meta_risk_list],
                    "meta_ambiguity_policies": [np.array(trial).tolist() for trial in meta_ambiguity_list],
                    "meta_info_gain_policies": [np.array(trial).tolist() for trial in meta_info_gain_list],
                }

                f.write(json.dumps(artifacts) + "\n")
            #controller.save(Path(output_dir) / f"q_learning_policy_{cpu_load_name}_{goal[0]}_{goal[1]}.json")
            #print(f"Saved: {out_path}")


if __name__ == "__main__":
    run_experiment()
