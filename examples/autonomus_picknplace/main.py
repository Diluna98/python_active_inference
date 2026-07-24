import numpy as np
import os
from generative_model import create_generative_model
from PyAIF import (
    ActiveInfAgent,
    CategoricalLikelihood,
    DeepTemporalInference,
    GenerativeModel,
    utils,
)
from environment import SortingEnv
import matplotlib.pyplot as plt


if __name__ == "__main__":
    
    # Create the directory if it doesn't exist
    save_dir = "simulations_results"
    os.makedirs(save_dir, exist_ok=True)

    # Number of simulation runs
    NUM_SIMULATIONS = 1
    # Number of trials and generative models in each simulation
    TRIALS = 100
    MODELS = 1

    action_mappings = {0:'slot1', 1:'slot2', 2:'slot3', 3:'ideal'}

    def filter_policies(policies):
        factor_idx = 4
        filtered = []

        for policy in policies:
            control_seq = policy[:, factor_idx]  # shape: (T,)

            # Condition 1: starts with non-zero
            if control_seq[0] == 3:
                continue

            # Condition 2: no repeated slot actions
            slot_actions = control_seq[control_seq != 3]
            unique_slot_actions = np.unique(slot_actions)
            if len(slot_actions) != len(unique_slot_actions):
                continue

            # Passed both conditions
            policy[:, [1, 2, 3]] = policy[:, [4]]
            filtered.append(policy)

        return filtered
    

    for sim_id in range(1, NUM_SIMULATIONS + 1):
        print(f"Running simulation {sim_id}...")
        A, B, C, D, num_states, num_obs, num_controls, control_fac_idx, Temp_horizon = create_generative_model()
        num_controls = [1, 1, 1, 1, 4]
        policies_to_filter = utils.construct_policies(num_states, num_controls, Temp_horizon-1, control_fac_idx)
        policies = filter_policies(policies_to_filter)  # Use only the first model for this simulation
        model = GenerativeModel(
            B=B,
            D=D,
            controls_dim=num_controls,
            controllable_factors=control_fac_idx,
            policies=policies,
        )
        likelihood = CategoricalLikelihood(A=A, preferences=C)
        inference = DeepTemporalInference(
            horizon=Temp_horizon,
            message_passing_iterations=100,
        )
        ainf_agent = ActiveInfAgent(
            model=model,
            likelihood=likelihood,
            inference=inference,
            trials=TRIALS,
            policy_pruning=False,
            learning_A=True,
            learning_D=True,
            learning_B=True,
            learning_C=False,
        )
        
        env = SortingEnv()

        actionlist = []
        for trial in range(TRIALS):
            ainf_agent.store_parameters()
            ainf_agent.reset(trial=trial)
            obs = env.reset()
            a_action = 'ideal'
            for t in range(Temp_horizon):
                if t != 0:
                    obs = env.step(a_action)
                ainf_agent.observe(obs, time_step=t)
                ainf_agent.infer_states(trial, t)
                _, _ = ainf_agent.infer_policies(trial, t)
                ainf_agent.calculate_counterfactual_disparity(t)
                mod_averages = ainf_agent.perform_modal_average()
                chosen_action, action_list = ainf_agent.choose_action(trial, t)
                actionlist.append(action_list)
                if chosen_action is not None:
                    executable_actions = chosen_action[4]
                    print(f"\033[92mChosen action at trial {trial}, time {t}: {action_mappings[int(executable_actions)]}\033[0m")
                    a_action = action_mappings[int(executable_actions)]
            ainf_agent.perform_learning(trial)
