import numpy as np
import os
from generative_model_new import create_generative_model
from PyAIF import utils, ActiveInfAgent
from environment import SortingEnv
import matplotlib.pyplot as plt


if __name__ == "__main__":
    
    # Create the directory if it doesn't exist
    save_dir = "simulations_results"
    os.makedirs(save_dir, exist_ok=True)

    # Number of simulation runs
    NUM_SIMULATIONS = 1
    # Number of trials and generative models in each simulation
    TRIALS = 10
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
        num_controls = [1, 1, 1, 1, 4, 1, 1]
        policies_to_filter = utils.construct_policies(num_states, num_controls, Temp_horizon-1, control_fac_idx)
        policies = filter_policies(policies_to_filter)  # Use only the first model for this simulation
        ainf_agent = ActiveInfAgent(A=A, B=B, states_dim=num_states, obs_dim=num_obs, controls_dim=num_controls,
                                    controlable_states=control_fac_idx, trial_length=Temp_horizon,
                                    number_of_msg_passing=100, trials=TRIALS, D=D, C=C,
                                    policies=policies, policy_pruning=False, learning_A=False, learning_D=True, learning_B=False)
        
        env = SortingEnv(reliability=100)

        actionlist = []
        commandlist = []
        for trial in range(TRIALS):
            ainf_agent.store_parameters()
            ainf_agent.normalize_columns()
            ainf_agent.initialize_variables()
            obs = env.reset()
            a_action = 'ideal'
            for t in range(Temp_horizon):
                if t != 0:
                    obs = env.step(a_action)
                if t != Temp_horizon - 1:
                    commandlist.append(obs[4])
                ainf_agent.observations[trial, t, :] = np.array(obs[:7])
                ainf_agent.infer_states(trial, t)
                _, _ = ainf_agent.infer_policies(trial, t)
                mod_averages = ainf_agent.perform_modal_average(trial, t)
                chosen_action, action_list = ainf_agent.choose_action(trial, t)
                actionlist.append(action_list)
                if chosen_action is not None:
                    executable_actions = chosen_action[4]
                    print(f"\033[92mChosen action at trial {trial}, time {t}: {action_mappings[int(executable_actions)]}\033[0m")
                    a_action = action_mappings[int(executable_actions)]
            _, _, _, _, _, _ = ainf_agent.perform_learning(trial)

        filtered_actionlist = []
        for a in actionlist:
            if a is not None:
                filtered_actionlist.append(a)

        # Prepare confidence array
        conf_matrix = np.array([list(a.values())[0] for a in filtered_actionlist])
        action_labels = ['slot1', 'slot2', 'slot3', 'ideal']

        fig, ax = plt.subplots(figsize=(6, 4))
        im = ax.imshow(conf_matrix.T, aspect='auto', cmap='Blues')

        # Plot command markers
        for step, cmd in enumerate(commandlist):
            if step == 0:
                ax.plot(step, cmd, 'ro', label='Human Command')
            else:
                ax.plot(step, cmd, 'ro')

        ax.set_yticks(range(len(action_labels)))
        ax.set_yticklabels(action_labels)
        ax.set_xticks(range(len(commandlist)))
        ax.set_xlabel('Step')
        ax.set_ylabel(f"Agent's Intended Action Confidence")
        plt.colorbar(im, ax=ax, label='Confidence')

        ax.legend(loc='center left', bbox_to_anchor=(0.5, 1.05))
        plt.show()
        plt.show()