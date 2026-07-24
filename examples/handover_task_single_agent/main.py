import numpy as np
import os
from generative_model_new import create_generative_model
from PyAIF import (
    ActiveInfAgent,
    CategoricalLikelihood,
    DeepTemporalInference,
    GenerativeModel,
    utils,
)
from environment import SortingEnv
import matplotlib.pyplot as plt


def filter_policies(policies):
    factor_idx = 4
    filtered = []

    for policy in policies:
        control_seq = policy[:, factor_idx]

        if control_seq[0] == 3:
            continue

        slot_actions = control_seq[control_seq != 3]
        if len(slot_actions) != len(np.unique(slot_actions)):
            continue

        policy[:, [1, 2, 3]] = policy[:, [4]]
        filtered.append(policy)

    return filtered


def build_agent(trials=10, message_passing_iterations=100):
    """Construct the discrete handover agent using the PyAIF v0.1 API."""

    (
        A,
        B,
        C,
        D,
        num_states,
        _,
        _,
        control_fac_idx,
        temporal_horizon,
    ) = create_generative_model()
    controls_dim = [1, 1, 1, 1, 4, 1, 1]
    policies = filter_policies(
        utils.construct_policies(
            num_states,
            controls_dim,
            temporal_horizon - 1,
            control_fac_idx,
        )
    )
    model = GenerativeModel(
        B=B,
        D=D,
        controls_dim=controls_dim,
        controllable_factors=control_fac_idx,
        policies=policies,
    )
    likelihood = CategoricalLikelihood(A=A, preferences=C)
    inference = DeepTemporalInference(
        horizon=temporal_horizon,
        message_passing_iterations=message_passing_iterations,
    )
    agent = ActiveInfAgent(
        model=model,
        likelihood=likelihood,
        inference=inference,
        trials=trials,
        policy_pruning=False,
        learning_A=False,
        learning_D=True,
        learning_B=True,
        learning_C=True,
    )
    return agent, temporal_horizon


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

    for sim_id in range(1, NUM_SIMULATIONS + 1):
        print(f"Running simulation {sim_id}...")
        ainf_agent, Temp_horizon = build_agent(trials=TRIALS)
        
        env = SortingEnv(reliability=100)

        actionlist = []
        commandlist = []
        for trial in range(TRIALS):
            ainf_agent.store_parameters()
            ainf_agent.reset(trial=trial)
            obs = env.reset()
            a_action = 'ideal'
            for t in range(Temp_horizon):
                if t != 0:
                    obs = env.step(a_action)
                if t != Temp_horizon - 1:
                    commandlist.append(obs[4])
                ainf_agent.observe(obs[:7], time_step=t)
                ainf_agent.infer_states_custom(trial, t)
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
