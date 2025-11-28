import numpy as np
from PyAIF import utils, ActiveInfAgent
import copy
import os
from collections import deque

num_states_GP = [3, 2]
num_obs_GP = [2, 2, 4]
num_controls_GP = [3, 1]

def reset_GP():
    global A_GP, B_GP, D_GP
    # Create random arrays for A, B, C. Later these arrays will be filled!
    A_GP = utils.uniform_A_matrix(num_obs_GP, num_states_GP) # create sensory likelihood (A matrix) # matrix encodes the probability of
    B_GP = utils.uniform_B_matrix(num_states_GP, num_controls_GP) # create transition likelihood (B matrix)
    D_GP = utils.uniform_D_matrix(num_states_GP)

    D_GP[0] = np.array([1, 0])
    D_GP[1] = np.array([0, 1]) #trust #no trust

    #__HumanLocation_____#

    for i in range(num_states_GP[1]):

            A_GP[0][:,:,i] = np.array([[0.9, 0.1, 0.9], #Cargo_workplace
                                       [0.1, 0.9, 0.1]]) #Handlebars_workplace
        
    #__HumanHandLocation_____#

    for i in range(num_states_GP[1]):

            A_GP[1][:,:,i] = np.array([[0, 0, 0.9], #workplace
                                       [1, 1, 0.1]]) #stretchout
        
    #__HumanVoiceCommand_____#

    A_GP[2][:,:,1] = np.array([[0.8, 0.2, 0], #cargo
                               [0.2, 0.8, 0], #handlebars
                               [0, 0, 0.8], #wait
                               [0, 0, 0.2]]) #nothing

    A_GP[2][:,:,0] = np.array([[0, 0, 0], #cargo
                                [0, 0, 0], #handlebars
                                [0, 0, 0], #wait
                                [1, 1, 1]]) #nothing
    
    B_GP[0][:,:,0] = np.array([[0, 0, 0],
                               [0, 1, 0],
                               [1, 0, 1]])

    B_GP[0][:,:,1] = np.array([[1, 0, 0],
                            [0, 0, 0],
                            [0, 1, 1]])

    B_GP[0][:,:,2] = np.array([[1, 0, 0.50],
                            [0, 1, 0.00],
                            [0, 0, 0.50]])

    B_GP[1][:,:,0] = np.array([[0, 0],
                               [1, 1]])



def update_D_GP(trust_param):
    D_GP[0] = np.array([1, 0])
    D_GP[1] = np.array([trust_param, 1-trust_param]) #trust #no trust

def update_A_GP(trial):
    if trial > 99:
        #__HumanLocation_____#

        for i in range(num_states_GP[1]):

                A_GP[0][:,:,i] = np.array([[0.9, 0.1, 0.1], #Cargo_workplace
                                            [0.1, 0.9, 0.9]]) #Handlebars_workplace

def update_B_GP(trial, trust_param):

    trust_trans_prob = max(0.0, trust_param - 0.2)

    B_GP[1][:,:,0] = np.array([[trust_trans_prob, trust_trans_prob],
                            [1-trust_trans_prob, 1-trust_trans_prob]])

    if trial > 99:
        B_GP[0][:,:,2] = np.array([[1, 0, 0.00],
                                   [0, 1, 0.45],
                                   [0, 0, 0.55]])

def update_params(trust_param, command_param, true_state, action, obs, decision_history=None, window_size=10):
    """
    trust_param: current trust [0,1]
    command_param: unused
    true_state: ground-truth action
    action: agent's chosen action
    obs: observation vector, obs[2] == 3 means no instruction
    decision_history: deque of recent correctness values (1=correct, 0=wrong)
    """
    if decision_history is None:
        decision_history = deque(maxlen=window_size)

    expected_action = int(true_state)
    correct = int(action == expected_action)
    instruction_given = obs[2] != 3
    decision_history.append(correct)

    # Recent statistics
    accuracy = np.mean(decision_history)
    volatility = sum(decision_history[i] != decision_history[i - 1] for i in range(1, len(decision_history))) / max(1, len(decision_history) - 1)
    last_outcome = decision_history[-1]

    # Base rates
    gain_base = 0.03 + 0.04 * accuracy
    loss_base = 0.06 + 0.05 * (1 - accuracy)

    if instruction_given:
        if correct:
            trust_param += gain_base * 0.8 * (1 - trust_param)
        else:
            trust_param -= loss_base * 1.2 * trust_param
    else:
        if correct:
            trust_param += gain_base * 1.5 * (1 - trust_param)
        else:
            trust_param -= loss_base * 0.7 * trust_param

    trust_param = np.clip(trust_param, 0, 1)
    #print(f"obs: {obs}, expected: {expected_action}, action: {action}, trust: {trust_param:.3f}")
    return trust_param, command_param, correct


def get_true_state(trial, t, action_idx=None):
    for factor_idx in range(len(D_GP)):
        if t == 0:
            prob_state = D_GP[factor_idx]
        else:
            if factor_idx in control_fac_idx:
                prob_state = B_GP[factor_idx][:,int(true_states[trial, t-1, factor_idx]), action_idx]
            else:
                prob_state = B_GP[factor_idx][:,int(true_states[trial, t-1, factor_idx]), 0]
        true_states[trial, t, factor_idx] = np.searchsorted(np.cumsum(prob_state), np.random.rand())
        #true_states[trial, t, factor_idx] = np.argmax(prob_state)
    #print (f"state at time {t} is {true_states[t, :]}")
    return true_states

def get_obs(trial, t, true_states):
    for modality_idx in range(len(A_GP)):       
        outcomes[trial, t, modality_idx] = np.searchsorted(np.cumsum(A_GP[modality_idx][:,int(true_states[trial, t, 0]), int(true_states[trial, t, 1])]), np.random.rand())
    #print (f"Observation at time {t} is {outcomes[t, :]}")
    return outcomes[trial, t, :]

################___________Generative_model_____________################

num_states = [3, 2]
num_obs = [2, 2, 4]
num_controls = [3, 1]
control_fac_idx = [0]

def reset_Gmodel():
    global A, B, D, C
    A = utils.uniform_A_matrix(num_obs_GP, num_states_GP) # create sensory likelihood (A matrix) # matrix encodes the probability of
    B = utils.uniform_B_matrix(num_states_GP, num_controls_GP) # create transition likelihood (B matrix)
    D = utils.uniform_D_matrix(num_states_GP)
    C = utils.zero_C_matrix(num_obs, Temp_horizon)

    A = copy.deepcopy(A_GP)*0.1

    #__HumanLocation_____#

    for i in range(num_states_GP[1]):

            A[0][:,:,i] = np.array([[0.2, 0.1, 0.1], #Cargo_workplace
                                        [0.1, 0.2, 0.1]]) #Handlebars_workplace
        
    #__HumanHandLocation_____#

    for i in range(num_states_GP[1]):

            A[1][:,:,i] = np.array([[0.1, 0.1, 0.2], #workplace
                                        [0.2, 0.2, 0.1]]) #stretchout

    #__HumanVoiceCommand_____#

    A[2][:,:,1] = np.array([[0.2, 0.1, 0.1], #cargo
                                [0.1, 0.2, 0.1], #handlebars
                                [0.1, 0.1, 0.2], #wait
                                [0.1, 0.1, 0.1]]) #nothing

    A[2][:,:,0] = np.array([[0.1, 0.1, 0.1], #cargo
                                [0.1, 0.1, 0.1], #handlebars
                                [0.1, 0.1, 0.1], #wait
                                [0.5, 0.5, 0.5]]) #nothing


    B[0][:,:,0] = np.array([[0.1, 0.1, 0.1],
                            [0.1, 0.1, 0.1],
                            [0.1, 0.1, 0.1]])

    B[0][:,:,1] = np.array([[0.1, 0.1, 0.1],
                            [0.1, 0.1, 0.1],
                            [0.1, 0.1, 0.1]])

    B[0][:,:,2] = np.array([[0.1, 0.1, 0.1],
                            [0.1, 0.1, 0.1],
                            [0.1, 0.1, 0.1]])

    B[1][:,:,0] = np.array([[0.1, 0.1],
                            [0.1, 0.1]])
                            

    D = copy.deepcopy(D_GP)

    D[0] = np.array([0.1, 0.1, 0.1])
    D[1] = np.array([0.1, 0.1])

    C[0] = np.array([
        [0.00, 0.00, 0.00, 0.00],
        [0.00, 0.00, 0.00, 0.00]
    ])

    C[1] = np.array([
        [0.00, 0.00, 0.00, 0.00],
        [0.00, 0.00, 0.00, 0.00]
    ]) 

    C[2] = np.array([
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0]
    ])



# Create the directory if it doesn't exist
save_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(save_dir, exist_ok=True)

# Number of simulation runs
NUM_SIMULATIONS = 1

for sim_id in range(1, NUM_SIMULATIONS + 1):
    print(f"Running simulation {sim_id}...")

    TRIALS = 300
    MODELS = 1
    Temp_horizon = 4
    ini_trust_param = 0
    ini_command_param = 1
    trust_param = ini_trust_param
    command_param = ini_command_param

    reset_GP()
    reset_Gmodel()
    ainf_agent = ActiveInfAgent(A=A, B=B, states_dim=num_states, obs_dim=num_obs, controls_dim=num_controls,
                                controlable_states=control_fac_idx, trial_length=Temp_horizon,
                                number_of_msg_passing=100, trials=TRIALS, D=D, C=C,
                                policies=False, policy_pruning=False, learning_A=True, learning_B=True,
                                learning_D=True)

    true_states = np.zeros([TRIALS, Temp_horizon, 2])
    outcomes = np.zeros([TRIALS, Temp_horizon, 3])
    trust = np.zeros([TRIALS, Temp_horizon])
    voice = np.zeros([TRIALS, Temp_horizon])
    selected_actions = np.zeros([TRIALS, Temp_horizon - 1])
    decisions = np.zeros([TRIALS, Temp_horizon])
    vfe_fa = np.empty(TRIALS, dtype=object)
    vfe_fb = np.empty(TRIALS, dtype=object)
    vfe_fd = np.empty(TRIALS, dtype=object)
    vfe_fe = np.empty(TRIALS, dtype=object)
    lr_rates = np.empty(TRIALS, dtype=object)
    fr_rates = np.empty(TRIALS, dtype=object)
    D_norm_trials = np.empty(TRIALS, dtype=object)
    B_norm_trials = np.empty(TRIALS, dtype=object)
    A_norm_trials = np.empty(TRIALS, dtype=object)
    G_policies_trials = np.empty(TRIALS, dtype=object)
    F_policies_trials = np.empty(TRIALS, dtype=object)
    B_infog_policies_trials = np.empty(TRIALS, dtype=object)

    update_D_GP(trust_param)
    update_B_GP(0, trust_param)
    update_A_GP(0)

    state = get_true_state(trial=0, t=0)

    for trial in range(TRIALS):
        print(f"running trial ({trial}/{TRIALS})............")
        if trial > 0:
            true_states[trial, 0, :] = copy.deepcopy(state[trial - 1, 3, :])

        ainf_agent.store_parameters()
        D_norm, B_norm, A_norm = ainf_agent.normalize_columns()
        ainf_agent.initialize_variables()
        D_norm_trials[trial] = D_norm
        B_norm_trials[trial] = B_norm
        A_norm_trials[trial] = A_norm

        update_B_GP(trial, trust_param)
        update_A_GP(trial)
        for t in range(Temp_horizon):
            if t != 0:
                state = get_true_state(trial, t, executable_actions)

            obs = get_obs(trial, t, state)
            ainf_agent.observations[trial, t, :] = np.array(obs)
            ainf_agent.infer_states(trial, t)
            G_policies, F_policies = ainf_agent.infer_policies(trial, t)
            mod_averages = ainf_agent.perform_modal_average(trial, t)
            chosen_action, _ = ainf_agent.choose_action(trial, t)

            if chosen_action is not None:
                executable_actions = chosen_action[0]
                trust_param, command_param, decision = update_params(trust_param, command_param, state[trial][t][0],
                                                           executable_actions, obs)
                
                update_D_GP(trust_param)
                update_B_GP(trial, trust_param)
                update_A_GP(trial)
                selected_actions[trial, t] = executable_actions
                decisions[trial, t] = decision
            trust[trial, t] = trust_param
            voice[trial, t] = command_param

        fa, fb, fd, fe, lr, fr = ainf_agent.perform_learning(trial)
        vfe_fa[trial] = copy.deepcopy(fa)
        vfe_fb[trial] = copy.deepcopy(fb)
        vfe_fd[trial] = copy.deepcopy(fd)
        vfe_fe[trial] = copy.deepcopy(fe)
        lr_rates[trial] = copy.deepcopy(lr)
        fr_rates[trial] = copy.deepcopy(fr)
        G_policies_trials[trial] = copy.deepcopy(G_policies)
        F_policies_trials[trial] = copy.deepcopy(F_policies)
        #B_infog_policies_trials[trial] = copy.deepcopy(B_infog_policies)

    data = {
        "true_states": true_states,
        "outcomes": outcomes,
        "trust": trust,
        "voice": voice,
        "selected_actions": selected_actions,
        "decisions": decisions,
        "vfe_fa": vfe_fa,
        "vfe_fb": vfe_fb,
        "vfe_fd": vfe_fd,
        "vfe_fe": vfe_fe,
        "lr_rates": lr_rates,
        "fr_rates": fr_rates,
        "D_norm": D_norm_trials,
        "B_norm": B_norm_trials,
        "A_norm": A_norm_trials,
        "G_policies": G_policies_trials,
        "F_policies": F_policies_trials,
        "model_averages": mod_averages
    }

    save_path = os.path.join(save_dir, f"aleotric_simulation_results_{sim_id}.npy")
    np.save(save_path, data)
    print(f"Simulation {sim_id} saved to {save_path}")