import numpy as np
from PyAIF import utils, ActiveInfAgent
import matplotlib.pyplot as plt
import time

"""
## Hidden state factors ##

Current_Resolution: 0, 1, 2, 3
current_context: 0, 1, 2, 3
available_cpu: 0, 1, 2
"""

"""
## observation modalities ##
info_gain_proxy
expected_surprise
inference_latency
cpu_availability
"""

"""
## Controllable hidden state factors ##

Current_Resolution: 0, 1, 2, 3, do_nothing
"""

def create_generative_model():

    num_states = [4, 4, 3]
    num_obs = [1, 1, 1, 1]
    num_controls = [5, 1, 1]
    control_fac_idx = [0]
    Temp_horizon = 1
    
    B = utils.zeros_B_matrix(num_states, num_controls)
    # fill action 0 transistions for factor 0 / do_nothing
    B[0][:,:,4] = np.eye(num_states[0])

    B[0][:, :, 0] = np.array([[1., 1., 1., 1.],
                              [0., 0., 0., 0.],
                              [0., 0., 0., 0.],
                              [0., 0., 0., 0.]], dtype=np.float32)

    B[0][:, :, 1] = np.array([[0., 0., 0., 0.],
                              [1., 1., 1., 1.],
                              [0., 0., 0., 0.],
                              [0., 0., 0., 0.]], dtype=np.float32)
    
    B[0][:, :, 2] = np.array([[0., 0., 0., 0.],
                              [0., 0., 0., 0.],
                              [1., 1., 1., 1.],
                              [0., 0., 0., 0.]], dtype=np.float32)
    
    B[0][:, :, 3] = np.array([[0., 0., 0., 0.],
                              [0., 0., 0., 0.],
                              [0., 0., 0., 0.],
                              [1., 1., 1., 1.]], dtype=np.float32)
    
    # fill action 0 transistions for factor 1 / do_nothing
    B[1][:,:,0] = np.eye(num_states[1])

    # fill action 0 transistions for factor 2 / do_nothing
    B[2][:,:,0] = np.eye(num_states[2])

    D = utils.uniform_D_matrix(num_states)

    return B, D, num_states, num_obs, num_controls, control_fac_idx, Temp_horizon
        
class MetaAgent():
    def __init__(self):

        B, D, num_states, num_obs, num_controls, control_fac_idx, Temp_horizon = create_generative_model()
        
        policies = utils.construct_policies(num_states, num_controls, 1, control_fac_idx)
        self.meta_agent = ActiveInfAgent(states_dim=num_states, obs_dim=num_obs, controls_dim=num_controls,
                                    controlable_states=control_fac_idx, planning_depth=Temp_horizon,
                                    number_of_msg_passing=30, trials=None, B=B, D=D,
                                    policies=policies, policy_pruning=False, learning_A=False, learning_D=False, learning_B=False, learning_C=False,
                                    continous_obs=True, lm_name="meta", mod_dependency=[[1], [0,1], [0,2], [2]], pref_dep=[[0,1]], obs_limits=None, learning_rate=0.1, forgeting_rate=0.95,
                                    obstacles_dic=None)
        self.meta_agent.store_parameters()
        self.meta_agent.normalize_columns()
        self.meta_agent.initialize_variables()

    def run_meta_inference(self, res_idx, obs):
        self.meta_agent.infer_states(0, 0, res_idx, obs)
        self.meta_agent.infer_policies(0, 0)
        chosen_action, _ = self.meta_agent.choose_action(0, 0)
        print(f"Chosen action: {chosen_action[0]}")
        self.meta_agent.perform_learning(0, 0)
        return chosen_action[0]

        

