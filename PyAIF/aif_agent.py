import numpy as np
import math
import copy
import random
import os
import sys
import time
import string
from PyAIF import utils
from collections.abc import Iterable
from scipy.special import digamma, gammaln, loggamma
import matplotlib.pyplot as plt
import multiprocessing
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import shared_memory
from joblib import Parallel, delayed
from PyAIF.numerics import (
    categorical_kl_terms,
    dirichlet_kl,
    factor_dot,
    log_beta as numerical_log_beta,
    log_stable_additive,
    log_stable_object_array,
    log_stable_probability,
    one_hot,
    softmax as numerical_softmax,
    spm_dot as numerical_spm_dot,
    spm_psi as numerical_spm_psi,
    transpose_transition,
    wnorm,
)
from PyAIF.inference.shallow import (
    infer_shallow_policies,
    infer_shallow_states,
)
from PyAIF.inference.deep_temporal import infer_deep_temporal_states

EPS_VAL = 1e-16 # global constant for use in spm_log() function

def infer_states_single_policy(t, policy_idx, num_nmp, num_f, temp_hor, state_posteriors, obs_taus, A, B, D, policy, time_cost): #implimentation of the MMP
        depolarization = None
        F = None
        for nmp in range(num_nmp):  # Number of gradient descent iterations
            previous_F = F
            policy_F = previous_F
            F = 0
            for factor in range(num_f):
                third_msg = np.zeros(state_posteriors[0, factor].size)
                for tau in range(temp_hor):
                    depolarization = log_stable(state_posteriors[tau, factor])
                    if tau <= t:
                        # Third message
                        third_msg = expected_log_likelihood(obs_taus[tau], factor, tau, state_posteriors, num_f, A)                        
                    if tau == 0:
                        # First message
                        first_msg = log_stable(D[factor])
                        # Second message
                        action_tau = policy[tau, :]
                        qs_future = state_posteriors[tau+1, factor]
                        transposed_B = transpose_Bfa(B[factor][:, :, action_tau[factor]])
                        second_msg = log_stable(transposed_B.dot(qs_future))
                    
                    elif tau == temp_hor-1:
                        # First message
                        actions_tau_1 = policy[tau-1, :]
                        qs_prev = state_posteriors[tau-1, factor]
                        first_msg = log_stable(B[factor][:, :, actions_tau_1[factor]].dot(qs_prev))
                        # Second message
                        second_msg = np.zeros((D[factor]).shape)
                    else:
                        # First message
                        actions_tau_1 = policy[tau-1, :]
                        qs_prev = state_posteriors[tau-1, factor]
                        first_msg = log_stable(B[factor][:, :, actions_tau_1[factor]].dot(qs_prev))
                        # Second message
                        action_tau = policy[tau, :]
                        qs_future = state_posteriors[tau+1, factor]
                        transposed_B = transpose_Bfa(B[factor][:, :, action_tau[factor]])
                        second_msg = log_stable(transposed_B.dot(qs_future))

                    # Compute state prediction error
                    state_pred_err = 0.5*(first_msg + second_msg) + third_msg - depolarization
                    depolarization += state_pred_err/time_cost
                    #@NOTE equation of F in tbl 2 on page 19 of the paper and MATLAB line of code for this is different.
                    # Following is the implimentation from the MATLAB.
                    Fintermediate = (state_posteriors[tau, factor]).dot(-log_stable(state_posteriors[tau, factor]) + 0.5*(first_msg + second_msg) +third_msg)
                    F += Fintermediate
                    state_posteriors[tau, factor] = softmax(np.array(depolarization))     
            #Early stopping condition to exit gradient descent if minimum VFE reached!
            if nmp > 0 and previous_F is not None:
                if F - previous_F < np.exp(-8):
                    policy_F = previous_F
                    break
        return t, policy_idx, state_posteriors, policy_F

def softmax(x, axis = 0, gamma=1.0):
    return numerical_softmax(x, axis=axis, gamma=gamma)

def transpose_Bfa(B_fa):
    # @NOTE: this function is not correct
    return transpose_transition(
        B_fa,
        normalize=True,
        replace_nan=True,
    )

def expected_log_likelihood(obs, factor, tau, qs, num_f, A):
    log_likelihoods = np.zeros(qs[tau, factor].size)
    if obs is not None:
        for modal_idx, modality in enumerate(A):
            lnA = log_stable(np.take(modality, obs[modal_idx], axis=0))
            lnA = np.moveaxis(lnA, factor, -1)
            for fj in range(num_f):
                if fj != factor:
                    lnAs = np.tensordot(lnA, qs[tau, fj], axes=(0,0))
                    del lnA
                    lnA = lnAs
                    del lnAs
            log_likelihoods += lnA
    return log_likelihoods

def cell_md_dot_py(X, x):
    return factor_dot(X, x)

def log_stable(array, val=np.exp(-16)):
    """
    Adds small epsilon value to an array before natural logging it
    """
    return log_stable_additive(array, val=val)

# Reconstructs an object array (like A or C) from shared memory
def _reconstruct_object_array_from_shm(shm_info):
    shm_name = shm_info['name']
    metadata = shm_info['metadata']

    existing_shm = shared_memory.SharedMemory(name=shm_name)
    
    # Create a list of NumPy array views
    reconstructed_arrays = [None] * len(metadata) # Pre-allocate for efficiency

    for item_meta in metadata:
        idx = item_meta['idx']
        shape = tuple(item_meta['shape'])
        dtype = np.dtype(item_meta['dtype'])
        offset = item_meta['offset']
        
        arr_view = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf, offset=offset)
        reconstructed_arrays[idx] = arr_view
    
    # Don't close shm here; let the worker function manage its shm handles
    return reconstructed_arrays, existing_shm # Return views and the shm handle to

def _reconstruct_deeply_nested_object_array_from_shm(shm_info):
    """
    Reconstructs the original deeply nested NumPy object array structure from shared memory
    using the metadata generated by _create_shm_for_deeply_nested_object_array.
    Returns the reconstructed object array and the shared memory handle.
    """
    shm_name = shm_info['name']
    metadata = shm_info['metadata']
    original_outer_shape = tuple(shm_info['original_outer_shape'])

    existing_shm = shared_memory.SharedMemory(name=shm_name)
    
    # Create the top-level object array (all elements are None initially)
    reconstructed_array = np.empty(original_outer_shape, dtype=object)

    # Populate the object array by placing views from shared memory
    for item_meta in metadata:
        original_indices = item_meta['original_indices']
        shape = tuple(item_meta['shape'])
        dtype = np.dtype(item_meta['dtype'])
        offset = item_meta['offset']
        
        arr_view = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf, offset=offset)
        
        # Place the view into the correct position in the reconstructed object array
        # This requires setting elements using a tuple of indices
        reconstructed_array[original_indices] = arr_view
    
    return reconstructed_array, existing_shm

def _reconstruct_single_array_from_shm(shm_info):
    shm_name = shm_info['name']
    shape = tuple(shm_info['shape'])
    dtype = np.dtype(shm_info['dtype'])

    existing_shm = shared_memory.SharedMemory(name=shm_name)
    arr_view = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)
    return arr_view, existing_shm

# GLOBAL VARIABLES for worker processes (initialized by _worker_initializer)
_global_A_list = None
_global_B_list = None
_global_C_list = None
_global_D_list = None
_global_E_list = None
_global_P_reconstructed_nested_array = None
_global_Policies_reconstructed_nested_array = None
_global_shm_handles = [] # To keep references to SharedMemory objects so they don't get garbage collected/unlinked too early

def _worker_initializer(A_info, C_info, P_info):
    """
    This function is run once in each worker process when it starts up.
    It attaches to the shared memory and stores the reconstructed views in global variables.
    """
    global _global_A_list, _global_C_list, \
           _global_P_reconstructed_nested_array, \
           _global_shm_handles

    #print(f"Worker {os.getpid()}: Initializing shared memory views.")

    try:
        # Reconstruct all the data once per worker process
        _global_A_list, shm_A = _reconstruct_object_array_from_shm(A_info)
        _global_shm_handles.append(shm_A)

        #_global_B_list, shm_B = _reconstruct_object_array_from_shm(B_info)
        #_global_shm_handles.append(shm_B)

        _global_C_list, shm_C = _reconstruct_object_array_from_shm(C_info)
        _global_shm_handles.append(shm_C)

        #_global_D_list, shm_D = _reconstruct_object_array_from_shm(D_info)
        #_global_shm_handles.append(shm_D)

        #_global_E_list, shm_E = _reconstruct_object_array_from_shm(E_info)
        #_global_shm_handles.append(shm_E)

        # Use _reconstruct_deeply_nested_object_array_from_shm for P and Policies
        _global_P_reconstructed_nested_array, shm_P = _reconstruct_deeply_nested_object_array_from_shm(P_info)
        _global_shm_handles.append(shm_P)

        #_global_Policies_reconstructed_nested_array, shm_Policies = _reconstruct_deeply_nested_object_array_from_shm(Policies_info)
        #_global_shm_handles.append(shm_Policies)

        #print(f"Worker {os.getpid()}: Shared memory views initialized successfully.")

    except Exception as e:
        print(f"!!!! CRITICAL ERROR IN WORKER INITIALIZER {os.getpid()} !!!!", file=sys.stderr)
        print(f"Exception during shared memory setup: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        # Attempt to close handles even if initialization fails
        for handle in _global_shm_handles:
            try: handle.close()
            except: pass
        raise # Re-raise to signal the main process that this worker is bad

def full_eval_policy_worker(args):
    # Don't need to unpack shm_info anymore
    t, policy_idx, temporal_horizon, num_factors, num_modalities, \
    learning_D, learning_A, learning_B = args

    # Access the already reconstructed data from global variables
    A_list = _global_A_list
    #B_list = _global_B_list
    C_list = _global_C_list
    #D_list = _global_D_list
    #E_list = _global_E_list
    P_list = _global_P_reconstructed_nested_array
    #policies_reconstructed_nested_array = _global_Policies_reconstructed_nested_array
    try:
        # Example: Risk calculation (adapted from your original risk_worker)
        risk_term = 0
        for timestep in range(t, temporal_horizon):
            for modality_idx in range(num_modalities):
                modality_A = A_list[modality_idx]
                modality_C = C_list[modality_idx]
                # P_list[policy_idx, timestep, :] needs to be adjusted based on the actual structure of P
                # If P is (num_policies, temporal_horizon, num_factors, factor_dim) as a single array,
                # then P_array[policy_idx, timestep, :] is correct.
                # If P_list means P[policy_idx] is itself an object array containing time steps/factors,
                # then you need to index into P_list further. Let's assume P is flattened into P_array
                # as in the previous example with _reconstruct_single_array_from_shm
                # For `P`, it's more likely a single array of (policies, time, factors, factor_dim),
                # so re-check if `_create_shm_for_object_array` is right for P or if it needs simpler `tobytes`.

                # Assuming P is a single large array:
                # If you use _reconstruct_single_array_from_shm for P:
                # P_array = P_list # Since _reconstruct_single_array_from_shm returns just the array view
                # expected_obs = cell_md_dot_py(modality_A, P_array[policy_idx, timestep, :])

                # If P is ALSO an object array of object arrays (like A/C):
                # P_policy_view = P_list[policy_idx] # This would be an array of time steps
                # P_timestep_view = P_policy_view[timestep] # This would be an array of factors
                # expected_obs = cell_md_dot_py(modality_A, P_timestep_view[:]) # This is getting complex!

                # For now, let's proceed assuming P is a large single array `P_array`
                # (which means _reconstruct_single_array_from_shm would be used for it in the worker)
                # You need to adjust based on the TRUE structure of self.policy_dep_posteriors.
                # Let's revert to assuming P is `P_array` that was made from _reconstruct_single_array_from_shm.
                # This needs a decision on how P is shared: either as one big array or as a list of arrays (if policies makes it object dtype).
                # Your `P_shm, self.P_shm_info = _create_shm_for_object_array(self.policy_dep_posteriors)` line suggests P is ALSO an object array.
                # If P is a `(num_policies, )` object array where each element is a (temporal_horizon, num_factors, factor_dim) array:
                P_policy_view = P_list[policy_idx] # This would be (temporal_horizon, num_factors, factor_dim)
                expected_obs = cell_md_dot_py(modality_A, P_policy_view[timestep, :])
                risk_term += expected_obs.dot(modality_C[:, timestep])


        # Example: Ambiguity calculation (adapted from your original ambiguity_worker)
        ambiguity_term = 0.0
        for tau in range(t, temporal_horizon):
            entropy_over_expected = 0.0
            expected_entropy = 0.0

            for modality_idx in range(num_modalities):
                A_mod = A_list[modality_idx]
                p_o_given_s = A_mod.copy()
                for factor_idx in reversed(range(num_factors)):
                    q_s = P_policy_view[tau, factor_idx] # Access from P_policy_view
                    p_o_given_s = np.tensordot(p_o_given_s, q_s, axes=(1 + factor_idx, 0))

                entropy_over_expected += -np.sum(p_o_given_s * log_stable(p_o_given_s))

                A_logA = A_mod * log_stable(A_mod)
                entropy_tensor = -np.sum(A_logA, axis=0)
                for factor_idx in reversed(range(num_factors)):
                    q_s = P_policy_view[tau, factor_idx]
                    entropy_tensor = np.tensordot(entropy_tensor, q_s, axes=([-1], [0]))

                expected_entropy += entropy_tensor

            ambiguity_tau = expected_entropy - entropy_over_expected
            ambiguity_term += ambiguity_tau

        final_g_val = risk_term + ambiguity_term
        return policy_idx, final_g_val

    except Exception as e:
        # ... (error handling for worker logic, as before) ...
        print(f"!!!! CRITICAL ERROR IN WORKER {os.getpid()} for policy {policy_idx} !!!!", file=sys.stderr)
        print(f"Exception: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        raise # Re-raise to propagate

class ActiveInfAgent:
    
    def __init__(
        self, states_dim=None, obs_dim=None, controls_dim=None, controlable_states=None,
        planning_depth=1, number_of_msg_passing = 100, learning_rate = 0.2,
        forgeting_rate = 0.99, trials = 100, alpha = 512, zeta = 0.01, timeconst = 1,  A=None, B=None, D=None,
        C=None, E=None, policies=False, policy_pruning = False, learning_D = False, learning_A = False,
        learning_B = False, learning_E = False, learning_C = False, learning_window = 4,
        continous_obs = False, lm_name = None, mod_dependency = None, pref_dep = None, factor_dep = None, obs_limits = None,
        obstacles_dic = None, action_selection = "marginal", *,
        model=None, likelihood=None, inference=None,
    ):
        component_values = (model, likelihood, inference)
        using_component_api = any(value is not None for value in component_values)
        if using_component_api:
            if not all(value is not None for value in component_values):
                raise ValueError(
                    "model, likelihood, and inference must be provided together."
                )

            from PyAIF.likelihoods import CategoricalLikelihood

            if not isinstance(likelihood, CategoricalLikelihood):
                raise TypeError(
                    "PyAIF v0.1 supports CategoricalLikelihood only. "
                    "Continuous likelihoods are planned for v0.2."
                )

            likelihood.validate_states(model.states_dim)
            states_dim = list(model.states_dim)
            obs_dim = list(likelihood.obs_dim)
            controls_dim = list(model.controls_dim)
            controlable_states = list(model.controllable_factors)
            planning_depth = inference.horizon
            number_of_msg_passing = inference.message_passing_iterations
            A = likelihood.A
            C = likelihood.preferences
            B = model.B
            D = model.D
            mod_dependency = [
                list(dependencies)
                for dependencies in likelihood.modality_dependencies
            ]
            if model.policies is not None:
                policies = model.policies
            continous_obs = False

            self.model = model
            self.likelihood = likelihood
            self.inference = inference

        required_dimensions = {
            "states_dim": states_dim,
            "obs_dim": obs_dim,
            "controls_dim": controls_dim,
            "controlable_states": controlable_states,
        }
        missing = [name for name, value in required_dimensions.items() if value is None]
        if missing:
            raise ValueError(f"Missing required agent configuration: {', '.join(missing)}")
        if continous_obs:
            raise NotImplementedError(
                "PyAIF v0.1 supports categorical observations only. "
                "Continuous likelihood components are planned for v0.2."
            )

        self.factor_dep = factor_dep
        self.pref_dep = pref_dep
        if mod_dependency is None:
            mod_dependency = [
                list(range(len(states_dim)))
                for _ in range(len(obs_dim))
            ]

        # 1. Initialize the plot OUTSIDE the loop
        # Initialize plot outside loop
        #plt.ion()
        #self.fig, self.ax = plt.subplots(figsize=(6, 5))
        # Create the colorbar once with the fixed 0-1 range
        #self.sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(vmin=0, vmax=1))
        #self.cbar = self.fig.colorbar(self.sm, ax=self.ax, label='Probability')

        if planning_depth > 1:
            self.deep_inference = True
        else:
            self.deep_inference = False

        if continous_obs == True:
            self.continous_obs = True
        else:
            self.continous_obs = False

        if self.deep_inference:
            # Construct policies
            if policies is False or policies is None:
                self.policies = utils.construct_policies(states_dim, controls_dim, planning_depth-1, controlable_states)
            else:
                self.policies = policies        
            self.policy_pruning = policy_pruning
            self.num_modalities = len(obs_dim)
            self.num_policies = len(self.policies)
            if self.factor_dep:
                self.states_dim = [math.prod(states_dim[i] for i in dep) for dep in factor_dep]
                """
                state_to_factor = {}
                for factor_idx, states in enumerate(factor_dep):
                    for s in states:
                        state_to_factor[s] = factor_idx
                self.mod_dep = [sorted(list(set(state_to_factor[s] for s in dep))) for dep in mod_dependency]
                """
                
            else:
                self.states_dim = states_dim
            self.mod_dep = mod_dependency
            self.num_factors = len(self.states_dim)
            self.obs_dim = obs_dim
            self.controls_dim = controls_dim
            self.num_trials = trials


            if not self.continous_obs:

                self.pA = A
                self.pB = B
                self.pC = C
                for num_el in range(len(C)):
                    self.pC[num_el] += 1/32 
                self.pD = D
                self.pE = E if E else self.create_object_tensor('ones', 1, last_dim = [len(self.policies)])

                self.learning_A = learning_A
                self.learning_B = learning_B
                self.learning_E = learning_E
                self.learning_D = learning_D
                self.learning_C = learning_C

                if self.learning_B:
                    self.pB_0 = copy.deepcopy(self.pB)
                    self.pB_prior = copy.deepcopy(self.pB)
                    self.pB_complexity = copy.deepcopy(self.pB)

                if self.learning_D:
                    self.pD_0 = copy.deepcopy(self.pD)
                    self.pD_prior = copy.deepcopy(self.pD)
                    self.pD_complexity = copy.deepcopy(self.pD)
                if self.learning_E:
                    self.pE_0 = copy.deepcopy(self.pE)
                if self.learning_C:
                    self.pC_0 = copy.deepcopy(self.pC)
                
                # Store the shared memory objects and their info
                self.A_shm = None
                self.A_shm_info = None # Stores {name: '...', metadata: [...]}
                self.B_shm = None
                self.B_shm_info = None
                self.D_shm = None
                self.D_shm_info = None
                self.E_shm = None
                self.E_shm_info = None
                self.C_shm = None
                self.C_shm_info = None
                self.policies_shm = None
                self.policies_shm_info = None
                self.policy_dep_posteriors_shm = None
                self.policy_dep_posteriors_shm_info = None

                self.temporal_horizon = planning_depth
                self.planning_from = 0
                self.planning_to = self.temporal_horizon
                self.controlable_states = controlable_states
                self.number_of_msg_passing = number_of_msg_passing
                self.learning_rate = learning_rate
                self.forgeting_rate = forgeting_rate
                self.policy_dep_posteriors = None #self.create_object_tensor(last_dim=self.states_dim)
                #self.joint_policy_dep_posteriors = self.create_object_tensor('zeros', len(self.policies), self.temporal_horizon)
                self.posterior_pi = None #self.create_object_tensor('uniform', len(self.policies), self.temporal_horizon)
                self.action_posteriors = None #self.create_object_tensor('zeros', self.num_factors, self.temporal_horizon - 1)
                #for factor_idx in range(self.num_factors):
                    #if controls_dim[factor_idx] == 1:
                        #self.action_posteriors[factor_idx, :] = np.ones([1, self.temporal_horizon - 1])
                self.bayesian_mod_avg = self.create_object_tensor(
                    'zeros',
                    self.temporal_horizon,
                    self.num_factors,
                    last_dim=self.states_dim,
                )
                self.Fd = self.create_object_tensor('zeros', 1, last_dim = [self.num_factors])
                self.Fb = copy.deepcopy(self.Fd)
                self.Fa = self.create_object_tensor('zeros', 1, last_dim = [self.num_modalities])
                self.Fe = 0
                self.alpha = alpha
                self.zeta = zeta
                #self.action_selection = "deterministic" # use "stochastic" for action selection with some randomness
                #self.action_selection = "random"
                self.action_selection = action_selection # use "stochastic" for action selection with some randomness
                
                self.timeconst = timeconst #time constant for gradient descent
                self.gamma_0 = None
                self.posterior_beta = None
                self.total_dop_res = self.number_of_msg_passing * self.temporal_horizon
                #self.gamma_update = self.create_object_tensor('zeros', self.num_trials, self.total_dop_res)

                self.previous_lr = copy.deepcopy(self.learning_rate)
            else:
                self.temporal_horizon = planning_depth
                self.planning_from = 0
                self.planning_to = self.temporal_horizon
                self.controlable_states = controlable_states
                self.number_of_msg_passing = number_of_msg_passing
                self.learning_rate = learning_rate
                self.forgeting_rate = forgeting_rate
                self.policy_dep_posteriors = None #self.create_object_tensor(last_dim=self.states_dim)
                self.posterior_pi = None #self.create_object_tensor('uniform', len(self.policies), self.temporal_horizon)
                self.action_posteriors = None #self.create_object_tensor('zeros', self.num_factors, self.temporal_horizon - 1)
                self.observations = {}
                self.bayesian_mod_avg = self.create_object_tensor('zeros', self.temporal_horizon, self.num_factors, last_dim=self.states_dim)
                self.alpha = alpha
                self.zeta = zeta
                self.action_selection = action_selection # use "stochastic" for action selection with some randomness
                
                self.timeconst = timeconst #time constant for gradient descent
                self.gamma_previous = 1 
                self.beta_posterior = 1
                self.beta_prior = 1
            
                self.pB = B
                self.B = copy.deepcopy(self.pB)
                self.transposed_B = self._transpose_B_matrix() 
                self.pD = D
                self.D = copy.deepcopy(self.pD)
                self.pE = E if E else self.create_object_tensor('ones', 1, last_dim = [len(self.policies)])
                self.E = copy.deepcopy(self.pE)

                self.learning_A = learning_A
                self.learning_B = learning_B
                self.learning_C = learning_C
                self.learning_E = learning_E
                self.learning_D = learning_D

                if self.learning_B:
                    self.pB_0 = copy.deepcopy(self.pB)
                    self.pB_prior = copy.deepcopy(self.pB)
                    self.pB_complexity = copy.deepcopy(self.pB)

                if self.learning_D:
                    self.pD_0 = copy.deepcopy(self.pD)
                    self.pD_prior = copy.deepcopy(self.pD)
                    self.pD_complexity = copy.deepcopy(self.pD)
                if self.learning_E:
                    self.pE_0 = copy.deepcopy(self.pE)



        else:
            #performs shallow inference

            if not self.continous_obs:

                self.policies = utils.construct_policies(states_dim, controls_dim, 1, controlable_states)

                self.num_factors = len(states_dim)
                self.num_modalities = len(obs_dim)
                self.num_policies = len(self.policies)
                self.states_dim = states_dim
                self.obs_dim = obs_dim
                self.controls_dim = controls_dim
                self.controlable_states = controlable_states
                self.num_trials = trials
                self.num_iterations = number_of_msg_passing
                self.mod_dep = mod_dependency

                self.pA = A
                self.pB = B
                self.pC = C
                self.A = copy.deepcopy(self.pA)
                self.C = copy.deepcopy(self.pC)
                self.B = copy.deepcopy(self.pB)
                for num_el in range(len(C)):
                    self.pC[num_el] += 1/32 
                self.pD = D
                self.D = copy.deepcopy(self.pD)
                self.pE = E if E else np.ones(self.num_policies)
                self.E = copy.deepcopy(self.pE)

                self.learning_A = learning_A
                self.learning_B = learning_B
                self.learning_E = learning_E
                self.learning_D = learning_D
                self.learning_C = learning_C

                self.learning_window = learning_window
                self.learning_rate = learning_rate
                self.forgeting_rate = forgeting_rate
                

                if self.learning_A:
                    self.pA_0 = copy.deepcopy(self.pA)
                    self.pA_prior = copy.deepcopy(self.pA)
                    self.pA_complexity = copy.deepcopy(self.pA)

                if self.learning_B:
                    self.pB_0 = copy.deepcopy(self.pB)
                    self.pB_prior = copy.deepcopy(self.pB)
                    self.pB_complexity = copy.deepcopy(self.pB)

                if self.learning_D:
                    self.pD_0 = copy.deepcopy(self.pD)
                    self.pD_prior = copy.deepcopy(self.pD)
                    self.pD_complexity = copy.deepcopy(self.pD)
                if self.learning_E:
                    self.pE_0 = copy.deepcopy(self.pE)
                if self.learning_C:
                    self.pC_0 = copy.deepcopy(self.pC)

                self.alpha = alpha
                self.action_selection = action_selection # use "stochastic" for action selection with some randomness

            else:
                self.policies = policies
                self.num_factors = len(states_dim)
                self.num_modalities = len(obs_dim)
                self.num_policies = len(self.policies)
                self.states_dim = states_dim
                self.obs_dim = obs_dim
                self.controls_dim = controls_dim
                self.controlable_states = controlable_states
                self.num_trials = trials
                self.num_iterations = number_of_msg_passing
                self.mod_dep = mod_dependency

                self.pB = B
                self.pD = D
                self.pE = E if E else np.ones(self.num_policies)

                self.learning_A = learning_A
                self.learning_B = learning_B
                self.learning_C = learning_C
                self.learning_E = learning_E
                self.learning_D = learning_D

                if self.learning_B:
                    self.pB_0 = copy.deepcopy(self.pB)
                    self.pB_prior = copy.deepcopy(self.pB)
                    self.pB_complexity = copy.deepcopy(self.pB)

                if self.learning_D:
                    self.pD_0 = copy.deepcopy(self.pD)
                    self.pD_prior = copy.deepcopy(self.pD)
                    self.pD_complexity = copy.deepcopy(self.pD)
                if self.learning_E:
                    self.pE_0 = copy.deepcopy(self.pE)

                self.learning_window = learning_window
                self.learning_rate = learning_rate
                self.forgeting_rate = forgeting_rate

                self.alpha = alpha
                self.action_selection = action_selection # use "stochastic" for action selection with some randomness

        if not hasattr(self, "gamma_previous"):
            self.gamma_previous = 1.0
        if not hasattr(self, "beta_prior"):
            self.beta_prior = 1.0
        if not hasattr(self, "beta_posterior"):
            self.beta_posterior = 1.0


    def reset(self, trial=0, normalize=True):
        """Reset transient beliefs for the component-based public API.

        The historical ``initialize_variables`` and ``normalize_columns``
        methods remain available while examples migrate to this lifecycle.
        """

        if normalize:
            self.normalize_columns()
        self.initialize_variables()
        self._current_trial = int(trial)
        self._current_time = 0
        self._pending_observation = None
        return self

    def observe(self, observation, time_step=None):
        """Record one multimodal observation for the current time step."""

        observation = np.asarray(observation)
        if observation.shape != (self.num_modalities,):
            raise ValueError(
                f"Expected {self.num_modalities} modality values; "
                f"received shape {observation.shape}."
            )

        if time_step is not None:
            self._current_time = int(time_step)

        self._pending_observation = observation.copy()
        if self.deep_inference:
            self.observations[self._current_time] = observation.copy()
        return self

    def select_action(self):
        """Select an action using the current policy posterior."""

        action, _ = self.choose_action(self._current_trial, self._current_time)
        self._current_time += 1
        return action

    def learn(self):
        """Apply configured parameter learning at the current agent time."""

        return self.perform_learning(
            self._current_trial,
            actual_t=self._current_time,
        )


    def initialize_variables(self):
        if self.deep_inference:
            self.policy_dep_posteriors = self.create_object_tensor(last_dim=self.states_dim)
            #self.single_policy_dep_posteriors = copy.deepcopy(self.policy_dep_posteriors[0,:,:])       
            self.posterior_pi = self.create_object_tensor('zeros', self.num_policies)
            #self.posterior_updates = self.create_object_tensor('NaN', self.total_dop_res, last_dim = [len(self.policies)])
            self.prior_pi = self.create_object_tensor('zeros', self.num_policies)
            self.action_posteriors = self.create_object_tensor('zeros', self.num_factors)       
            #self.action_confidance = self.create_object_tensor('ones', self.temporal_horizon - 1, self.num_factors, last_dim=self.controls_dim)
            #self.vfe_ft = self.create_object_tensor('zeros', len(self.policies), self.temporal_horizon, self.number_of_msg_passing, self.temporal_horizon, self.num_factors)
            #self.normalized_firing_rates = self.create_object_tensor('NaN', len(self.policies), self.temporal_horizon, self.temporal_horizon, self.number_of_msg_passing, last_dim=self.states_dim)
            #self.prediction_error = self.create_object_tensor('NaN', len(self.policies), self.temporal_horizon, self.temporal_horizon, self.num_factors)
            self.F_policy = self.create_object_tensor('zeros', self.num_policies)
            self.G_policy = self.create_object_tensor('zeros', self.num_policies)
            self.disparity_nu = self.create_object_tensor('zeros', self.temporal_horizon, self.num_modalities, last_dim = self.obs_dim) 
            self.chosen_policy = self.create_object_tensor('NaN', self.temporal_horizon) 
            self.expected_obs_chosen = self.create_object_tensor('NaN', self.temporal_horizon, self.num_modalities, last_dim=self.obs_dim)
            self.policy_dep_expected_obs = self.create_object_tensor(
                'NaN',
                self.num_policies,
                self.temporal_horizon,
                self.num_modalities,
                last_dim=self.obs_dim,
            )
            self.planning_from = 0
            self.planning_to = self.temporal_horizon
            self.observations = {}
        else:
            self.posteriors = self.create_object_tensor('uniform', self.num_factors, last_dim=self.states_dim)
            self.posteriors_cache = self.create_object_tensor('NaN', self.learning_window, self.num_factors, last_dim=self.states_dim)
            self.observations_cache = self.create_object_tensor('NaN', self.learning_window, self.num_modalities)
            self.G_policy = self.create_object_tensor('zeros', self.num_policies)
            self.action_posteriors = self.create_object_tensor('zeros', self.num_factors)
            self.action_posteriors_cache = self.create_object_tensor('NaN', self.num_factors, self.learning_window)

    def normalize_columns(self):
        if not self.continous_obs:
            self.A = self._normalize_colums(self.pA)
            self.B = self._normalize_colums(self.pB)
            B_T = self._transpose_B_matrix()
            self.transposed_B = self._normalize_colums(B_T)
            self.D = self._normalize_colums(self.pD)
            self.E = self._normalize_colums(self.pE)
            self.C = self.softmax_whole(self.pC)
            for modality_idx in range(self.num_modalities):
                self.C[modality_idx] = self.log_stable(self.C[modality_idx])
        else:
            self.B = self._normalize_colums(self.pB)
            self.D = self._normalize_colums(self.pD)
            self.E = self._normalize_colums(self.pE)
            B_T = self._transpose_B_matrix()
            self.transposed_B = self._normalize_colums(B_T)

    def _normalize_colums(self, matrix):
        matrix_copy = copy.deepcopy(matrix)
        matrix_copy = matrix_copy
        for modality_idx, modality in enumerate(matrix_copy):
            if modality.ndim == 1:
                matrix_copy[modality_idx] = np.divide(modality, modality.sum(axis=0))
            elif modality.ndim == 0:
                matrix_copy = np.divide(matrix_copy, matrix_copy.sum(axis=0))
                return matrix_copy
            else:
                modality_shape = modality.shape
                for index in np.ndindex(modality_shape):
                    sliced_index = (slice(None), slice(None)) + index[2:]
                    modality_to_norm = matrix_copy[modality_idx][sliced_index]
                    matrix_copy[modality_idx][sliced_index] = np.divide(modality_to_norm, modality_to_norm.sum(axis=0))
        return matrix_copy
    
    def _normalize_columns_min_max(self, matrix, min_val=0.1, max_val=3.0):
        matrix_copy = copy.deepcopy(matrix)

        for modality_idx, modality in enumerate(matrix_copy):
            if modality.ndim == 1:
                min_m = modality.min()
                max_m = modality.max()
                denom = (max_m - min_m) if max_m != min_m else 1.0
                matrix_copy[modality_idx] = min_val + (modality - min_m) * (max_val - min_val) / denom

            elif modality.ndim == 0:
                min_m = matrix_copy.min()
                max_m = matrix_copy.max()
                denom = (max_m - min_m) if max_m != min_m else 1.0
                return min_val + (matrix_copy - min_m) * (max_val - min_val) / denom

            else:
                modality_shape = modality.shape
                for index in np.ndindex(modality_shape[2:]):
                    sliced_index = (slice(None), slice(None)) + index
                    submatrix = matrix_copy[modality_idx][sliced_index]
                    min_m = submatrix.min()
                    max_m = submatrix.max()
                    denom = (max_m - min_m) if max_m != min_m else 1.0
                    matrix_copy[modality_idx][sliced_index] = min_val + (submatrix - min_m) * (max_val - min_val) / denom

        return matrix_copy

    
    def store_parameters(self):
        if not self.continous_obs:
            if self.learning_A == True:
                for modality_idx, modality in enumerate(self.pA):
                    self.pA_prior[modality_idx] = copy.deepcopy(modality)
                    self.pA_complexity[modality_idx] = self.wnorm_new(self.pA_prior[modality_idx])*(self.pA_prior[modality_idx] > 0)

            if self.learning_D == True:
                for factor_idx, factor in enumerate(self.pD):
                    self.pD_prior[factor_idx] = copy.deepcopy(factor)
                    self.pD_complexity[factor_idx] = self.wnorm_new(self.pD_prior[factor_idx])

            if self.learning_B == True:
                for factor_idx, factor in enumerate(self.pB):
                    self.pB_prior[factor_idx] = copy.deepcopy(factor)
                    self.pB_complexity[factor_idx] = self.wnorm_new(self.pB_prior[factor_idx])*(self.pB_prior[factor_idx] > 0)            

    def infer_states_multiprocessing(self, trial, t):
        num_nmp = self.number_of_msg_passing
        num_f = self.num_factors
        temp_hor = self.temporal_horizon
        obs_taus = self.observations[:, :]
        A = self.A
        B = self.B
        D = self.D
        timeconst = self.timeconst

        # Create a list of arguments for each policy
        tasks = []
        for policy_idx, policy in enumerate(self.policies):
            # Pass a deep copy of the initial state_posteriors for each process
            # Each process needs its own independent copy to modify
            initial_state_posteriors_copy = copy.deepcopy(self.single_policy_dep_posteriors)
            #result_t, result_policy_idx, result_state_posteriors, result_policy_F = infer_states_single_policy(t, policy_idx, num_nmp, num_f, temp_hor,
                          #initial_state_posteriors_copy, obs_taus, A, B, D, policy, timeconst)
            tasks.append((t, policy_idx, num_nmp, num_f, temp_hor,
                          initial_state_posteriors_copy, obs_taus, A, B, D, policy, timeconst))

        # Determine the number of processes to use
        # It's generally good practice not to use more processes than CPU cores
        num_processes = multiprocessing.cpu_count()
        # You can also limit this if you have a very large number of policies but fewer cores,
        # or if you want to leave some cores free for other tasks.
        # e.g., num_processes = min(multiprocessing.cpu_count(), len(self.policies))

        # Use a Pool to manage the processes
        with multiprocessing.Pool(processes=num_processes) as pool:
            # map applies the function to each item in the iterable (tasks)
            # The order of results will correspond to the order of tasks
            results = pool.starmap(infer_states_single_policy, tasks)

        # Collect results and update self attributes
        # Ensure F_policy for the current time step 't' is initialized

        for result_t, result_policy_idx, result_state_posteriors, result_policy_F in results:
            self.policy_dep_posteriors[result_policy_idx,:,:] = result_state_posteriors
            self.F_policy[result_t][result_policy_idx] = result_policy_F

    def step_time(self, t):
        """
            Updates the planning window for fixed planning depth, once a planning window is completed
            and trial/epoch is not finnished. This function should call at the begining of each time step
            in the main loop of the experiment.
        """
        if self.deep_inference:
            if t%self.temporal_horizon == self.temporal_horizon-1:
                self.planning_from = t + 1 
                self.planning_to = t + 1 + self.temporal_horizon
                self.observations = {}


    def update_goal_plot(self, t, policy_idx=0):
        self.ax.clear()
        
        # 1. Extract marginals and compute joint
        joint_1d = self.policy_dep_posteriors[policy_idx, t % self.temporal_horizon, 2]  #

        joint = joint_1d.reshape(int(np.sqrt(self.states_dim[2])), int(np.sqrt(self.states_dim[2])))
        #print(np.max(joint))
        # 2. Plot with fixed vmin/vmax
        # joint.T ensures Factor 2 is X (columns) and Factor 3 is Y (rows)
        im = self.ax.pcolormesh(joint, shading='auto', cmap='coolwarm', vmin=0, vmax=1)
        self.ax.invert_yaxis()
        
        # 3. Aesthetics
        self.ax.set_title(f"Goal Belief Map (Step: {t})")
        self.ax.set_xlabel("Goal X State")
        self.ax.set_ylabel("Goal Y State")
        
        # Optional: Add a marker for the true goal if you have the coordinates
        # self.ax.plot(true_x, true_y, 'k*', markersize=15) 

        plt.draw()
        plt.pause(0.01)
    
    def infer_states(self, trial=None, t=None, res_idx=None, obs=None, dF_tol=None):
        if trial is None:
            trial = getattr(self, "_current_trial", 0)
        if t is None:
            t = getattr(self, "_current_time", 0)
        if obs is None and hasattr(self, "_pending_observation"):
            obs = self._pending_observation
        if self.deep_inference and obs is not None and t not in self.observations:
            self.observations[t] = np.asarray(obs).copy()
        
        if self.deep_inference:
            inference = getattr(self, "inference", None)
            convergence_tolerance = getattr(
                inference,
                "convergence_tolerance",
                np.exp(-8),
            )
            self.last_state_inference = infer_deep_temporal_states(
                self,
                t,
                convergence_tolerance=convergence_tolerance,
            )

        else:
            if dF_tol is None:
                inference = getattr(self, "inference", None)
                dF_tol = getattr(
                    inference,
                    "convergence_tolerance",
                    1e-4,
                )
            self.last_state_inference = infer_shallow_states(
                self,
                obs,
                t,
                fixed_factor_index=res_idx,
                convergence_tolerance=dF_tol,
            )



    
    def infer_states_custom(self, trial, t): #implimentation of the MMP #Only for blind_obident_agent_example

        #@NOTE: Policy_pruning functionality needs to be debugged.
        #if self.policy_pruning:

        for policy_idx, policy in enumerate(self.policies):
            depolarization = None
            F = None
            for nmp in range(self.number_of_msg_passing):  # Number of gradient descent iterations
                previous_F = F
                self.F_policy[policy_idx] = previous_F
                F = 0
                for factor in range(self.num_factors):
                    for tau in range(self.temporal_horizon):
                        third_msg = self.create_object_tensor('zeros', 1, last_dim=self.states_dim[factor])
                        depolarization = self.log_stable(self.policy_dep_posteriors[policy_idx, tau, factor])
                        if tau <= t:
                            # Third message
                            if factor != 5:
                                third_msg = self.expected_log_likelihood_einsum(
                                    self.observations[tau],
                                    factor,
                                    policy_idx,
                                    tau,
                                )
                            
                        if tau == 0:
                            # First message
                            first_msg = self.log_stable(self.D[factor])
                            # Second message
                            if factor != 5:
                                action_tau = policy[tau, :]
                                qs_future = self.policy_dep_posteriors[policy_idx, tau+1, factor]
                                transposed_B = self.transpose_Bfa(self.B[factor][:, :, action_tau[factor]])
                                second_msg = self.log_stable(transposed_B.dot(qs_future))
                            else:
                                #obs_mod = int(self.observations[trial, tau, 4])
                                #qs_future = self.one_hot_encode(4, int(obs_mod), self.obs_dim)
                                qs_future = self.policy_dep_posteriors[policy_idx, tau+1, factor+1]
                                transposed_B = self.transpose_Bfa(self.B[factor][:, :, 0])
                                second_msg = self.log_stable(transposed_B.dot(qs_future)) 

                        elif tau == self.temporal_horizon-1:
                            if factor != 5:
                                # First message
                                actions_tau_1 = policy[tau-1, :]
                                qs_prev = self.policy_dep_posteriors[policy_idx, tau-1, factor]
                                first_msg = self.log_stable(self.B[factor][:, :, actions_tau_1[factor]].dot(qs_prev))
                                # Second message
                                second_msg = np.zeros((self.D[factor]).shape)
                            else:
                                #if not np.isnan(self.observations[trial, tau-1, 4]):
                                    #obs_mod = int(self.observations[trial, tau-1, 4])
                                    #qs_prev = self.one_hot_encode(4, int(obs_mod), self.obs_dim)
                                #else:
                                    #qs_prev = self.policy_dep_posteriors[policy_idx, tau-1, factor]
                                #first_msg = self.log_stable(self.B[factor][:, :, action_tau[factor]].dot(qs_prev))
                                qs_prev = self.policy_dep_posteriors[policy_idx, tau-1, factor+1]
                                first_msg = self.log_stable(self.B[factor][:, :, 0].dot(qs_prev))
                                # Second message
                                second_msg = np.zeros((self.D[factor]).shape)
                        else:
                            if factor != 5:
                                # First message
                                actions_tau_1 = policy[tau-1, :]
                                qs_prev = self.policy_dep_posteriors[policy_idx, tau-1, factor]
                                first_msg = self.log_stable(self.B[factor][:, :, actions_tau_1[factor]].dot(qs_prev))
                                # Second message
                                action_tau = policy[tau, :]
                                qs_future = self.policy_dep_posteriors[policy_idx, tau+1, factor]
                                transposed_B = self.transpose_Bfa(self.B[factor][:, :, action_tau[factor]])
                                second_msg = self.log_stable(transposed_B.dot(qs_future))
                            else:
                                #if not np.isnan(self.observations[trial, tau-1, 4]):
                                    #obs_mod = int(self.observations[trial, tau-1, 4])
                                    #qs_prev = self.one_hot_encode(4, int(obs_mod), self.obs_dim)
                                #else:
                                    #qs_prev = self.policy_dep_posteriors[policy_idx, tau-1, factor]
                                #first_msg = self.log_stable(self.B[factor][:, :, action_tau[factor]].dot(qs_prev))
                                qs_prev = self.policy_dep_posteriors[policy_idx, tau-1, factor+1]
                                first_msg = self.log_stable(self.B[factor][:, :, 0].dot(qs_prev))

                                # Second message
                                #if not np.isnan(self.observations[trial, tau, 4]):
                                    #obs_mod = int(self.observations[trial, tau, 4])
                                    #qs_future = self.one_hot_encode(4, int(obs_mod), self.obs_dim)
                                #else:
                                    #qs_future = self.policy_dep_posteriors[policy_idx, tau+1, factor]
                                qs_future = self.policy_dep_posteriors[policy_idx, tau+1, factor+1]
                                transposed_B = self.transpose_Bfa(self.B[factor][:, :, 0])
                                second_msg = self.log_stable(transposed_B.dot(qs_future))

                        # Compute state prediction error
                        state_pred_err = 0.5*(first_msg + second_msg) + third_msg - depolarization
                        depolarization += state_pred_err/self.timeconst
                        #@NOTE equation of F in tbl 2 on page 19 of the paper and MATLAB line of code for this is different.
                        # Following is the implimentation from the MATLAB.
                        Fintermediate = (self.policy_dep_posteriors[policy_idx, tau, factor]).dot(-self.log_stable(self.policy_dep_posteriors[policy_idx, tau, factor]) + 0.5*(first_msg + second_msg) +third_msg)
                        F += Fintermediate
                        self.policy_dep_posteriors[policy_idx, tau, factor] = self.softmax(np.array(depolarization))     
                #Early stopping condition to exit gradient descent if minimum VFE reached!
                if nmp > 0 and previous_F is not None:
                    if F - previous_F < np.exp(-8):
                        self.F_policy[policy_idx] = previous_F
                        break
        #self._setup_shared_memory()
                          
    def _eval_policy(self, t, policy_idx):
        risk_term = self.calculate_policy_risk(t, policy_idx)
        ambiguity_term = self.calculate_policy_ambiguity(t, policy_idx)
        info_gain_tot = 0
        if self.learning_D:
            info_gain_tot += self.calculate_pD_info_gain(policy_idx)
        if self.learning_A:
            info_gain_tot += self.calculate_pA_info_gain(t, policy_idx)
        if self.learning_B:
            info_gain_tot += self.calculate_pB_info_gain_vectorized(t, policy_idx)
        #if self.learning_E:
        #    info_gain_tot += self.calculate_pE_info_gain(policy_idx)
        return policy_idx, risk_term + ambiguity_term - info_gain_tot

    
    def get_expected_states(self, policy):
        # this function use during shallow inference
        qs_fur = np.empty(self.num_factors, dtype=object)
        for cntrl_factor, action in enumerate(policy):
            qs_fur[cntrl_factor] = self.B[cntrl_factor][...,int(action)] @ self.posteriors[cntrl_factor]
        return qs_fur
    
    def get_expected_obs(self, qs_fur):
        # this function use during shallow inference
        obs_fur = []
        for mod_idx, modality in enumerate(self.A):
            obs_fur.append(self.cell_md_dot_py(modality, qs_fur))
        return obs_fur
    
    def _compute_policy_terms(self, t, policy_idx):

        info_gain_tot = 0.0

        if self.continous_obs:

            ambiguity_term, predictions, H_Qo_tot = (
                self.calculate_policy_ambiguity_continuous_mc_vec(
                    t,
                    policy_idx
                )
            )

            risk_term = (
                self.calculate_policy_risk_continuous_mc_vec(
                    t,
                    policy_idx
                )
            )

        else:

            ambiguity_term = (
                self.calculate_policy_ambiguity(
                    t,
                    policy_idx
                )
            )

            risk_term = (
                self.calculate_policy_risk(
                    t,
                    policy_idx
                )
            )

            if self.learning_D:

                info_gain_tot += (
                    self.calculate_pD_info_gain(
                        policy_idx
                    )
                )

            if self.learning_A:

                info_gain_tot += (
                    self.calculate_pA_info_gain(
                        t,
                        policy_idx
                    )
                )

            if self.learning_B:

                info_gain_tot += (
                    self.calculate_pB_info_gain_vectorized(
                        t,
                        policy_idx
                    )
                )

        G = (info_gain_tot - risk_term + ambiguity_term*0.5)
            


        return (
            policy_idx,
            risk_term,
            ambiguity_term,
            predictions,
            H_Qo_tot,
            info_gain_tot,
            G
        )
    
    def infer_policies_parallel(self, trial, t, gamma_const=16.0):

        results = Parallel(
        n_jobs=-1,
        backend="loky"
    )(
        delayed(self._compute_policy_terms)(
            t,
            policy_idx
        )
        for policy_idx in range(len(self.policies))
    )

        self.risk = np.zeros(len(self.policies))
        self.ambiguity = np.zeros(len(self.policies))
        self.info_gain = np.zeros(len(self.policies))
        self.H_Qo = np.zeros(len(self.policies))

        for (
            policy_idx,
            risk_term,
            ambiguity_term,
            predictions,
            H_Qo_tot,
            info_gain_tot,
            G
        ) in results:

            self.risk[policy_idx] = risk_term
            self.ambiguity[policy_idx] = ambiguity_term
            self.policy_dep_expected_obs[policy_idx, :, :] = predictions
            self.info_gain[policy_idx] = info_gain_tot
            self.H_Qo[policy_idx] = H_Qo_tot
            self.G_policy[policy_idx] = G

        self.update_policy_posterior(trial, t)
    
    def infer_policies(self, trial=None, t=None, gamma_const=240.0):
        if trial is None:
            trial = getattr(self, "_current_trial", 0)
        if t is None:
            t = getattr(self, "_current_time", 0)

        if self.deep_inference:
            #if not t%self.temporal_horizon == self.temporal_horizon-1:
            self.risk = []
            self.ambiguity = []
            self.info_gain = []
            for policy_idx in range(len(self.policies)):
                info_gain_tot = 0

                #epistemic value term (Bayesian surprise)
                if self.continous_obs:
                    ambiguity_term, bn = self.calculate_policy_ambiguity_continuous_mc_vec(t, policy_idx)
                    #ambiguity_termx = self.calculate_policy_ambiguity_continuous_mc(t, policy_idx)
                    self.ambiguity.append(ambiguity_term)
                    risk_term = self.calculate_policy_risk_continuous_mc_vec(t, policy_idx)
                    #risk_termx = self.calculate_policy_risk_continuous_mc(t, policy_idx)
                    self.risk.append(risk_term)
                    #print(f"Policy {policy_idx}: Ambiguity (MC Vec) = {ambiguity_term}, Ambiguity (MC) = {ambiguity_termx}, Risk (MC Vec) = {risk_term}, Risk (MC) = {risk_termx}")
                    if self.learning_D:
                        info_gain_tot += self.calculate_pD_info_gain(policy_idx)
                        self.info_gain.append(self.calculate_pD_info_gain(policy_idx))
                else:
                    ambiguity_term = self.calculate_policy_ambiguity(t, policy_idx)
                    risk_term = self.calculate_policy_risk(t, policy_idx)
                    if self.learning_D:
                        info_gain_tot += self.calculate_pD_info_gain(policy_idx) 
                    if self.learning_A:
                        info_gain_tot += self.calculate_pA_info_gain(t, policy_idx)
                    if self.learning_B:
                        info_gain_tot += self.calculate_pB_info_gain_vectorized(t, policy_idx)

                #if self.learning_E:
                #    info_gain_tot += self.calculate_pE_info_gain(policy_idx)
                self.G_policy[policy_idx] = -risk_term -ambiguity_term +info_gain_tot

            
            self.update_policy_posterior(trial, t)
        else:
            self.last_policy_inference = infer_shallow_policies(
                self,
                t,
                policy_precision=gamma_const,
            )

        return (
            copy.deepcopy(self.G_policy),
            copy.deepcopy(getattr(self, "F_policy", None)),
        )


    def _calculate_cost(self, policy_idx, base_change_cost=0.15):
        """
        Calculates the cost of choosing a specific policy given the current posterior belief.
        
        policy_idx: int (0 to 4)
        posterior_belief: 1D numpy array of probabilities summing to 1
        """
        # Policy 4 means "No Action" -> Cost is 0
        if policy_idx == 4:
            return 0.0
        
        # If we choose a specific model, find its index
        
        # The probability that the model we are switching TO is already active
        prob_already_active = self.posteriors[0][policy_idx]
        
        # The probability that we are actually CHANGING the model
        prob_of_change = 1.0 - prob_already_active
        
        # Total cost is proportional to the probability that a change actually occurs
        cost = base_change_cost * prob_of_change
        
        return np.log(cost + 0.00001)

    '''
    def calculate_pE_info_gain(self, policy_idx):
        # @NOTE not sure if this is the way it should be done.
        # This part seems to be not implimented in both PYMDP and MATLAB versions.

        wE_term_policy = 0
        wE = self.wnorm_new(self.pE[policy_idx])
        expected_habits = self.E[policy_idx].dot(self.policy_dep_posteriors[policy_idx, :, :])
        expected_habits_pE = wE.dot(self.policy_dep_posteriors[policy_idx, :, :])
        wE_term_policy += -(expected_habits.dot(expected_habits_pE))
        return wE_term_policy
    '''
    
    def calculate_pD_info_gain(self, policy_idx):
        wD_term_policy = 0
        if self.deep_inference:
            # @NOTE according to the MATLAB code, pD info gain do not
            # depend on the time step of the policy. Therefore, we can
            # calculate it only when t==0, for the very first time step.
            for factor_idx in [2]:
                wD_factor = self.pD_complexity[factor_idx]
                #expected_sts = self.D[factor_idx].dot(self.policy_dep_posteriors[policy_idx, 0, factor_idx])
                expected_sts_pD = wD_factor.dot(self.policy_dep_posteriors[policy_idx, 0, factor_idx])
                wD_term_policy += expected_sts_pD
                #wD_term_policy += expected_sts*expected_sts_pD
            return wD_term_policy
        else:
            #@NOTE according to PYMDP in shallow inference, pD info gain is not included in EFE calculation.
            for factor_idx in range(self.num_factors):
                wD_factor = self.pD_complexity[factor_idx]
                expected_sts_pD = wD_factor.dot(self.posteriors[factor_idx])
                wD_term_policy += expected_sts_pD
            return wD_term_policy

    def calculate_pB_info_gain(self, t, policy_idx, qs_fur=None):
        wB_term_policy = 0
        policy = self.policies[policy_idx]
        if self.deep_inference:
            for timestep in range(t%self.temporal_horizon, self.temporal_horizon):
                action_t = policy[timestep]
                for factor_idx in range(self.num_factors):
                    wB_factor = self.pB_complexity[factor_idx][:, :, action_t[factor_idx]]
                    expected_states_t = self.policy_dep_posteriors[policy_idx, timestep, factor_idx]
                    expected_states_t1 = self.policy_dep_posteriors[policy_idx, timestep + 1, factor_idx]
                    wB_term_policy += expected_states_t.T @ wB_factor @ expected_states_t1
                    #expected_sts_pB_2 = wB_factor.dot(self.policy_dep_posteriors[policy_idx, timestep+1, factor_idx])
                    #expected_sts_pB_1 = wB_factor.dot(self.policy_dep_posteriors[policy_idx, timestep, factor_idx])
                    #wB_term_policy += expected_sts_pB_1.dot(expected_sts_pB_2)
            return wB_term_policy

        else:
            action_t = policy[0]
            for factor_idx in range(self.num_factors):
                wB_factor = self.pB_complexity[factor_idx][:, :, action_t[factor_idx]]
                wB_term_policy += self.posteriors[factor_idx].T @ wB_factor @ qs_fur[factor_idx]
            return wB_term_policy 

    def calculate_pB_info_gain_vectorized(self, t, policy_idx):
        T = self.temporal_horizon - 1 - t%self.temporal_horizon
        if T <= 0:
            return 0.0  # no timesteps to process
        F = self.num_factors

        policy_actions = self.policies[policy_idx][t%self.temporal_horizon:t%self.temporal_horizon+T]  # shape [T, F]
        wB_term_policy = 0.0

        for f in range(F):
            states_f = self.pB_complexity[f].shape[0]

            # Extract all transition matrices corresponding to actions at each timestep
            actions_f = policy_actions[:, f]  # shape [T]
            # Advanced indexing to get [T, states_f, states_f]
            wB_matrices = self.pB_complexity[f][:, :, actions_f]  # shape [states_f, states_f, T]
            wB_matrices = np.transpose(wB_matrices, (2, 0, 1))  # -> [T, states_f, states_f]

            # Extract state posteriors at t and t+1 for factor f
            # policy_dep_posteriors indexed as: [policy_idx, timestep, factor_idx, states_f]
            expected_states_t = np.array([
                self.policy_dep_posteriors[policy_idx, timestep, f]
                for timestep in range(t, t+T)
            ])  # shape [T, states_f]

            expected_states_t1 = np.array([
                self.policy_dep_posteriors[policy_idx, timestep, f]
                for timestep in range(t+1, t+T+1)
            ])  # shape [T, states_f]

            # Batch bilinear form: (x_t.T @ W @ x_t+1) for all T
            inter = np.einsum('ti,tij->tj', expected_states_t, wB_matrices)  # [T, states_f]
            terms = np.einsum('tj,tj->t', inter, expected_states_t1)  # [T]

            wB_term_policy += np.sum(terms)
        return wB_term_policy
   
    
    def calculate_pA_info_gain(self, t, policy_idx, qs_fur=None):
        wA_term_policy = 0
        if self.deep_inference:
            for timestep in range(t%self.temporal_horizon, self.temporal_horizon):
                for modality_idx, modality in enumerate(self.A):
                    wA_mod = self.pA_complexity[modality_idx]
                    expected_obs = self.cell_md_dot_py(modality, self.policy_dep_posteriors[policy_idx, timestep, :]) 
                    expected_obs_pA = self.cell_md_dot_py(wA_mod, self.policy_dep_posteriors[policy_idx, timestep, :])
                    wA_term_policy += expected_obs.dot(expected_obs_pA)
            return wA_term_policy
        
        else:
            predicted_gamma = np.outer(qs_fur[0], qs_fur[1])
            mu_old = self.external_lm.mu_err
            kappa_old = self.external_lm.kappa_err
            alpha_old = self.external_lm.alpha_err
            beta_old = self.external_lm.beta_err

            # === Hypothetical conjugate update using expected sufficient statistics ===
            # For pure epistemic value we use the *current predictive mean* as expected obs
            # (this makes the mean-shift term expected zero while still updating precision)
            obs_expected = mu_old   # key for production-grade approximation

            kappa_new = kappa_old + predicted_gamma
            alpha_new = alpha_old + 0.5 * predicted_gamma
            mu_new = (kappa_old * mu_old + predicted_gamma * obs_expected) / kappa_new
            

            # Your exact beta update
            beta_new = beta_old + (
                0.5 * (kappa_old * predicted_gamma / kappa_new) * (obs_expected - mu_old) ** 2
            )

            # === Compute KL per element (vectorized) ===
            # Broadcast everything
            IG_var = self.kl_normal_gamma(
                mu_new, kappa_new, alpha_new, beta_new,
                mu_old, kappa_old, alpha_old, beta_old
            )

            # Expected information gain: weight by predicted responsibility gamma
            # (this is the Monte-Carlo / expectation approximation over predictive o)
            IG_mu = 0.5 * np.log((kappa_new + EPS_VAL) / (kappa_old + EPS_VAL))


            IG_policy = np.sum(predicted_gamma * (IG_mu + IG_var))
            return IG_policy
        
    def kl_normal_gamma(self, mu_new, kappa_new, alpha_new, beta_new,
                        mu_old, kappa_old, alpha_old, beta_old):
        """
        KL[ NG_new (posterior) || NG_old (prior) ] 
        Univariate case matching your Normal-Gamma / Student's-t setup.
        """
        eps = 1e-12
        alpha_new = np.maximum(alpha_new, eps)
        alpha_old = np.maximum(alpha_old, eps)
        beta_new = np.maximum(beta_new, eps)
        beta_old = np.maximum(beta_old, eps)
        kappa_new = np.maximum(kappa_new, eps)
        kappa_old = np.maximum(kappa_old, eps)

        # === Normal component (mean) ===
        # Weighted squared difference + precision ratio terms
        term_mu = 0.5 * (alpha_new / beta_new) * kappa_old * (mu_new - mu_old)**2
        term_kappa = 0.5 * (np.log(kappa_old / kappa_new) - (kappa_old / kappa_new) + 1.0)

        # === Gamma component (precision / variance) ===
        term_gamma = (
            alpha_old * np.log(beta_new / beta_old)
            - (gammaln(alpha_new) - gammaln(alpha_old))
            + (alpha_new - alpha_old) * digamma(alpha_new)
            - (beta_new - beta_old) * (alpha_new / beta_new)
        )

        kl = term_mu + term_kappa + term_gamma
        return np.clip(kl, 0.0, None)  # Ensure non-negativity for numerical safety
    
    def calculate_policy_ambiguity_continuous_mc_vec(
        self,
        t,
        policy_idx,
        qs_next=None,
        num_samples=500
    ):

        if self.deep_inference:

            ambiguity = 0.0
            H_Qo_tot = 0.0
            predictions = np.zeros([2,self.num_modalities] , dtype=object)

            for timestep in range(t%self.temporal_horizon,self.temporal_horizon):

                qs_t = [
                    self.policy_dep_posteriors[
                        policy_idx,
                        timestep,
                        f
                    ]
                    for f in range(self.num_factors)
                ]

                for m, dep_factors in enumerate(self.mod_dep):

                    o_grid = self.external_lm.get_o_grid(m)

                    # -----------------------------------
                    # 1. vectorized latent sampling
                    # -----------------------------------

                    samples = [
                        np.random.choice(
                            len(qs_t[f]),
                            size=num_samples,
                            p=qs_t[f]
                        )
                        for f in dep_factors
                    ]

                    # -----------------------------------
                    # 2. vectorized likelihood evaluation
                    # -----------------------------------

                    if len(samples) == 1:
                        latent_samples = samples[0]
                    else:
                        latent_samples = tuple(samples)

                    P_samples = (
                        self.external_lm.likelihoods_grid_vec(
                            o_grid,
                            m,
                            latent_samples
                        )
                    )

                    # shape:
                    # (num_samples, len(o_grid))

                    # -----------------------------------
                    # 3. predictive distribution Q(o)
                    # -----------------------------------

                    Q_o = P_samples.mean(axis=0)

                    Q_o /= (
                        Q_o.sum() + EPS_VAL
                    )

                    # -----------------------------------
                    # store expected observations
                    # -----------------------------------

                    if t%self.temporal_horizon == 0:

                        if timestep in (0, 1):

                            predictions[timestep][m] = Q_o

                    # -----------------------------------
                    # 4. entropy of predictive distribution
                    # -----------------------------------

                    log_Qo = np.log(
                        Q_o + EPS_VAL
                    )

                    H_Qo = -np.sum(
                        Q_o * log_Qo
                    )

                    # -----------------------------------
                    # 5. expected conditional entropy
                    # -----------------------------------

                    log_P = np.log(
                        P_samples + EPS_VAL
                    )

                    H_cond_samples = -np.sum(
                        P_samples * log_P,
                        axis=1
                    )

                    H_cond = np.mean(
                        H_cond_samples
                    )

                    # -----------------------------------
                    # 6. ambiguity contribution
                    # -----------------------------------

                    ambiguity += H_Qo - H_cond
                    H_Qo_tot += H_cond
            #print(f"Policy {policy_idx} vectorized ambiguity calculation took {end_t - start_t:.2f} seconds and ambiguity value is {ambiguity:.4f}.")
            return float(ambiguity), predictions, float(H_Qo_tot)
        
        else:
            ambiguity = 0.0

            for m, dep_factors in enumerate(self.mod_dep):

                o_grid = self.external_lm.get_o_grid(m)

                # -----------------------------------
                # 1. vectorized latent sampling
                # -----------------------------------

                samples = [
                    np.random.choice(
                        len(qs_next[f]),
                        size=num_samples,
                        p=qs_next[f]
                    )
                    for f in dep_factors
                ]

                # -----------------------------------
                # 2. vectorized likelihood evaluation
                # -----------------------------------

                if len(samples) == 1:
                    latent_samples = samples[0]
                else:
                    latent_samples = tuple(samples)

                P_samples = (
                    self.external_lm.likelihoods_grid_vec(
                        o_grid,
                        m,
                        latent_samples
                    )
                )

                # shape:
                # (num_samples, len(o_grid))

                # -----------------------------------
                # 3. predictive distribution Q(o)
                # -----------------------------------

                Q_o = P_samples.mean(axis=0)

                Q_o /= (
                    Q_o.sum() + EPS_VAL
                )

                # -----------------------------------
                # 4. entropy of predictive distribution
                # -----------------------------------

                log_Qo = np.log(
                    Q_o + EPS_VAL
                )

                H_Qo = -np.sum(
                    Q_o * log_Qo
                )

                # -----------------------------------
                # 5. expected conditional entropy
                # -----------------------------------

                log_P = np.log(
                    P_samples + EPS_VAL
                )

                H_cond_samples = -np.sum(
                    P_samples * log_P,
                    axis=1
                )

                H_cond = np.mean(
                    H_cond_samples
                )

                # -----------------------------------
                # 6. ambiguity contribution
                # -----------------------------------

                ambiguity += H_Qo-H_cond
        #print(f"Policy {policy_idx} vectorized ambiguity calculation took {end_t - start_t:.2f} seconds and ambiguity value is {ambiguity:.4f}.")
        return float(ambiguity*0.6)

    def calculate_policy_ambiguity_continuous_mc(self, t, policy_idx, num_samples=500):
        if not self.deep_inference:
            return 0.0

        ambiguity = 0.0
        self.amb_t = 0.0

        for timestep in range(t % self.temporal_horizon, self.temporal_horizon):

            qs_t = [
                self.policy_dep_posteriors[policy_idx, timestep, f]
                for f in range(self.num_factors)
            ]

            for m, dep_factors in enumerate(self.mod_dep):

                o_grid = self.external_lm.get_o_grid(m)

                # -----------------------------
                # 1. sample from Q(s)
                # -----------------------------
                samples = []
                for f in dep_factors:
                    s_f = np.random.choice(
                        len(qs_t[f]),
                        size=num_samples,
                        p=qs_t[f]
                    )
                    samples.append(s_f)

                # -----------------------------
                # 2. compute p(o | s)
                # -----------------------------
                P_samples = np.zeros((num_samples, len(o_grid)))

                for i, s_vals in enumerate(zip(*samples)):

                    p_o_given_s = self.external_lm.likelihoods_grid(
                        o_grid, m, s_vals
                    )  # already probability, sums to 1

                    P_samples[i, :] = p_o_given_s

                # -----------------------------
                # 3. predictive distribution Q(o)
                # -----------------------------
                Q_o = P_samples.mean(axis=0)
                Q_o = Q_o / (Q_o.sum() + EPS_VAL)

                if t % self.temporal_horizon == 0:
                    if timestep == 0:
                        self.policy_dep_expected_obs[policy_idx, timestep, m] = Q_o

                    if timestep == 1:
                        self.policy_dep_expected_obs[policy_idx, timestep, m] = Q_o
                # entropy of predictive distribution
                H_Qo = -np.sum(Q_o * np.log(Q_o + EPS_VAL))

                # -----------------------------
                # 4. expected conditional entropy
                # -----------------------------
                H_cond_samples = -np.sum(
                    P_samples * np.log(P_samples + EPS_VAL),
                    axis=1
                )

                H_cond = np.mean(H_cond_samples)

                # -----------------------------
                # 5. ambiguity contribution
                # -----------------------------
                ambiguity += (H_Qo - H_cond)

        return ambiguity

    def calculate_policy_ambiguity(self, t, policy_idx, qs_t=None):
        """
        Calculates policy ambiguity using factorized posteriors to avoid iterating
        over the joint state space.
        """
        ambiguity = 0.0
        if self.deep_inference:

            for timestep in range(t%self.temporal_horizon, self.temporal_horizon):
                # Get the factorized posteriors for this timestep
                # qs_t is a list of vectors, one for each state factor
                qs_t = [self.policy_dep_posteriors[policy_idx, timestep, f] for f in range(self.num_factors)]

                # Term 1: Entropy of expected outcomes: H[Q(o)]
                H_Qo = 0.0
                for m, A_m in enumerate(self.A):
                    test_A_m = 0
                    # Q(o_m) = sum_s Q(s) P(o_m|s)
                    # We compute this via sequential tensor contraction
                    q_o_m = A_m
                    for f in range(self.num_factors):
                        # Contract with the posterior for factor f
                        q_o_m = np.tensordot(q_o_m, qs_t[f], axes=(1, 0))
                    
                    # Add entropy of this modality to the total
                    test_A_m = -q_o_m.dot(self.log_stable(q_o_m))
                    H_Qo += -q_o_m.dot(self.log_stable(q_o_m))

                # Term 2: Expected entropy of outcomes: E_Q(s)[H(P(o|s))]
                E_qs_H_A = 0.0
                for m, A_m in enumerate(self.A):
                    # H_A_m = H[P(o_m|s)] for all s
                    # This results in a tensor with dimensions of the state space
                    H_A_m = -np.sum(A_m * self.log_stable(A_m), axis=0)

                    # E_Q(s)[H_A_m] = sum_s Q(s) H_A_m(s)
                    # We compute this via sequential tensor contraction
                    expected_H_A_m = H_A_m
                    for f in range(self.num_factors):
                        expected_H_A_m = np.tensordot(expected_H_A_m, qs_t[f], axes=(0, 0))
                    
                    E_qs_H_A += expected_H_A_m

                # Ambiguity for this timestep
                ambiguity_tau = H_Qo - E_qs_H_A
                ambiguity += ambiguity_tau

            return ambiguity
        
        else:

            # Term 1: Entropy of expected outcomes: H[Q(o)]
            H_Qo = 0.0
            for m, A_m in enumerate(self.A):
                test_A_m = 0
                # Q(o_m) = sum_s Q(s) P(o_m|s)
                # We compute this via sequential tensor contraction
                q_o_m = A_m
                for f in range(self.num_factors):
                    # Contract with the posterior for factor f
                    q_o_m = np.tensordot(q_o_m, qs_t[f], axes=(1, 0))
                
                # Add entropy of this modality to the total
                test_A_m = -q_o_m.dot(self.log_stable(q_o_m))
                H_Qo += -q_o_m.dot(self.log_stable(q_o_m))

            # Term 2: Expected entropy of outcomes: E_Q(s)[H(P(o|s))]
            E_qs_H_A = 0.0
            for m, A_m in enumerate(self.A):
                # H_A_m = H[P(o_m|s)] for all s
                # This results in a tensor with dimensions of the state space
                H_A_m = -np.sum(A_m * self.log_stable(A_m), axis=0)

                # E_Q(s)[H_A_m] = sum_s Q(s) H_A_m(s)
                # We compute this via sequential tensor contraction
                expected_H_A_m = H_A_m
                for f in range(self.num_factors):
                    expected_H_A_m = np.tensordot(expected_H_A_m, qs_t[f], axes=(0, 0))
                
                E_qs_H_A += expected_H_A_m

            # Ambiguity for this timestep
            ambiguity_tau = H_Qo - E_qs_H_A
            ambiguity += ambiguity_tau

        return ambiguity

    """
    def calculate_policy_ambiguity_old(self, t, policy_idx): 
        # This functions follows the same implimentation used in the Pymdp
        # However here we use np.multiply.outter() for outter products.
        
        ambiguity = 0
        for timestep in range(t, self.temporal_horizon):
            ambiguity_tau = 0
            qo = 0
            qs = self.joint_policy_dep_posteriors[policy_idx, timestep]
            # get the indexs of where probabilities are larger than exp(-16)
            idx = np.argwhere(qs > np.exp(-16))
            for i in idx:
                po = np.ones(1) #used to store probabilities over outcome
                for modality in self.A:
                    index_vector = [slice(0, modality.shape[0])] + list(i)
                    po = np.multiply.outer(po, modality[tuple(index_vector)])
                po = po.ravel()
                qo += qs[tuple(i)] * po
                ambiguity_tau += qo.T.dot(self.log_stable(po, val=np.exp(-16)))
        
            # entropy of expectations: i.e., E_{Q(o)}[lnQ(o)]
            exp_qo_tau = qo.T.dot(self.log_stable(qo, val=np.exp(-16)))
            ambiguity_tau += -exp_qo_tau
            ambiguity += ambiguity_tau
        
        return ambiguity
    """
    def calculate_policy_risk_continuous_mc_vec(self, t, policy_idx, qs_next=None, num_samples=500):
        """
        Monte Carlo estimation of policy risk for continuous observations.

        Parameters
        ----------
        policy_idx : int
            Index of the policy to evaluate
        num_samples : int
            Number of Monte Carlo samples per timestep
        """
        risk_term_policy = 0.0
        risk_joint = 0
        risk_single = 0
        if self.deep_inference:
            o_grids = [self.external_lm.get_o_grid(m) for m in range(self.num_modalities)]
            for timestep in range(t%self.temporal_horizon, self.temporal_horizon):
                # factorized posterior for each factor
                qs_t = [self.policy_dep_posteriors[policy_idx, timestep, f] for f in range(self.num_factors)]
                
                """
                # --- handle joint modalities ---
                if self.pref_dep is not None:

                    for joint in self.pref_dep:

                        joint_dep_factors = sorted(set(
                            f for m in joint
                            for f in self.mod_dep[m]
                        ))

                        samples = [
                            np.random.choice(
                                len(qs_t[f]),
                                size=num_samples,
                                p=qs_t[f]
                            )
                            for f in joint_dep_factors
                        ]

                        # vectorized likelihood evaluation
                        Px = self.external_lm.likelihoods_grid_vec(
                            o_grids[joint[0]],
                            joint[0],
                            samples[0]
                        )

                        Py = self.external_lm.likelihoods_grid_vec(
                            o_grids[joint[1]],
                            joint[1],
                            samples[1]
                        )

                        # outer product for all samples
                        Q_o_joint = (
                            Py[:, :, None] *
                            Px[:, None, :]
                        )

                        # normalize jointly
                        Q_o_joint /= (
                            Q_o_joint.sum(axis=(1, 2), keepdims=True)
                            + EPS_VAL
                        )

                        C_joint = self.log_preferences_dict[tuple(joint)]

                        Q_o_joint_mean = Q_o_joint.mean(axis=0)
                        Q_o_joint_mean /= (Q_o_joint_mean.sum() + EPS_VAL)
                        H_Qo_joint = -np.sum(Q_o_joint_mean * np.log(Q_o_joint_mean + EPS_VAL))

                        # direct expectation
                        risk_joint += np.mean(np.sum(Q_o_joint * C_joint[None,:,:], axis=(1,2))) + H_Qo_joint*0
                """
                # --- handle single modalities ---
                for m in range(len(o_grids)):

                    if (
                        self.pref_dep is not None
                        and any(int(m) in joint for joint in self.pref_dep)
                    ):
                        continue

                    dep_factors = self.mod_dep[m]

                    samples = [
                        np.random.choice(
                            len(qs_t[f]),
                            size=num_samples,
                            p=qs_t[f]
                        )
                        for f in dep_factors
                    ]

                    if len(samples) == 1:
                        latent_samples = samples[0]
                    else:
                        latent_samples = tuple(samples)

                    P_samples = self.external_lm.likelihoods_grid_vec(
                        o_grids[m],
                        m,
                        latent_samples
                    )

                    Q_o = P_samples.mean(axis=0)

                    Q_o /= (
                        Q_o.sum() + EPS_VAL
                    )

                    C_o = self.log_preferences_dict[m]

                    H_Qo = -np.sum(Q_o * np.log(Q_o + EPS_VAL))

                    risk_single += -np.sum(Q_o * C_o) + H_Qo*0

            risk_term_policy = risk_joint + risk_single
            #print(f"Joint time: {joint_time:.4f}, Signal time: {signal_time:.4f}")
            return risk_term_policy
        
        else:
            o_grids = [self.external_lm.get_o_grid(m) for m in range(self.num_modalities)]

            # --- handle joint modalities ---
            if self.pref_dep is not None:

                for joint in self.pref_dep:
                    p_samples = {}
                    for m in joint:
                        
                        dep_factors = self.mod_dep[m]

                        samples = [
                            np.random.choice(
                                len(qs_next[f]),
                                size=num_samples,
                                p=qs_next[f]
                            )
                            for f in dep_factors
                        ]

                        if len(samples) == 1:
                            latent_samples = samples[0]
                        else:
                            latent_samples = tuple(samples)

                        P_m = self.external_lm.likelihoods_grid_vec(
                            o_grids[m],
                            m,
                            latent_samples
                        )

                        p_samples[m] = P_m

                    # outer product for all samples
                    Q_o_joint = (
                        p_samples[joint[0]][:, :, None] *  # Modality 0 (Info Gain)
                        p_samples[joint[1]][:, None, :]    # Modality 1 (Accuracy)
                    )

                    # normalize jointly
                    Q_o_joint /= (
                        Q_o_joint.sum(axis=(1, 2), keepdims=True)
                        + EPS_VAL
                    )

                    C_joint = self.log_preferences_dict[tuple(joint)]

                    Q_o_joint_mean = Q_o_joint.mean(axis=0)
                    Q_o_joint_mean /= (Q_o_joint_mean.sum() + EPS_VAL)
                    H_Qo_joint = -np.sum(Q_o_joint_mean * np.log(Q_o_joint_mean + EPS_VAL))

                    # direct expectation
                    risk_joint += -np.mean(np.sum(Q_o_joint * C_joint[None,:,:], axis=(1,2))) + H_Qo_joint*0
            
            # --- handle single modalities ---
            for m in range(len(o_grids)):

                if (
                    self.pref_dep is not None
                    and any(int(m) in joint for joint in self.pref_dep)
                ):
                    continue

                dep_factors = self.mod_dep[m]

                samples = [
                    np.random.choice(
                        len(qs_next[f]),
                        size=num_samples,
                        p=qs_next[f]
                    )
                    for f in dep_factors
                ]

                if len(samples) == 1:
                    latent_samples = samples[0]
                else:
                    latent_samples = tuple(samples)

                P_samples = self.external_lm.likelihoods_grid_vec(
                    o_grids[m],
                    m,
                    latent_samples
                )

                Q_o = P_samples.mean(axis=0)

                Q_o /= (
                    Q_o.sum() + EPS_VAL
                )

                C_o = self.log_preferences_dict[m]

                H_Qo = -np.sum(Q_o * np.log(Q_o + EPS_VAL))

                risk_single += -np.sum(Q_o * C_o) + H_Qo*0

        risk_term_policy = risk_joint + risk_single
        return risk_term_policy

    def calculate_policy_risk_continuous_mc(self, t, policy_idx, num_samples=500):
        """
        Monte Carlo estimation of policy risk for continuous observations.

        Parameters
        ----------
        policy_idx : int
            Index of the policy to evaluate
        num_samples : int
            Number of Monte Carlo samples per timestep
        """
        risk_term_policy = 0.0
        if self.deep_inference:
            o_grids = [self.external_lm.get_o_grid(m) for m in range(self.num_modalities)]
            risk_joint = 0
            risk_signal = 0
            for timestep in range(t%self.temporal_horizon, self.temporal_horizon):
                # factorized posterior for each factor
                qs_t = [self.policy_dep_posteriors[policy_idx, timestep, f] for f in range(self.num_factors)]
                
                # --- handle joint modalities ---
                if self.pref_dep is not None:
                    for joint in self.pref_dep:
                        joint_dep_factors = sorted(set(
                            f for m in joint for f in self.mod_dep[m]
                        ))

                        samples = [
                            np.random.choice(len(qs_t[f]), size=num_samples, p=qs_t[f])
                            for f in joint_dep_factors
                        ]

                        P_samples = np.zeros((num_samples,
                                            *(len(self.external_lm.get_o_grid(m)) for m in joint)))

                        for i, s_vals in enumerate(zip(*samples)):

                            log_Lx = self.external_lm.likelihoods_grid(o_grids[0], joint[0], [s_vals[0]])
                            log_Ly = self.external_lm.likelihoods_grid(o_grids[1], joint[1], [s_vals[1]])

                            logP = np.log(log_Ly + 1e-12)[:, None] + np.log(log_Lx + 1e-12)[None, :]

                            P = np.exp(logP)
                            P = P / (P.sum() + 1e-12)

                            P_samples[i] = P

                        Q_joint = P_samples.mean(axis=0)
                        Q_joint = Q_joint / (Q_joint.sum() + 1e-12)

                        C_joint = self.log_preferences_dict[tuple(joint)]

                        risk_joint += np.sum(Q_joint.T * C_joint)
                
                # --- handle single modalities ---
                for m in range(len(o_grids)):
                    # skip modalities already included in joint
                    if self.pref_dep is not None and any(int(m) in joint for joint in self.pref_dep):
                        continue

                    dep_factors = self.mod_dep[m]
                    samples = [np.random.choice(len(qs_t[f]), size=num_samples, p=qs_t[f]) for f in dep_factors]

                    P_samples = np.zeros((num_samples, len(o_grids[m])))
                    for i, s_vals in enumerate(zip(*samples)):
                        P_samples[i, :] = self.external_lm.likelihoods_grid(o_grids[m], m, s_vals)

                    Q_o = P_samples.mean(axis=0)
                    Q_o /= (Q_o.sum() + EPS_VAL)
                    #delta_o = o_grids[m][1] - o_grids[m][0]

                    C_o = self.log_preferences_dict[m]
                    risk_signal += np.sum(Q_o * C_o)

            return risk_joint + risk_signal
    
        
    def calculate_policy_risk(self, t, policy_idx, qs_fur=None):
        risk_term_policy = 0

        if self.deep_inference:
            #risk_term_policy_old = 0
            for timestep in range(t%self.temporal_horizon, self.temporal_horizon):
                #self.policy_dep_expected_obs = self.create_object_tensor(last_dim=self.obs_dim)
                for modality_idx, modality in enumerate(self.A):
                    # @NOTE both of the following lines finds the posteriors over observations
                    # One use tensordot with the joint_policy_dep_posteriors and the
                    # other uses matlab custom dot function spm_cell_md_dot 
                    #expected_obs = np.tensordot(modality, self.joint_policy_dep_posteriors[policy_idx, t], axes=(tuple(range(1, modality.ndim)), tuple(range(self.joint_policy_dep_posteriors[policy_idx, t].ndim))))
                    
                    # @NOTE: the following implimetation follows the equations in the paper
                    # but it is not the same as the one used in the MATLAB code.
                    #### MATLAB implimetation:
                    #expected_obs = self.cell_md_dot(modality, self.policy_dep_posteriors[policy_idx, t, :])
                    #risk_term_policy_old += expected_obs.dot(self.C[modality_idx][:, t])

                    #@NOTE cell_md_dot() and cell_md_dot_py() do the same. cell_md_dot_py() should give better performance.
                    #expected_obs_1 = self.cell_md_dot(modality, self.policy_dep_posteriors[policy_idx, timestep, :])
                    #expected_obs = self.cell_md_dot_py(modality, self.policy_dep_posteriors[policy_idx, timestep, :])
                    self.policy_dep_expected_obs[policy_idx, timestep][modality_idx] = self.cell_md_dot_py(modality, self.policy_dep_posteriors[policy_idx, timestep, :])
                    #KL_modality = self.log_stable(expected_obs) - self.C[modality_idx][:, t]
                    risk_term_policy += self.policy_dep_expected_obs[policy_idx, timestep][modality_idx].dot(self.C[modality_idx][:, timestep])
            return risk_term_policy
        
        else:

            for modality_idx, modality in enumerate(self.A):
                obs_next_mod = self.cell_md_dot_py(modality, qs_fur)
                risk_term_policy += obs_next_mod.dot(self.C[modality_idx])
            return float(risk_term_policy)
    
    def cell_md_dot_py(self, X, x):
        # use this for observation prediction only
        return factor_dot(X, x)
    
    def spm_dot(self, X, x):
        return numerical_spm_dot(X, x)


    
    def cell_md_dot(self, X, x):
        return np.squeeze(factor_dot(X, x))
            
    def choose_action(self, trial, t):
        action_list = None

        if self.deep_inference:

            if t%self.temporal_horizon < self.temporal_horizon-1:
                #self.alpha = 0.1 * np.exp(0.05 * trial)
                if self.action_selection == "deterministic":
                    policy_idx = np.argmax(self.posterior_pi)
                    for factor_idx in self.controlable_states:
                        self.action_posteriors[factor_idx] = self.policies[policy_idx][t%self.temporal_horizon, factor_idx]

                elif self.action_selection == "marginal":    
                    action_list = {}

                    for f in self.controlable_states:

                        n_actions = self.controls_dim[f]

                        # ------------------------------------------------------------
                        # 1. extract actions chosen by each policy at time t
                        # ------------------------------------------------------------
                        actions = np.array(self.policies)[:, t%self.temporal_horizon, f]   # shape: (NumPolicies,)

                        # ------------------------------------------------------------
                        # 2. accumulate posterior mass over actions
                        # ------------------------------------------------------------
                        action_mass = np.zeros(n_actions)

                        np.add.at(
                            action_mass,
                            actions,
                            self.posterior_pi
                        )

                        action_list[f] = action_mass

                    for factor_idx in self.controlable_states:
                        action_list[factor_idx] = self.softmax(self.log_stable(action_list[factor_idx]), axis=None, gamma = self.alpha)
                        self.action_posteriors[factor_idx] = np.searchsorted(np.cumsum(action_list[factor_idx]), np.random.rand())

                elif self.action_selection == "random":
                    action_prob = {}
                    # Initialize action_list properly
                    for idx, i in enumerate(self.controls_dim):  # Iterate with index
                        if i == 1:
                            continue  # Skip if control dimension is 1
                        else:
                            action_prob[idx] = np.zeros(i)
                    
                    for factor_idx in self.controlable_states:
                        self.action_posteriors[factor_idx] = random.choice(range(self.controls_dim[factor_idx]))
                        action_prob[factor_idx] = 1
                
                elif self.action_selection == "stochastic":
                    log_posterior_pi = self.log_stable(self.posterior_pi)
                    p_policies = self.softmax(log_posterior_pi * self.alpha) 
                    policy_idx = self.sample(p_policies)
                    for factor_idx in self.controlable_states:
                        self.action_posteriors[factor_idx] = self.policies[policy_idx][t%self.temporal_horizon, factor_idx]

                return self.action_posteriors, action_list
            else:
                return None, None
            

        else:
            # shallow inference does not have time-varying policy posteriors
            if self.action_selection == "deterministic":
                policy_idx = np.argmax(self.posterior_pi)
                for factor_idx in self.controlable_states:
                    self.action_posteriors[factor_idx] = self.policies[policy_idx][0, factor_idx]

            elif self.action_selection == "marginal":    
                action_list = {}

                # Initialize action_list properly
                for idx, i in enumerate(self.controls_dim):  # Iterate with index
                    if i == 1:
                        continue  # Skip if control dimension is 1
                    else:
                        action_list[idx] = np.zeros(i)  # Correctly initialize

                # Accumulate probabilities into action_list
                for policy_idx, policy in enumerate(self.policies):
                    policy_t_action = policy[0]
                    for factor_idx in self.controlable_states:
                            fac_action = policy_t_action[factor_idx]
                            action_list[factor_idx][fac_action] += self.posterior_pi[policy_idx]

                for factor_idx in self.controlable_states:
                    action_list[factor_idx] = self.softmax(self.log_stable(action_list[factor_idx]), axis=None, gamma = self.alpha)
                    self.action_posteriors[factor_idx] = np.searchsorted(np.cumsum(action_list[factor_idx]), np.random.rand())

            elif self.action_selection == "random":
                action_prob = {}
                # Initialize action_list properly
                for idx, i in enumerate(self.controls_dim):  # Iterate with index
                    if i == 1:
                        continue  # Skip if control dimension is 1
                    else:
                        action_prob[idx] = np.zeros(i)
                
                for factor_idx in self.controlable_states:
                    self.action_posteriors[factor_idx] = random.choice(range(self.controls_dim[factor_idx]))
                    action_prob[factor_idx] = 1
            
            elif self.action_selection == "stochastic":
                log_posterior_pi = self.log_stable(self.posterior_pi)
                p_policies = self.softmax(log_posterior_pi * self.alpha) 
                policy_idx = self.sample(p_policies)
                for factor_idx in self.controlable_states:
                    self.action_posteriors[factor_idx] = self.policies[policy_idx][0, factor_idx]

            self.action_posteriors_cache[:, t%self.learning_window] = copy.deepcopy(self.action_posteriors)

            return self.action_posteriors, action_list

    def sample(self, probabilities):
        """
        Sample an index from a categorical distribution.

        Args:
            probabilities (np.ndarray): 1D array of probabilities (must sum to 1)

        Returns:
            int: Index of the sampled category
        """
        probabilities = np.ravel(probabilities)
        return np.argmax(np.random.multinomial(1, probabilities))
    

    def _extract_vfe_features(self):
        """
        F_policy: list/array of length T (time steps), each element an array of length P (policies)
        Returns: feature vector describing the trial
        """
        F = np.stack(self.F_policy)  # shape (T, P)
        T, P = F.shape

        # Basic across-time summaries per policy
        mean_p = F.mean(axis=0)           # mean per policy (27,)
        var_p = F.var(axis=0)
        max_p = F.max(axis=0)
        
        # Temporal dynamics (averaged across policies)
        mean_t = F.mean(axis=1)           # mean per time step (4,)
        slope_t = np.polyfit(np.arange(T), mean_t, 1)[0]  # trend over time

        # Aggregate summaries
        global_mean = F.mean()
        global_max = F.max()
        global_var = F.var()
        global_mad = np.median(np.abs(F - np.median(F)))

        # Compact feature vector
        # Here we include aggregate and a few distributional stats across policies
        features = np.array([
            global_mean,
            global_max,
            global_var,
            global_mad,
            slope_t,
            np.mean(var_p),
            np.std(var_p),
            np.mean(max_p),
            np.std(max_p)
        ])

        return features

        
    def perform_learning(self, trial, actual_t = None):
        
        if self.deep_inference:
            #self.perform_modal_average()
            if self.learning_C:
                if self.continous_obs:
                    self.external_lm.update_C_vectorized(self.bayesian_mod_avg)

                else:
                    for t in range(self.temporal_horizon):
                        for modality_idx in range(len(self.pA)):
                            self.pC[modality_idx][:, t] += self.learning_rate*self.disparity_nu[t, modality_idx]*self.expected_obs_chosen[t, modality_idx]
            
            if self.learning_A:
                if self.continous_obs:
                    self.external_lm.update_mu_sigma_vectorized(self.observations, self.bayesian_mod_avg)
                else:
                    for t in range(self.temporal_horizon):
                        for modality_idx in range(len(self.pA)):
                            obs_mod = int(self.observations[t, modality_idx])
                            A_mm = self.one_hot_encode(modality_idx, int(obs_mod), self.obs_dim)
                            for factor_idx in range(self.num_factors):
                                A_mm = np.multiply.outer(A_mm, self.bayesian_mod_avg[trial, t,factor_idx])
                                            
                            #A_mm = A_mm * (A_mm == np.max(A_mm))
                            i = self.pA[modality_idx] > 0
                            self.pA[modality_idx] = np.where(
                                i,
                                self.forgeting_rate * (self.pA[modality_idx] - self.pA_0[modality_idx]) +
                                self.pA_0[modality_idx] +
                                self.learning_rate * A_mm,  
                                self.pA[modality_idx]
                            )                    
                            del A_mm
                            """
                            A_mm_modality = copy.deepcopy(A_mm[obs_mod])
                            max_vals = np.max(A_mm_modality, axis=0)
                            max_only = np.zeros_like(A_mm_modality)
                            mask = A_mm_modality == max_vals
                            max_only[mask] = A_mm_modality[mask]
                            i = max_only > 0
                            self.pA[modality_idx][obs_mod] = np.where(
                                i,
                                self.forgeting_rate * (self.pA[modality_idx][obs_mod] - self.pA_0[modality_idx][obs_mod]) +
                                self.pA_0[modality_idx][obs_mod] +
                                self.learning_rate * A_mm_modality,  
                                self.pA[modality_idx][obs_mod]
                            )                    
                            del A_mm
                            """
                            
                    # free energy of a
                    #for modality_idx in range(len(self.pA)):
                        #self.Fa[modality_idx] += self.KL_dirichlet(self.pA[modality_idx], self.pA_prior[modality_idx])
                    self.A = self._normalize_colums(self.pA)

            if self.learning_D:
                for factor_idx in range(self.num_factors):
                    if factor_idx in self.controlable_states:
                        fr = 0
                    else:
                        fr = self.forgeting_rate
                    i = self.bayesian_mod_avg[self.temporal_horizon -1, factor_idx] >= 0.01
                    self.pD[factor_idx] = np.where(
                        i,
                        fr * (self.pD[factor_idx] - self.pD_0[factor_idx]) 
                        + self.pD_0[factor_idx] 
                        + self.learning_rate * self.bayesian_mod_avg[self.temporal_horizon -1, factor_idx], #self.temporal_horizon -1
                        self.pD[factor_idx]
                    )                
                
                    # free energy of d
                    #self.Fd[factor_idx] = self.KL_dirichlet(self.pD[factor_idx], self.pD_prior[factor_idx])
                self.D = self._normalize_colums(self.pD)

            if self.learning_B:
                
                for t in range(self.temporal_horizon):
                    if t > 0:
                        
                        for factor_idx in range(self.num_factors):
                            action = int(self.action_posteriors[factor_idx, t-1])
                            if factor_idx not in self.controlable_states:
                                continue
                            
                            state_before = self.bayesian_mod_avg[trial, t-1, factor_idx]
                            state_after = self.bayesian_mod_avg[trial, t, factor_idx]
                            joint_states = np.outer(state_after, state_before)
                            #joint_states = joint_states*self.action_confidance[t-1, factor_idx][action]
                            joint_states *= (self.B[factor_idx][:, :, action] > 0).astype("float")
                            
                            # Get index of column containing the highest value
                            max_col_idx = np.unravel_index(np.argmax(joint_states), joint_states.shape)[1]

                            # Create a mask for selecting only that column
                            col_mask = np.zeros_like(joint_states)
                            col_mask[:, max_col_idx] = joint_states[:, max_col_idx]
                            # Update only that column in self.pB
                            
                            #i = self.pB[factor_idx][:, :, action] > 0
                            i = col_mask > 0
                            self.pB[factor_idx][:, :, action] = np.where(
                                i,
                                self.forgeting_rate * (self.pB[factor_idx][:, :, action] - self.pB_0[factor_idx][:, :, action]) +
                                self.pB_0[factor_idx][:, :, action] +
                                self.learning_rate * col_mask,
                                self.pB[factor_idx][:, :, action]
                            )
                            del joint_states, state_before, state_after, i, col_mask
                            
                            """
                            i = self.pB[factor_idx][:, :, action] > 0
                            self.pB[factor_idx][:, :, action] = np.where(
                                i,
                                self.forgeting_rate*(self.pB[factor_idx][:, :, action] - self.pB_0[factor_idx][:, :, action])
                                + self.pB_0[factor_idx][:, :, action]
                                + self.learning_rate*joint_states,
                                self.pB[factor_idx][:, :, action]
                            )
                            del joint_states, state_before, state_after, i
                            """
                        """
                        for policy_idx, policy in enumerate(self.policies):
                            action = policy[t-1, factor_idx]
                            state_before = copy.deepcopy(self.policy_dep_posteriors[policy_idx, t-1, factor_idx])
                            state_after = copy.deepcopy(self.policy_dep_posteriors[policy_idx, t, factor_idx])
                            joint_states = np.outer(state_after, state_before)
                            joint_states = joint_states*self.posterior_pi[t][policy_idx]
                            joint_states *= (self.B[factor_idx][:, :, action] > 0).astype("float")
                            i = joint_states > 0
                            self.pB[factor_idx][:, :, action] = np.where(
                                i,
                                self.forgeting_rate*(self.pB[factor_idx][:, :, action] - self.pB_0[factor_idx][:, :, action])
                                                    + self.pB_0[factor_idx][:, :, action]
                                                    +self.learning_rate*joint_states,
                                                    self.pB[factor_idx][:, :, action])
                            del joint_states, state_before, state_after, i
                        """ 
                self.B = self._normalize_colums(self.pB)
            if self.learning_E:
                self.pE = self.forgeting_rate*(self.pE - self.pE_0) + self.pE_0 + self.learning_rate*self.posterior_pi
                # negative free energy of e
                self.Fe = self.KL_dirichlet(self.pE, self.E)
        
        else:
            if self.continous_obs:
                self.external_lm.update_mu_sigma_vectorized(self.observations_cache, self.posteriors)
            else:
                # Learning in shallow inference after accumulating sufficient evidence in each learning window.
                if actual_t%self.learning_window == self.learning_window-1:
                    if self.learning_A:
                        for t_i in range(self.learning_window):
                            for modality_idx in range(len(self.pA)):
                                obs_mod = int(self.observations_cache[t_i, modality_idx])
                                A_mm = self.one_hot_encode(modality_idx, int(obs_mod), self.obs_dim)
                                for factor_idx in range(self.num_factors):
                                    A_mm = np.multiply.outer(A_mm, self.posteriors_cache[t_i,factor_idx])
                                                
                                i = self.pA[modality_idx] > 0
                                self.pA[modality_idx] = np.where(
                                    i,
                                    self.forgeting_rate * (self.pA[modality_idx] - self.pA_0[modality_idx]) +
                                    self.pA_0[modality_idx] +
                                    self.learning_rate * A_mm,  
                                    self.pA[modality_idx]
                                )                    
                                del A_mm

                    if self.learning_B:
                        
                        for t_i in range(self.learning_window):
                            if t_i > 0:
                                
                                for factor_idx in range(self.num_factors):
                                    action = int(self.action_posteriors_cache[factor_idx, t_i-1])
                                    if factor_idx not in self.controlable_states:
                                        continue
                                    
                                    state_before = self.posteriors_cache[t_i-1, factor_idx]
                                    state_after = self.posteriors_cache[t_i, factor_idx]
                                    joint_states = np.outer(state_after, state_before)
                                    joint_states *= (self.B[factor_idx][:, :, action] > 0).astype("float")
                                    
                                    # Get index of column containing the highest value
                                    max_col_idx = np.unravel_index(np.argmax(joint_states), joint_states.shape)[1]

                                    # Create a mask for selecting only that column
                                    col_mask = np.zeros_like(joint_states)
                                    col_mask[:, max_col_idx] = joint_states[:, max_col_idx]
                                    # Update only that column in self.pB
                                    
                                    #i = self.pB[factor_idx][:, :, action] > 0
                                    i = col_mask > 0
                                    self.pB[factor_idx][:, :, action] = np.where(
                                        i,
                                        self.forgeting_rate * (self.pB[factor_idx][:, :, action] - self.pB_0[factor_idx][:, :, action]) +
                                        self.pB_0[factor_idx][:, :, action] +
                                        self.learning_rate * col_mask,
                                        self.pB[factor_idx][:, :, action]
                                    )
                                    del joint_states, state_before, state_after, i, col_mask

                    if self.learning_D:
                        if actual_t == 0:

                            for factor_idx in range(self.num_factors):
                                i = self.pD[factor_idx] > 0
                                self.pD[factor_idx] = np.where(
                                    i,
                                    (self.pD[factor_idx] - self.pD_0[factor_idx]) 
                                    + self.pD_0[factor_idx] 
                                    + self.learning_rate * self.posteriors_cache[actual_t, factor_idx], 
                                )                

                                del i

    
    def softmax(self, x, axis = 0, gamma=1.0):
        return numerical_softmax(x, axis=axis, gamma=gamma)
    
    def softmax_whole(self, x, gamma=1.0):
        x_copy = copy.deepcopy(x)
        for i in range(len(x_copy)):
            x_copy[i] = self.softmax(x_copy[i], gamma=gamma)
        return x_copy
    
    def perform_modal_average(self):
        # Following function averages the posterior of states over policies.
        # by executing following function, we update the self.bayesian_mod_avg
        # which stores the posterior over states for each time step of the previous trial
        # average over all policies
        qs_temp = copy.deepcopy(self.policy_dep_posteriors)
        for factor_idx in range(self.num_factors):
            for tau in range(self.temporal_horizon):
                v_stack_states = np.vstack(qs_temp[:,tau,factor_idx])
                self.bayesian_mod_avg[tau, factor_idx] = v_stack_states.T.dot(self.posterior_pi[:])

        return self.bayesian_mod_avg
    """"
    def perform_modal_average(self):
            qs_temp = copy.deepcopy(self.policy_dep_posteriors)
            for policy_idx in range(len(self.policies)):
                qs_temp[policy_idx,:,:] = qs_temp[policy_idx,:,:] * self.current_posterior_pi[policy_idx]
            for t in range(self.temporal_horizon):
                for policy_idx in range(len(self.policies)):
                    self.bayesian_mod_avg[t, :] += qs_temp[policy_idx,t,:]
            del qs_temp
    """

    def update_policy_posterior(self, trial, t):
        # SPM Initialization: gamma(t) = gamma(t-1)
        gamma_t = self.gamma_previous

        # In SPM, 'beta' is the prior and 'posterior_beta' is the value we optimize
        # Initialize posterior_beta for this time step
        posterior_beta = self.beta_prior 
        
        tolerance = 1e-8
        previous_beta = None

        for ni in range(self.number_of_msg_passing):
            # 1. UPDATE POLICIES using the current gamma
            # Note: self.E is the preference/habit, G is expected free energy, F is VFE (evidence)
            ln_prior = self.log_stable(self.E) + gamma_t * self.G_policy
            self.prior_pi = self.softmax(np.float64(ln_prior), axis=None)
            
            ln_posterior = ln_prior + self.F_policy
            self.posterior_pi = self.softmax(np.float64(ln_posterior), axis=None)
            
            # 2. CALCULATE GRADIENT (G_error)
            # This is the difference in Expected Free Energy between prior and posterior
            G_error = (self.posterior_pi - self.prior_pi).dot(self.G_policy)
            
            # 3. UPDATE BETA (The gradient descent step)
            # beta_update = (current_estimate - prior) + error
            beta_update = (posterior_beta - self.beta_prior) + G_error
            
            # ..... 
            # /2 is faster, /10 is more stable.
            posterior_beta = posterior_beta - beta_update / 2 
            
            # 4. UPDATE GAMMA for the next iteration
            # This is the "Dopamine" update
            gamma_t = 1.0 / posterior_beta
            
            # 5. Convergence check
            if previous_beta is not None:
                if abs(posterior_beta - previous_beta) < tolerance:
                    break
            previous_beta = posterior_beta

        # Update class states and record history
        self.beta_posterior = posterior_beta
        self.gamma_previous = gamma_t
        self.perform_modal_average()
        if t%self.temporal_horizon == self.temporal_horizon-1:
            #store the beliefs at the end of the current planning window to be used
            # as the prior for the first message in the next planning window.
            self.previous_qs_T = self.bayesian_mod_avg[self.temporal_horizon-1] 

    def calculate_counterfactual_disparity(self, t, K= None):
        best_policy = int(np.argmin(-self.G_policy[t][:]))
        self.chosen_policy[t] = best_policy
        self.expected_obs_chosen[t] = self.policy_dep_expected_obs[best_policy, t, :]
        if K == None:
            for modality_idx in range(self.num_modalities):
                disparity = 0
                for policy_idx in range(self.num_policies):
                    if policy_idx == best_policy:
                        continue
                    for timestep in range(t, self.temporal_horizon):
                        p_best = self.policy_dep_expected_obs[best_policy, timestep][modality_idx]
                        p_cf   = self.policy_dep_expected_obs[policy_idx, timestep][modality_idx]
                        disparity += self.KL_categorical(p_best, p_cf)
                self.disparity_nu[t, modality_idx] += 2 / (1 + np.exp(-1 * disparity)) - 1
                #self.disparity_nu[modality_idx] = disparity
                #self.disparity_nu[modality_idx] = 2 / (1 + np.exp(-1 * disparity)) - 1
        
        else:
            # use G policy values to pick K worst
            G_t = -self.G_policy[t][:]
            G_t[best_policy] = np.inf
            cf_indices = np.argsort(G_t)[:K]

            
            for modality_idx in range(self.num_modalities):
                disparity = 0
                for policy_idx in cf_indices:
                    for timestep in range(t, self.temporal_horizon):
                        p_best = self.policy_dep_expected_obs[best_policy, timestep][modality_idx]
                        p_cf   = self.policy_dep_expected_obs[policy_idx, timestep][modality_idx]
                        disparity += self.KL_categorical(p_best, p_cf)
                self.disparity_nu[modality_idx] = 2 / (1 + np.exp(-1 * disparity)) - 1


 
    def is_normalized(self, dist):
        """
        Check whether a single distribution or a NumPy object array of conditional 
        categorical distributions is normalized along the first axis (categories).

        Args:
            dist (np.ndarray or np.ndarray[object]): Distribution(s) to check

        Returns:
            bool: True if all distributions are normalized, False otherwise
        """
        # Helper function for a single array
        def check_array(arr):
            return np.allclose(arr.sum(axis=0), 1.0)

        if isinstance(dist, np.ndarray) and dist.dtype == object:
            # NumPy object array: check each sub-array
            return all(check_array(arr) for arr in dist)
        else:
            # Single array
            return check_array(dist)
            
    
    def _validate_and_assign_matrix(self, matrix, default=None):
        """Helper function to validate a matrix and assign a default if necessary."""
        if matrix is None and default is not None:
            matrix = default
        elif matrix is not None:
            self._is_the_matrix_valid(matrix)
        return matrix  # Converting matrix to an object array                    
                    
                    
    def get_joint_likelihood(self, obs):
        likelihood = 1
        for obs_modality, observation in enumerate(obs):
            if observation >= self.A.shape[1]:  
                raise ValueError(f"Observation index {observation} exceeds matrix dimensions.")
            likelihood *= self.A[obs_modality][observation, :]
        return likelihood
        
    def expected_log_likelihood(self, obs, factor, policy_idx, tau):
        log_likelihoods = self.create_object_tensor('zeros', 1, last_dim=self.states_dim[factor])
        if obs is not None:
            for modal_idx, modality in enumerate(self.A):
                lnA = self.log_stable(np.take(modality, obs[modal_idx], axis=0))
                lnA = np.moveaxis(lnA, factor, -1)
                for fj in range(self.num_factors):
                    if fj != factor:
                        lnAs = np.tensordot(lnA, self.policy_dep_posteriors[policy_idx, tau, fj], axes=(0,0))
                        del lnA
                        lnA = lnAs
                        del lnAs
                log_likelihoods += lnA
        return log_likelihoods
    
    def expected_log_likelihood_einsum(self, obs, factor, policy_idx=0, tau=0):
        """
        Calculates the expected log-likelihood for a factor using np.einsum.
        This is more efficient as it avoids creating intermediate arrays.
        """
        if self.deep_inference:
            # Initialize with zeros for the states of the target factor
            log_likelihoods = np.zeros(self.states_dim[factor])
            

            if obs is not None:
                for modal_idx in range(self.num_modalities):
                    # Get the log-likelihood slice for the current observation
                    # This is a tensor with dimensions for each state factor
                    if factor not in self.mod_dep[modal_idx]:
                        continue # Skip if the modality does not depend on the target factor

                    if not self.continous_obs:
                        lnA = self.log_stable(np.take(self.A[modal_idx], obs[modal_idx], axis=0))
                    else:
                        #t_start = time.perf_counter()
                        lnA = self.log_stable(self.external_lm.likelihoods(obs[modal_idx], modal_idx))  
                        #t_end = time.perf_counter()
                        #print(f"Log-likelihood computation for modality {modal_idx} took {t_end - t_start:.4f} seconds.")

                    if len(self.mod_dep[modal_idx]) == 1 and self.mod_dep[modal_idx][0] == factor:
                        # If the modality only depends on the target factor, we can skip einsum
                        expected_lnA = lnA
                    else:
                        # Pre-fetch the posteriors that will be used for marginalization
                        posteriors_to_marginalize = [
                            self.policy_dep_posteriors[policy_idx, tau, f] 
                            for f in (self.mod_dep[modal_idx]) if f != factor
                        ]

                        # Dynamically create the einsum string
                        # e.g., for 3 factors and target factor 0: 'ijk,j,k->i'
                        alphabet = string.ascii_lowercase
                        all_factors_str = ''.join([alphabet[f] for f in (self.mod_dep[modal_idx])])
                        # This part needs to list each individual posterior's dimension
                        # For 'b,c,d,e,f' each letter is a separate operand in the string
                        other_factors_dims = [alphabet[f] for f in (self.mod_dep[modal_idx]) if f != factor]
                        
                        other_factors_str = ",".join(other_factors_dims)
                        if other_factors_dims: # Check if there are other factors to marginalize
                            other_factors_str = ",".join(other_factors_dims)
                            einsum_str = f'{all_factors_str},{other_factors_str}->{alphabet[factor]}'
                        else: # No other factors to marginalize (e.g., when num_factors is 1)
                            einsum_str = f'{all_factors_str}->{alphabet[factor]}'
                        
                        expected_lnA = np.einsum(einsum_str, lnA, *posteriors_to_marginalize)
                        #if factor == 2:
                            #self.plot_lnA_model(lnA, goal_pos=(237, 471))
                            #self.plot_expected_posterior(expected_lnA, obs, goal_pos=(237, 471))
                        

                    
                    log_likelihoods += expected_lnA
        else:
            # Initialize with zeros for the states of the target factor
            log_likelihoods = np.zeros(self.states_dim[factor])
            precision = 1
            if obs is not None:
                for modal_idx in range(self.num_modalities):
                    # Get the log-likelihood slice for the current observation
                    # This is a tensor with dimensions for each state factor
                    if factor not in self.mod_dep[modal_idx]:
                        continue # Skip if the modality does not depend on the target factor
                    if factor == 2 and modal_idx == 2:
                        precision = 1
                    if factor == 1 and modal_idx == 1:
                        precision = 0.1
                    if factor == 0 and modal_idx == 1:
                        precision = 1

                    

                    # Get the log-likelihood slice for the current observation
                    if self.continous_obs:
                        lnA = self.log_stable(self.external_lm.likelihoods(obs[modal_idx], modal_idx))
                    else:
                        # This is a tensor with dimensions for each state factor
                        lnA = self.log_stable(np.take(self.A[modal_idx], obs[modal_idx], axis=0))

                    if len(self.mod_dep[modal_idx]) == 1 and self.mod_dep[modal_idx][0] == factor:
                        # If the modality only depends on the target factor, we can skip einsum
                        expected_lnA = lnA
                    
                    else:
                        # Pre-fetch the posteriors that will be used for marginalization
                        posteriors_to_marginalize = [
                            self.posteriors[f] 
                            for f in (self.mod_dep[modal_idx]) if f != factor
                        ]
                        # Dynamically create the einsum string
                        # e.g., for 3 factors and target factor 0: 'ijk,j,k->i'
                        alphabet = string.ascii_lowercase
                        all_factors_str = ''.join([alphabet[f] for f in (self.mod_dep[modal_idx])])
                        # This part needs to list each individual posterior's dimension
                        # For 'b,c,d,e,f' each letter is a separate operand in the string
                        other_factors_dims = [alphabet[f] for f in (self.mod_dep[modal_idx]) if f != factor]
                        
                        if other_factors_dims: # Check if there are other factors to marginalize
                            other_factors_str = ",".join(other_factors_dims)
                            einsum_str = f'{all_factors_str},{other_factors_str}->{alphabet[factor]}'
                        else: # No other factors to marginalize (e.g., when num_factors is 1)
                            einsum_str = f'{all_factors_str}->{alphabet[factor]}'
                        
                        # Perform the entire marginalization in one step
                        expected_lnA = np.einsum(einsum_str, lnA, *posteriors_to_marginalize)
                    
                    log_likelihoods += expected_lnA*precision
                
        return log_likelihoods
            

    def plot_lnA_model(self, lnA, goal_pos):
        """
        lnA: The log-likelihood tensor. 
            Expected shape: (num_observations, num_agent_states, num_goal_states)
        goal_pos: [phys_x, phys_y] of the goal to 'anchor' the plot
        """
        X_dim, Y_dim = self.states_dim[0], self.states_dim[1]
        goal_x, goal_y = goal_pos[0], goal_pos[1]

        # 1. Identify which 'Goal State' index corresponds to the physical goal
        g_idx_x = int(np.clip(goal_x / (500 / X_dim), 0, X_dim - 1))
        g_idx_y = int(np.clip(goal_y / (500 / Y_dim), 0, Y_dim - 1))
        
        # Flatten index based on your 'ij' mapping (x * Y_dim + y)
        goal_state_idx = g_idx_x * Y_dim + g_idx_y

        # 2. Extract the model for a specific observation (e.g., a specific RSI bin)
        # If lnA is (observations, agent_states, goal_states)
        # We take the mean across observations or pick a specific one
        # Here, we take the slice for our specific goal
        model_slice = lnA[:, :, goal_state_idx] 
        
        # If the model has multiple observation types, we average them 
        # to see the general energy landscape
        #model_2d = model_slice.mean(axis=0).reshape(X_dim, Y_dim)

        plt.figure(figsize=(10, 8))
        
        # 3. Plot the landscape
        # This shows the log-probability gradient
        im = plt.imshow(model_slice.T, origin='upper', cmap='viridis',
                        extent=[0, 500, 500, 0], interpolation='gaussian')
        plt.colorbar(im, label='Log-Likelihood (lnA)')

        # 4. Mark the Goal
        plt.scatter(goal_x, goal_y, color='lime', marker='*', s=300, 
                    label='Goal (The Anchor)', edgecolors='black')

        plt.title(f"Likelihood Field (lnA) for Goal at {goal_pos}")
        plt.xlabel("Physical X (cm)")
        plt.ylabel("Physical Y (cm)")
        plt.legend()
        plt.show()
    
    def plot_expected_posterior(self, expected_lnA, obs, goal_pos):
        """
        expected_lnA: array of shape (25,) from the einsum calculation
        obs: [phys_x, phys_y, rsi_val]
        goal_pos: [true_goal_x, true_goal_y]
        """
        X_dim, Y_dim = self.states_dim[0], self.states_dim[1]
        agent_x, agent_y = obs[0], obs[1]
        
        # 1. Convert Log-Likelihood to Probability Space
        # We subtract the max for numerical stability (Softmax-style)
        prob_vec = np.exp(expected_lnA - np.max(expected_lnA))
        prob_vec /= prob_vec.sum() # Normalize so it sums to 1
        
        # 2. Reshape to the 5x5 grid
        # Match the 'ij' indexing used in your precompute function
        posterior_map = prob_vec.reshape(X_dim, Y_dim)
        
        plt.figure(figsize=(10, 8))
        
        # 3. Plot the Posterior Heatmap
        # We use .T (transpose) to align X-columns and Y-rows with origin='upper'
        im = plt.imshow(posterior_map, origin='upper', cmap='plasma',
                        extent=[0, 500, 500, 0], interpolation='gaussian')
        plt.colorbar(im, label='Posterior Probability P(Goal | Obs)')

        # 4. Overlay Agent and Goal
        plt.scatter(agent_x, agent_y, color='cyan', s=100, label='Agent Current Pos', edgecolors='black')
        plt.scatter(goal_pos[0], goal_pos[1], color='lime', marker='*', s=300, label='True Goal', edgecolors='black')

        plt.title("Posterior Inference: Agent's Belief of Goal Location")
        plt.xlabel("Physical X (cm)")
        plt.ylabel("Physical Y (cm)")
        plt.legend()
        plt.show()
    
    def transpose_Bfa(self, B_fa):
        return transpose_transition(B_fa, epsilon=EPS_VAL)
    
    def transpose_Bfa_temp(self, B_fa):
        return transpose_transition(
            B_fa,
            epsilon=EPS_VAL,
            normalize=True,
        )
    
    def _transpose_B_matrix(self):
        T_pB = copy.deepcopy(self.pB)
        for factor_idx in range(self.num_factors):
            if factor_idx in self.controlable_states:
                for action_idx in range(self.controls_dim[factor_idx]):
                    T_pB[factor_idx][:,:,action_idx] = self.transpose_Bfa(self.pB[factor_idx][:,:, action_idx])
            else:
                T_pB[factor_idx] = copy.deepcopy(self.pB[factor_idx])
        return T_pB


    def create_object_tensor(self, dist='uniform', *dims, last_dim=None):
        """
        Create an object tensor filled with different distributions.

        If no dimensions are provided, it defaults to:
        (num_policies, temporal_horizon, num_factors).

        last_dim: This defines the size of the last dimensions.
                Ex: if dims = (3, 3, 4), last_dim = [3, 2, 1, 8] 

        Supported distributions:
        - 'uniform': Equal probability over states.
        - 'zeros': All zeros.
        - 'ones': All ones.
        - 'random': Random values sampled from a uniform distribution [0,1].
        """
        if not last_dim:
            last_dim = [1]

        # Default dimensions if none provided
        if not dims:
            dims = (len(self.policies), self.temporal_horizon, self.num_factors)

        # Ensure last_dim has correct size
        if isinstance(last_dim, int):  
            last_dim = [last_dim]  # Convert single int to list
        if len(last_dim) == 1:  
            last_dim = last_dim * dims[-1]  # Apply the same value across the last dimension

        # Initialize the tensor
        array = np.empty(dims, dtype=object)

        for indices in np.ndindex(array.shape):
            last_dim_idx = indices[-1]  # Index in the last dimension

            # Ensure within range
            if last_dim_idx >= len(last_dim):  
                size = last_dim[-1]  # Default to last available size
            else:
                size = last_dim[last_dim_idx]  

            # Assign probability distribution
            if dist == 'uniform' or dist == 'ones':
                array[indices] = np.ones(size)
            elif dist == 'zeros':
                array[indices] = np.zeros(size)
            #elif dist == 'ones':
                #array[indices] = np.ones(size)
            elif dist == 'random':
                array[indices] = np.random.rand(size)
            elif dist == 'NaN':
                array[indices] = np.full(size, np.nan)
            else:
                raise ValueError(f"'{dist}' is not a recognized distribution. Choose from: 'uniform', 'zeros', 'ones', 'random'.")
        if dist == 'uniform':
        # Normalize the array along the last dimension (axis=-1)
            
            if len(dims) == 1 and last_dim[0] == 1:
                array /= np.sum(array)
            else:
                for i in range(array.shape[-1]):
                    array[..., i] /= last_dim[i]
        if len(dims) == 1 and dims[0] == 1:
            array = array[0]
        return array

    def conver_to_joint_posterior(self):
        for policy_idx in range(len(self.policy_dep_posteriors)):
            for t in range(len(self.policy_dep_posteriors[0])):
                # Extract the first factor's probability distribution
                joint_prob = self.policy_dep_posteriors[policy_idx, t, 0] # probabilities of state factor 0
                # Iterate over the remaining factors and compute the outer product
                for factor in range(1, self.num_factors):  # Start from the second factor
                    factor_prob = self.policy_dep_posteriors[policy_idx, t, factor] 
                    joint_prob = np.multiply.outer(joint_prob, factor_prob)
                self.joint_policy_dep_posteriors[policy_idx, t] = joint_prob

    def log_stable_E(self, array):
        """
        Adds small epsilon value to an array before applying natural log for each element in arrays.
        This ensures numerical stability when working with very small numbers.
        """
        arr = copy.deepcopy(array)
        if isinstance(arr, Iterable):
            # Iterate through each subarray in the array
            for idx, subarr in enumerate(arr):
                if isinstance(subarr, np.ndarray):  # Check if it's an ndarray
                    # Apply log with small epsilon to each element in the subarray
                    arr[idx] = np.log(subarr + EPS_VAL)  # Modify the subarray with log values
                else:
                    # If subarr is a scalar, just apply the log to it directly
                    arr[idx] = np.log(subarr + EPS_VAL)
        else:
            arr = np.log(arr + EPS_VAL)
        
        return arr
    
    def log_stable(self, array, eps=1e-16):
        return log_stable_probability(array, eps=eps)
    
    def log_stable_numpy_obj(self, numpy_obj, eps=1e-16):
        # This function is designed to handle numpy objects (arrays or arrays) and apply log with epsilon
        # @NOTE This function can only be used if all the factors has same number of cardinalities
        # @NOTE this function changes the original numpy object to a regular numpy array expanding an axes for cardinalities
        return log_stable_object_array(numpy_obj, eps=eps)

    def update_observation(self, obs):
        # Store the new observation
        self.latest_obss.append(tuple(obs))
        """
        # Ensure buffer length is exactly temporal_horizon
        while len(self.latest_obss) < self.temporal_horizon:
            # Create a None observation 
            none_obs = None
            self.latest_obss.append(none_obs)

        """
        # Ensure buffer length is exactly temporal_horizon
        while len(self.latest_obss) < self.temporal_horizon:
            # Create a random observation based on obs_dim
            random_obs = []
            for i in range(len(self.obs_dim)):
                random_obs.append(random.randint(0, self.obs_dim[i] - 1))  # Fill with random values

            # Append the generated random observation
            self.latest_obss.append(tuple(random_obs))
        
        # Keep only the latest temporal_horizon observations
        self.latest_obss = self.latest_obss[-self.temporal_horizon:]

    def one_hot_encode(self, obs_modality, obs_value, obs_dims):
        return one_hot(obs_value, obs_dims[obs_modality]).tolist()
    
    def wnorm_new(self, p, val=np.exp(-16)):
        # @NOTE here the equation (40) is implimented
        # compared to the MATLAB code.
        # according to the equation (40); w = 0.5*(avg - norm)
        return wnorm(p, val=val)
    
    def KL_categorical(self, p, q):
        return categorical_kl_terms(p, q)

    
    def KL_dirichlet(self, q, p):
        # @NOTE this function perform the same operation as in spm_KL_dir
        # using the python functions gammaln and psi from scipy.special
        """
        Compute KL divergence between two Dirichlet distributions Q and P.
        
        Parameters:
        q : np.array
            Concentration parameter matrix of Q (shape: N x D)
        p : np.array
            Concentration parameter matrix of P (shape: N x D)
            
        Returns:
        d : float
            KL divergence sum over columns
        """
        return dirichlet_kl(q, p)
        
    def log_beta(self, z):
        """
        Compute the log Beta function for vectors and higher-dimensional arrays.
        
        Parameters:
        z : np.array
            Input concentration parameters.
        
        Returns:
        y : np.array
            Log Beta function values.
        """
        return numerical_log_beta(z)
        
    def get_expected_posterior_entropy(self):

        total_entropy = 0.0

        for factor_idx in range(self.num_factors):

            qs = np.array(self.policy_dep_posteriors[:, :, factor_idx].tolist(), dtype=float)
            log_qs = self.log_stable_numpy_obj(qs)

            # entropy per policy (sum over tau and states)
            H_pi = -np.sum(qs * log_qs, axis=tuple(range(1, qs.ndim)))

            # normalize entropy (optional but recommended)
            H_pi = H_pi / np.log(self.states_dim[factor_idx])

            # weight by policy posterior q(pi)
            total_entropy += np.sum(self.posterior_pi * H_pi)

        return total_entropy
    
    def get_predictive_divergence(self, t):
        divergence = 0
        for m in range(self.num_modalities):
            predictions_for_m0 = np.stack([self.policy_dep_expected_obs[p, 0, m] for p in range(self.num_policies)])
            predictions_for_m1 = np.stack([self.policy_dep_expected_obs[p, 1, m] for p in range(self.num_policies)])
            # KL per policy, shape: (num_policies, num_obs) -> (num_policies,)
            kl_per_policy = np.sum(self.KL_categorical(predictions_for_m0, predictions_for_m1), axis=-1)
            # Weight by policy probability
            divergence += np.sum(np.sort(kl_per_policy)[-3:])
        return divergence
    
    def get_observation_divergence(self, t):
        surprise = 0
        #for m in range(self.num_modalities):
        predictions_for_m0 = np.stack([
            self.policy_dep_expected_obs[p, 0, 2] 
            for p in range(self.num_policies)
        ])
        
        # Discretize raw observation into bin index
        num_bins = predictions_for_m0.shape[-1]  # 100
        raw_obs = self.observations[t][2]
        obs_idx = int(np.clip(
            raw_obs / 30.0 * num_bins,
            0, num_bins - 1
        ))
        
        # Surprise: -log p(actual obs | policy)
        # shape: (num_policies,)
        surprise += -np.log(predictions_for_m0[:, obs_idx] + 1e-12)
        
        return np.mean(surprise)            #.dot(self.posterior_pi)  # shape: (num_policies,) — surprise per policy
    
    def get_posterior_variance(self, t):
        var = 0
        for factor_idx in range(self.num_factors):
            qs = self.bayesian_mod_avg[t%self.temporal_horizon, factor_idx]
            mu = np.sum(np.arange(len(qs)) * qs)
            var += (np.sum(qs * (np.arange(len(qs)) - mu) ** 2))/500**2
        return var
    
    def get_expected_policy_ambiguity(self,t):
        A = np.array(self.H_Qo, dtype=float)/(len(range((t+1)%self.temporal_horizon, self.temporal_horizon)))   # A(pi)
        q_pi = np.array(self.posterior_pi, dtype=float)     # q(pi)

        expected_A = np.sum(q_pi * A)

        return expected_A
    
    def get_expected_policy_risk(self,t):
        R = np.array(self.risk, dtype=float)/(len(range((t+1)%self.temporal_horizon, self.temporal_horizon)))   # A(pi)
        q_pi = np.array(self.posterior_pi, dtype=float)     # q(pi)

        expected_R = np.sum(q_pi * R)

        return expected_R
    
    def get_model_spread(self, t):
        spread = 0
        #for factor_idx in range(self.num_factors):
        Qx = self.bayesian_mod_avg[0, 0]
        Qy = self.bayesian_mod_avg[0, 1]
        Qg = self.bayesian_mod_avg[0, 2]
        mu = self.external_lm.mu_signal
        mu_mean = np.sum(Qy[None,:,None] *Qx[:,None,None] * Qg[None,None,:] * mu)
        S = np.sum(
                        Qy[None,:,None] * Qx[:,None,None] * Qg[None,None,:] *
                        (mu - mu_mean)**2
                    )
        return S
    
    def get_spatial_fi(self, t):
        return self.external_lm.compute_sensitivity(self.observations[t])
    

    def get_stats(self, t):
        stats = {
            'info_gain_proxy': self.external_lm.compute_sensitivity(self.observations[t]),
            'mean_surprise': self.get_observation_divergence(t)
        }
        return stats
    
    def update_external_likelihood_model(self, new_dims):
        
        self.external_lm = self.lm.update_likelihood_model(new_dims)
        

    
        
def spm_psi(A):
    # This is the python implimentation of the Karl Friston's
    # spm_psi MATLAB function
    # Copyright (C) 2015 Wellcome Trust Centre for Neuroimaging
    # by Karl Friston
    return numerical_spm_psi(A)


class FeatureDeviationDetector:
    def __init__(self, n_features, alpha=0.95, h=5.0):
        """
        n_features: length of your feature vector
        alpha: smoothing factor for running mean/cov
        h: detection threshold on deviation score
        """
        self.alpha = alpha
        self.h = h
        self.mu = np.zeros(n_features)
        self.cov = np.eye(n_features)
        self.inv_cov = np.linalg.inv(self.cov)
        self.initialized = False
        self.change_points = []

    def update(self, features, t):
        x = np.asarray(features)

        if not self.initialized:
            self.mu = x
            self.cov = np.eye(len(x))
            self.inv_cov = np.linalg.inv(self.cov)
            self.initialized = True
            return False

        # Mahalanobis distance from running mean
        diff = x - self.mu
        d2 = float(diff.T @ self.inv_cov @ diff)

        # Update running statistics (exponential moving)
        self.mu = self.alpha * self.mu + (1 - self.alpha) * x
        centered = x - self.mu
        self.cov = self.alpha * self.cov + (1 - self.alpha) * np.outer(centered, centered)
        # regularize covariance and invert
        self.inv_cov = np.linalg.inv(self.cov + 1e-6 * np.eye(len(x)))

        detected = d2 > self.h
        if detected:
            self.change_points.append(t)
        return detected
