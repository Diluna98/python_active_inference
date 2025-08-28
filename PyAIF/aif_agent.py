import numpy as np
import copy
import random
import os
import sys
import time
import string
from PyAIF import utils
from collections.abc import Iterable
from scipy.special import gammaln, psi
import multiprocessing
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import shared_memory

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
    exp_x = np.exp(gamma * x - np.max(gamma * x))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def transpose_Bfa(B_fa):
    # @NOTE: this function is not correct
    B_T = copy.deepcopy(B_fa)
    
    B_T = np.transpose(B_T, (1, 0))  # Transpose state dimensions, keep actions
    B_T = np.divide(B_T, B_T.sum(axis=0))
    B_T = np.nan_to_num(B_T, nan=0.0)  # Replace NaNs with zero
    
    return B_T

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
    p = X.copy()
    for f in reversed(range(len(x))):
        p = np.tensordot(p, x[f], axes=(f + 1, 0))
    return p

def log_stable(array, val=np.exp(-16)):
    """
    Adds small epsilon value to an array before natural logging it
    """
    return np.log(array + val)

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
        self, A, B, states_dim, obs_dim, controls_dim, controlable_states,
        trial_length=1, number_of_msg_passing = 100, learning_rate = 0.2,
        forgeting_rate = 0.99, trials = 100, alpha = 512, zeta = 0.01, timeconst = 1, D=None,
        C=None, E=None, policies=False, policy_pruning = False,
        learning_D = False, learning_A = False, learning_B = False, learning_E = False
    ):
        # Construct policies
        if policies == False:
            self.policies = utils.construct_policies(states_dim, controls_dim, trial_length-1, controlable_states)
        else:
            self.policies = policies        
        self.policy_pruning = policy_pruning
        self.num_factors = len(states_dim)
        self.num_modalities = len(obs_dim)
        self.num_policies = len(self.policies)
        self.states_dim = states_dim
        self.obs_dim = obs_dim
        self.controls_dim = controls_dim
        self.num_trials = trials
        self.pA = A
        self.pA_0 = copy.deepcopy(self.pA)
        self.pA_prior = copy.deepcopy(self.pA)
        self.pA_complexity = copy.deepcopy(self.pA)
        self.pB = B
        self.pB_0 = copy.deepcopy(self.pB)
        self.pB_prior = copy.deepcopy(self.pB)
        self.pB_complexity = copy.deepcopy(self.pB)
        self.pD = D
        self.pD_0 = copy.deepcopy(self.pD)
        self.pD_prior = copy.deepcopy(self.pD)
        self.pD_complexity = copy.deepcopy(self.pD)
        self.pE = E if E else self.create_object_tensor('ones', 1, last_dim = [len(self.policies)])
        self.pE_0 = copy.deepcopy(self.pE)
        self.pC = C
        for num_el in range(len(C)):
            self.pC[num_el] += 1/32 
        
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

        self.temporal_horizon = trial_length
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
        self.observations = self.create_object_tensor('NaN', self.num_trials, self.temporal_horizon, self.num_modalities)
        self.bayesian_mod_avg = self.create_object_tensor('zeros', self.num_trials, self.temporal_horizon, self.num_factors, last_dim=self.states_dim)
        self.Fd = self.create_object_tensor('zeros', 1, last_dim = [self.num_factors])
        self.Fb = copy.deepcopy(self.Fd)
        self.Fa = self.create_object_tensor('zeros', 1, last_dim = [self.num_modalities])
        self.Fe = 0
        self.alpha = alpha
        self.zeta = zeta
        #self.action_selection = "deterministic" # use "stochastic" for action selection with some randomness
        #self.action_selection = "random"
        self.action_selection = "marginal" # use "stochastic" for action selection with some randomness
        
        self.timeconst = timeconst #time constant for gradient descent
        self.gamma_0 = None
        self.posterior_beta = None
        self.total_dop_res = self.number_of_msg_passing * self.temporal_horizon
        #self.gamma_update = self.create_object_tensor('zeros', self.num_trials, self.total_dop_res)

        self.learning_A = learning_A
        self.learning_B = learning_B
        self.learning_E = learning_E
        self.learning_D = learning_D

        self.previous_lr = copy.deepcopy(self.learning_rate)

    def initialize_variables(self):
        self.policy_dep_posteriors = self.create_object_tensor(last_dim=self.states_dim)
        self.single_policy_dep_posteriors = copy.deepcopy(self.policy_dep_posteriors[0,:,:])       
        self.posterior_pi = self.create_object_tensor('zeros', self.temporal_horizon, last_dim = [len(self.policies)])
        #self.posterior_updates = self.create_object_tensor('NaN', self.total_dop_res, last_dim = [len(self.policies)])
        self.prior_pi = self.create_object_tensor('zeros', self.temporal_horizon, last_dim = [len(self.policies)])
        self.action_posteriors = self.create_object_tensor('zeros', self.num_factors, self.temporal_horizon - 1)       
        #self.action_confidance = self.create_object_tensor('ones', self.temporal_horizon - 1, self.num_factors, last_dim=self.controls_dim)
        self.vfe_ft = self.create_object_tensor('zeros', len(self.policies), self.temporal_horizon, self.number_of_msg_passing, self.temporal_horizon, self.num_factors)
        #self.normalized_firing_rates = self.create_object_tensor('NaN', len(self.policies), self.temporal_horizon, self.temporal_horizon, self.number_of_msg_passing, last_dim=self.states_dim)
        #self.prediction_error = self.create_object_tensor('NaN', len(self.policies), self.temporal_horizon, self.temporal_horizon, self.num_factors)
        self.F_policy = self.create_object_tensor('zeros', self.temporal_horizon, last_dim = [len(self.policies)])
        self.G_policy = self.create_object_tensor('zeros', self.temporal_horizon, last_dim = [len(self.policies)]) 
        
        self.gamma = self.create_object_tensor('NaN', self.temporal_horizon) 
        self.beta_posterior = 1
        self.beta_prior = 1
        self.gamma[0] = 1/self.beta_posterior


    def normalize_columns(self):
        self.A = self._normalize_colums(self.pA)
        self.B = self._normalize_colums(self.pB)
        self.D = self._normalize_colums(self.pD)
        self.E = self._normalize_colums(self.pE)
        self.C = self.softmax_whole(self.pC)
        for modality_idx in range(self.num_modalities):
            self.C[modality_idx] = self.log_stable(self.C[modality_idx])

        return copy.deepcopy(self.D), copy.deepcopy(self.B), copy.deepcopy(self.A)

    def _setup_shared_memory(self):
        # Helper to consolidate common shared memory setup logic
        # This helper is for 1-level deep object arrays (e.g., A, B, C, D, E, if they are like that)

        def _flatten_nested_numpy_arrays(nested_object_array):
            """
            Recursively flattens a NumPy object array containing other NumPy arrays
            into a list of (actual_ndarray, original_indices_tuple) tuples.
            """
            flat_list = []
            
            # Helper to traverse
            def _traverse(arr, current_indices):
                if isinstance(arr, np.ndarray) and arr.dtype == object:
                    # It's an object array, so iterate its elements
                    # Using flatiter to handle multi-dimensional object arrays easily
                    for i, item in enumerate(arr.flat):
                        # Calculate the multi-dimensional index
                        # Need to convert 1D flat index to multi-dimensional index
                        multi_idx = np.unravel_index(i, arr.shape)
                        _traverse(item, current_indices + multi_idx)
                elif isinstance(arr, np.ndarray): # This is the numerical NumPy array we want
                    flat_list.append((arr, current_indices))
                else:
                    # Handle cases where elements might not be NumPy arrays (e.g., if any were None, or simple Python objects)
                    # This should ideally not happen if your data is well-formed
                    raise TypeError(f"Unexpected non-numpy element found at {current_indices}: {type(arr)}")

            _traverse(nested_object_array, ()) # Start with empty index tuple
            return flat_list

        def _create_shm_for_deeply_nested_object_array(nested_obj_array):
            """
            Creates a single shared memory segment for a deeply nested NumPy object array
            where the innermost elements are numerical NumPy arrays of varying shapes.
            Returns the shared memory object and metadata for reconstruction.
            """
            
            # Flatten the nested structure to get all actual numerical arrays and their original paths
            flat_numerical_arrays_with_indices = _flatten_nested_numpy_arrays(nested_obj_array)

            metadata = []
            total_bytes = 0

            # First pass: calculate total size and build preliminary metadata
            for actual_ndarray, original_indices in flat_numerical_arrays_with_indices:
                metadata.append({
                    'original_indices': original_indices, # Keep track of where this came from in original structure
                    'shape': actual_ndarray.shape,
                    'dtype': str(actual_ndarray.dtype),
                    'offset': total_bytes # This will be updated in second pass for absolute offsets
                })
                total_bytes += actual_ndarray.nbytes

            # Create the shared memory segment
            shm = shared_memory.SharedMemory(create=True, size=total_bytes)
            
            current_offset = 0
            # Second pass: copy data into shared memory and update absolute offsets in metadata
            for i, (actual_ndarray, original_indices) in enumerate(flat_numerical_arrays_with_indices):
                shm.buf[current_offset : current_offset + actual_ndarray.nbytes] = actual_ndarray.tobytes()
                metadata[i]['offset'] = current_offset # Store the actual offset in the single SHM
                current_offset += actual_ndarray.nbytes
            
            return shm, {'name': shm.name, 'metadata': metadata, 'original_outer_shape': nested_obj_array.shape}
        
        def _create_shm_for_single_level_object_array(obj_array):
            metadata = []
            total_bytes = 0
            for i, inner_array in enumerate(obj_array):
                # Ensure the inner_array is indeed a numpy array and not object dtype itself
                if not isinstance(inner_array, np.ndarray) or inner_array.dtype == object:
                    raise TypeError(f"Element {i} in object array is not a numerical numpy.ndarray. "
                                    f"Found type: {type(inner_array)}, dtype: {inner_array.dtype}. "
                                    f"Use _create_shm_for_deeply_nested_object_array if it's nested.")
                
                metadata.append({
                    'idx': i,
                    'shape': inner_array.shape,
                    'dtype': str(inner_array.dtype),
                    'offset': total_bytes # This is updated after copy
                })
                total_bytes += inner_array.nbytes

            shm = shared_memory.SharedMemory(create=True, size=total_bytes)
            current_offset = 0
            for inner_array in obj_array:
                shm.buf[current_offset : current_offset + inner_array.nbytes] = inner_array.tobytes()
                current_offset += inner_array.nbytes
            
            return shm, {'name': shm.name, 'metadata': metadata}


        print("Setting up shared memory for A...")
        # Assuming A, B, C, D, E are 1-level deep object arrays (e.g., A[0] is float64 array)
        # You need to verify this for each. If they are also deeply nested like P, use the new helper.
        self.A_shm, self.A_shm_info = _create_shm_for_single_level_object_array(self.A)
        print(f"A shared memory created: {self.A_shm_info['name']}")

        #print("Setting up shared memory for B...")
        #self.B_shm, self.B_shm_info = _create_shm_for_single_level_object_array(self.B)
        #print(f"B shared memory created: {self.B_shm_info['name']}")

        print("Setting up shared memory for C...")
        self.C_shm, self.C_shm_info = _create_shm_for_single_level_object_array(self.C)
        print(f"C shared memory created: {self.C_shm_info['name']}")

        #print("Setting up shared memory for D...")
        #self.D_shm, self.D_shm_info = _create_shm_for_single_level_object_array(self.D)
        #print(f"D shared memory created: {self.D_shm_info['name']}")

        #print("Setting up shared memory for E...")
        #self.E_shm, self.E_shm_info = _create_shm_for_single_level_object_array(self.E)
        #print(f"E shared memory created: {self.E_shm_info['name']}")

        print("Setting up shared memory for state posteriors (P)...")
        # *** USE THE NEW HELPER FOR DEEPLY NESTED ARRAYS ***
        self.P_shm, self.P_shm_info = _create_shm_for_deeply_nested_object_array(self.policy_dep_posteriors)
        print(f"P shared memory created: {self.P_shm_info['name']}")

        #print("Setting up shared memory for policies...")
        # If self.policies is also deeply nested like P, use the new helper
        # Otherwise, use _create_shm_for_single_level_object_array
        #self.policies_shm, self.policies_shm_info = _create_shm_for_single_level_object_array(self.policies) # Assuming it's also deeply nested
        #print(f"Policies shared memory created: {self.policies_shm_info['name']}")

    def _cleanup_shared_memory(self):
        print(f"Main process {os.getpid()}: Cleaning up shared memory.")
        self.A_shm.close()
        self.A_shm.unlink()
        self.C_shm.close()
        self.C_shm.unlink()
        self.P_shm.close()
        self.P_shm.unlink()

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
        obs_taus = self.observations[trial, :, :]
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

    
    def infer_states(self, trial, t): #implimentation of the MMP

        #@NOTE: Policy_pruning functionality needs to be debugged.
        if self.policy_pruning:
            if t > 0:
                temp_F = np.array(np.log(self.posterior_pi[t-1][:]))
                mask = (temp_F - np.max(temp_F)) > -self.zeta
                self.policies = [p for p, m in zip(self.policies, mask) if m]

        for policy_idx, policy in enumerate(self.policies):
            depolarization = None
            F = None
            for nmp in range(self.number_of_msg_passing):  # Number of gradient descent iterations
                previous_F = F
                self.F_policy[t][policy_idx] = previous_F
                F = 0
                for factor in range(self.num_factors):
                    for tau in range(self.temporal_horizon):
                        third_msg = self.create_object_tensor('zeros', 1, last_dim=self.states_dim[factor])
                        depolarization = self.log_stable(self.policy_dep_posteriors[policy_idx, tau, factor])
                        if tau <= t:
                            # Third message
                            if factor != 5:
                                third_msg = self.expected_log_likelihood_einsum(self.observations[trial, tau, :], factor, policy_idx, tau)
                            
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
                                transposed_B = self.transpose_Bfa(self.B[factor][:, :, action_tau[factor]])
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
                                first_msg = self.log_stable(self.B[factor][:, :, action_tau[factor]].dot(qs_prev))
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
                                first_msg = self.log_stable(self.B[factor][:, :, action_tau[factor]].dot(qs_prev))

                                # Second message
                                #if not np.isnan(self.observations[trial, tau, 4]):
                                    #obs_mod = int(self.observations[trial, tau, 4])
                                    #qs_future = self.one_hot_encode(4, int(obs_mod), self.obs_dim)
                                #else:
                                    #qs_future = self.policy_dep_posteriors[policy_idx, tau+1, factor]
                                qs_future = self.policy_dep_posteriors[policy_idx, tau+1, factor+1]
                                transposed_B = self.transpose_Bfa(self.B[factor][:, :, action_tau[factor]])
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
                        self.F_policy[t][policy_idx] = previous_F
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
    
    def infer_policies_multithread(self, trial, t):
        num_policies = len(self.policies)
        with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = [executor.submit(self._eval_policy, t, idx) for idx in range(num_policies)]

            for future in concurrent.futures.as_completed(futures):
                try:
                    idx, g_val = future.result()
                    self.G_policy[t][idx] += g_val
                except Exception as exc:
                    print(f'Policy evaluation generated an exception: {exc}', file=sys.stderr)
        self.update_policy_posterior(trial, t)
        return copy.deepcopy(self.G_policy), copy.deepcopy(self.F_policy)

    def infer_policies__multithread_old(self, trial, t):
        num_policies = len(self.policies)
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self._eval_policy, t, idx) for idx in range(num_policies)]
            for future in futures:
                idx, g_val = future.result()
                self.G_policy[t][idx] += g_val

        self.update_policy_posterior(trial, t)
        return copy.deepcopy(self.G_policy), copy.deepcopy(self.F_policy)
    
    def infer_policies_true_parallel(self, trial, t):
        num_policies = self.num_policies
        print(f"Starting parallel inference for {num_policies} policies at t={t} in main process {os.getpid()}...")

        # Prepare arguments for the worker function (WITHOUT shm_info)
        worker_args_list = [
            (
                t,
                idx,
                self.temporal_horizon,
                self.num_factors,
                self.num_modalities,
                self.learning_D,
                self.learning_A,
                self.learning_B
            )
            for idx in range(num_policies)
        ]

        # --- Execute parallel computation with initializer ---
        with concurrent.futures.ProcessPoolExecutor(
            initializer=_worker_initializer,
            initargs=(
                self.A_shm_info, self.C_shm_info,
                self.P_shm_info
            )
        ) as executor:
            results_iterator = executor.map(full_eval_policy_worker, worker_args_list)

            # Collect results
            for idx, g_val in results_iterator:
                self.G_policy[t][idx] += g_val

        print(f"Finished parallel inference at t={t} in main process {os.getpid()}.")
        self.update_policy_posterior(trial, t)
        self._cleanup_shared_memory() # Main process still unlinks shared memory segments
        return copy.deepcopy(self.G_policy), copy.deepcopy(self.F_policy)

    
    def infer_policies(self, trial, t):
        risk = []
        for policy_idx in range(len(self.policies)):
            info_gain_tot = 0

            #epistemic value term (Bayesian surprise)
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
            risk.append(risk_term)
            self.G_policy[t][policy_idx] += risk_term + ambiguity_term -info_gain_tot

        self.update_policy_posterior(trial, t)
        return self.G_policy, self.F_policy

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
        # @NOTE according to the MATLAB code, pD info gain do not
        # depend on the time step of the policy. Therefore, we can
        # calculate it only when t==0, for the very first time step.
        for factor_idx in range(self.num_factors):
            wD_factor = self.pD_complexity[factor_idx]
            #expected_sts = self.D[factor_idx].dot(self.policy_dep_posteriors[policy_idx, 0, factor_idx])
            expected_sts_pD = wD_factor.dot(self.policy_dep_posteriors[policy_idx, 0, factor_idx])
            wD_term_policy += expected_sts_pD
            #wD_term_policy += expected_sts*expected_sts_pD
        return wD_term_policy

    def calculate_pB_info_gain(self, t, policy_idx):
        wB_term_policy = 0
        policy = self.policies[policy_idx]
        for timestep in range(t, self.temporal_horizon):
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

    def calculate_pB_info_gain_vectorized(self, t, policy_idx):
        T = self.temporal_horizon - 1 - t
        if T <= 0:
            return 0.0  # no timesteps to process
        F = self.num_factors

        policy_actions = self.policies[policy_idx][t:t+T]  # shape [T, F]
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
   
    
    def calculate_pA_info_gain(self, t, policy_idx):
        wA_term_policy = 0
        for timestep in range(t, self.temporal_horizon):
            for modality_idx, modality in enumerate(self.pA):
                wA_mod = self.pA_complexity[modality_idx]
                expected_obs = self.cell_md_dot_py(self.A[modality_idx], self.policy_dep_posteriors[policy_idx, timestep, :]) 
                expected_obs_pA = self.cell_md_dot_py(wA_mod, self.policy_dep_posteriors[policy_idx, timestep, :])
                wA_term_policy += expected_obs.dot(expected_obs_pA)
        return wA_term_policy

    def calculate_policy_ambiguity(self, t, policy_idx):
        """
        Calculates policy ambiguity using factorized posteriors to avoid iterating
        over the joint state space.
        """
        ambiguity = 0.0
        #num_factors = len(self.num_states)

        for timestep in range(t, self.temporal_horizon):
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
        
    def calculate_policy_risk(self, t, policy_idx):
        risk_term_policy = 0
        #risk_term_policy_old = 0
        for timestep in range(t, self.temporal_horizon):
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
                expected_obs = self.cell_md_dot_py(modality, self.policy_dep_posteriors[policy_idx, timestep, :])
                #KL_modality = self.log_stable(expected_obs) - self.C[modality_idx][:, t]
                risk_term_policy += expected_obs.dot(self.C[modality_idx][:, timestep])
        return risk_term_policy
    
    def cell_md_dot_py(self, X, x):
        p = X.copy()
        for f in reversed(range(len(x))):
            p = np.tensordot(p, x[f], axes=(f + 1, 0))
        return p
    
    def cell_md_dot(self, X, x):
        # Initialize the dimensions to sum over
        temp_list = np.array(range(len(x)))
        ones_list = np.ones(len(temp_list))
        DIM = temp_list + ones_list*X.ndim - ones_list*len(x)
        s_ini = np.ones(X.ndim, dtype=int)
        for d in range(len(x)):
            # Create a shape for the current element of x
            s = copy.deepcopy(s_ini)
            s[int(DIM[d])] = x[d].shape[0]  # Set the corresponding dimension size
            reshaped_x = np.reshape(x[d], s) 
            # Perform element-wise multiplication (broadcasting)
            X = X * reshaped_x

            # Sum over the appropriate dimension
            X = np.sum(X, axis=int(DIM[d]))

        # Remove singleton dimensions
        X = np.squeeze(X)
        
        return X
            
    def choose_action(self, trial, t):

        if t < self.temporal_horizon-1:
            #self.alpha = 0.1 * np.exp(0.05 * trial)
            if self.action_selection == "deterministic":
                policy_idx = np.argmax(self.posterior_pi[t])
                for factor_idx in self.controlable_states:
                    self.action_posteriors[factor_idx, t] = self.policies[policy_idx][t, factor_idx]

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
                    policy_t_action = policy[t]
                    for factor_idx in self.controlable_states:
                            fac_action = policy_t_action[factor_idx]
                            action_list[factor_idx][fac_action] += self.posterior_pi[t][policy_idx]

                for factor_idx in self.controlable_states:
                    action_list[factor_idx] = self.softmax(self.log_stable(action_list[factor_idx]), axis=None, gamma = self.alpha)
                    self.action_posteriors[factor_idx, t] = np.searchsorted(np.cumsum(action_list[factor_idx]), np.random.rand())

            elif self.action_selection == "random":
                action_prob = {}
                # Initialize action_list properly
                for idx, i in enumerate(self.controls_dim):  # Iterate with index
                    if i == 1:
                        continue  # Skip if control dimension is 1
                    else:
                        action_prob[idx] = np.zeros(i)
                
                for factor_idx in self.controlable_states:
                    self.action_posteriors[factor_idx, t] = random.choice(range(self.controls_dim[factor_idx]))
                    action_prob[factor_idx] = 1
            
            elif self.action_selection == "stochastic":
                log_posterior_pi = self.log_stable(self.posterior_pi[t])
                p_policies = self.softmax(log_posterior_pi * self.alpha) 
                policy_idx = self.sample(p_policies)
                for factor_idx in self.controlable_states:
                    self.action_posteriors[factor_idx, t] = self.policies[policy_idx][0, factor_idx]

            return self.action_posteriors[:, t], action_list
        else:
            return None, None
        
    import numpy as np

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

        
    def perform_learning(self, trial):
        
        
        F_stacked = np.vstack(-copy.deepcopy(self.F_policy))
        temporal_horizon = F_stacked.shape[0]  # or set manually

        min_values = [F_stacked[t, :].min() for t in range(temporal_horizon)]
        self.learning_rate = np.mean(min_values)*0.1
        self.learning_rate = np.clip(self.learning_rate, 0.1, 10)

        self.forgeting_rate = self.previous_lr*0.1
        self.forgeting_rate = 1-np.clip(self.forgeting_rate, 0.01, 0.5)
        
        if self.learning_A:
            for t in range(self.temporal_horizon):
                #self.learning_rate = np.min(-np.vstack(self.F_policy)[t,:])
                for modality_idx in range(len(self.pA)):
                    obs_mod = int(self.observations[trial, t, modality_idx])
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
            for modality_idx in range(len(self.pA)):
                self.Fa[modality_idx] += self.KL_dirichlet(self.pA[modality_idx], self.pA_prior[modality_idx])

        if self.learning_D:
            #self.learning_rate = np.min(-np.vstack(self.F_policy)[self.temporal_horizon -1,:])
            for factor_idx in range(self.num_factors):
                i = self.pD[factor_idx] > 0
                self.pD[factor_idx] = np.where(
                    i,
                    (self.pD[factor_idx] - self.pD_0[factor_idx]) 
                    + self.pD_0[factor_idx] 
                    + self.learning_rate * self.bayesian_mod_avg[trial, self.temporal_horizon -1, factor_idx], #self.temporal_horizon -1
                    self.pD[factor_idx]
                )                
                
                # free energy of d
                self.Fd[factor_idx] = self.KL_dirichlet(self.pD[factor_idx], self.pD_prior[factor_idx])
                del i

        if self.learning_B:
            
            for t in range(self.temporal_horizon):
                #self.learning_rate = np.min(-np.vstack(self.F_policy)[t,:])
                if t > 0:
                    
                    for factor_idx in range(self.num_factors):
                        if factor_idx not in self.controlable_states:
                            continue
                        action = int(self.action_posteriors[factor_idx, t-1])
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
                
            # free energy of b
            for factor_idx in range(self.num_factors):
                self.Fb[factor_idx] = self.KL_dirichlet(self.pB[factor_idx], self.pB_prior[factor_idx])    

        if self.learning_E:
            self.pE = self.forgeting_rate*(self.pE - self.pE_0) + self.pE_0 + self.learning_rate*self.posterior_pi[self.temporal_horizon-1]
            # negative free energy of e
            self.Fe = self.KL_dirichlet(self.pE, self.E)
        self.previous_lr = copy.deepcopy(self.learning_rate)
        return self.Fa, self.Fb, self.Fd, self.Fe, self.learning_rate, self.forgeting_rate

    
    def softmax(self, x, axis = 0, gamma=1.0):
        exp_x = np.exp(gamma * x - np.max(gamma * x))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    def softmax_whole(self, x, gamma=1.0):
        x_copy = copy.deepcopy(x)
        for i in range(len(x_copy)):
            x_copy[i] = self.softmax(x_copy[i], gamma=gamma)
        return x_copy
    
    def perform_modal_average(self, trial, t):
        # Following function averages the posterior of states over policies.
        # by executing following function, we update the self.bayesian_mod_avg
        # which stores the posterior over states for each time step of the previous trial
        # average over all policies
        qs_temp = copy.deepcopy(self.policy_dep_posteriors)
        for factor_idx in range(self.num_factors):
            for tau in range(self.temporal_horizon):
                v_stack_states = np.vstack(qs_temp[:,tau,factor_idx])
                self.bayesian_mod_avg[trial, tau, factor_idx] = v_stack_states.T.dot(self.posterior_pi[t][:])

        del qs_temp
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
        if t > 0:
            self.gamma[t] = self.gamma[t-1]
        #psi = 2;                     # Step size parameter (promotes stable convergence)
        previous_beta_posterior = None
        tolerance = 1e-50  # Convergence threshold
        for nmp in range(self.number_of_msg_passing):  # Number of gradient descent iterations
            # posterior and prior over policies
            self.prior_pi[t][:] = self.softmax(self.log_stable(self.E) + self.gamma[t]*self.G_policy[t][:], axis=None)
            self.posterior_pi[t][:] = self.softmax(self.log_stable(self.E) + self.gamma[t]*self.G_policy[t][:] + self.F_policy[t][:], axis=None)
            
            # expected free energy precision (beta)
            G_err = (self.posterior_pi[t][:] - self.prior_pi[t][:]).dot(self.G_policy[t][:])
            beta_update = self.beta_posterior - self.beta_prior + G_err
            self.beta_posterior = self.beta_posterior - beta_update/2
            self.gamma[t] = 1/self.beta_posterior
            
            """
            # simulate dopamine responses
            n = t*self.number_of_msg_passing + nmp
            self.gamma_update[trial, n] = self.gamma[t]
            self.posterior_updates[n][:] = copy.deepcopy(self.posterior_pi[t][:])
            """
            # Early stopping condition to exit gradient descent if minimum VFE reached!
            if nmp > 0 and previous_beta_posterior is not None:
                if abs(self.beta_posterior - previous_beta_posterior) < tolerance:
                    break
            previous_beta_posterior = copy.deepcopy(self.beta_posterior)
        del previous_beta_posterior
 
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
    
    def expected_log_likelihood_einsum(self, obs, factor, policy_idx, tau):
        """
        Calculates the expected log-likelihood for a factor using np.einsum.
        This is more efficient as it avoids creating intermediate arrays.
        """
        # Initialize with zeros for the states of the target factor
        log_likelihoods = np.zeros(self.states_dim[factor])
        
        # Pre-fetch the posteriors that will be used for marginalization
        posteriors_to_marginalize = [
            self.policy_dep_posteriors[policy_idx, tau, f] 
            for f in range(self.num_factors) if f != factor
        ]

        if obs is not None:
            for modal_idx, modality in enumerate(self.A):
                # Get the log-likelihood slice for the current observation
                # This is a tensor with dimensions for each state factor
                lnA = self.log_stable(np.take(modality, obs[modal_idx], axis=0))

                # Dynamically create the einsum string
                # e.g., for 3 factors and target factor 0: 'ijk,j,k->i'
                alphabet = string.ascii_lowercase
                all_factors_str = alphabet[:self.num_factors]
                # This part needs to list each individual posterior's dimension
                # For 'b,c,d,e,f' each letter is a separate operand in the string
                other_factors_dims = [alphabet[f] for f in range(self.num_factors) if f != factor]
                
                if other_factors_dims: # Check if there are other factors to marginalize
                    other_factors_str = ",".join(other_factors_dims)
                    einsum_str = f'{all_factors_str},{other_factors_str}->{alphabet[factor]}'
                else: # No other factors to marginalize (e.g., when num_factors is 1)
                    einsum_str = f'{all_factors_str}->{alphabet[factor]}'
                
                # Perform the entire marginalization in one step
                expected_lnA = np.einsum(einsum_str, lnA, *posteriors_to_marginalize)
                
                log_likelihoods += expected_lnA
                
        return log_likelihoods
            
    def transpose_Bfa(self, B_fa):
        B_T = copy.deepcopy(B_fa)
        
        B_T = np.transpose(B_T, (1, 0))  # Transpose state dimensions, keep actions
        B_T = np.divide(B_T, B_T.sum(axis=0))
        B_T = np.nan_to_num(B_T, nan=0.0)  # Replace NaNs with zero
        
        return B_T

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
        return np.log(np.clip(array, eps, 1.0))
    
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
        # Create a zero vector of the required size for the given modality
        one_hot_obs = np.zeros(obs_dims[obs_modality], dtype=int)
        
        # Set the index corresponding to the observation value to 1
        one_hot_obs[obs_value] = 1
        
        return one_hot_obs.tolist()  # Convert to list if needed
    
    def wnorm_new(self, p, val=np.exp(-16)):
        # @NOTE here the equation (40) is implimented
        # compared to the MATLAB code.
        # according to the equation (40); w = 0.5*(avg - norm)
        p_temp = copy.deepcopy(p)
        p_temp = p + val
        norm = np.divide(1.0, np.sum(p_temp, axis=0))
        avg = np.divide(1.0, p_temp)
        del p_temp
        return 0.5*(norm - avg)
    
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
        # Compute KL divergence
        p_copy = copy.deepcopy(p)
        q_copy = copy.deepcopy(q)
        d = self.log_beta(p_copy) - self.log_beta(q_copy) - np.sum((p_copy - q_copy) * spm_psi(q_copy + 1/32), axis=0)
        del p_copy, q_copy
        return np.sum(d)  # Sum over all columns
        
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
        if z.ndim == 1:  # Vector case
            z = z[z > 0]  # Remove zeros
            return np.sum(gammaln(z)) - gammaln(np.sum(z))

        else:  # Multi-dimensional case
            shape = z.shape[1:]  # Exclude the first dimension
            y = np.zeros(shape)  # Initialize output
            
            # Iterate over all dimensions beyond the first
            it = np.ndindex(shape)
            for idx in it:
                y[idx] = self.log_beta(z[(slice(None),) + idx])  # Recursive computation
            
            return y
        
def spm_psi(A):
    # This is the python implimentation of the Karl Friston's
    # spm_psi MATLAB function
    # Copyright (C) 2015 Wellcome Trust Centre for Neuroimaging
    # by Karl Friston
    return psi(A) - psi(np.sum(A, axis=0, keepdims=True))
