import numpy as np
import itertools

def obj_array(num_arr):
    """
    Creates a NumPy object array with the desired number of sub-arrays
    """
    return np.empty(num_arr, dtype=object)

def uniform_A_matrix(num_obs, num_states):
    """
    Creates a uniform A matrix (likelihood) for active inference.
    
    Args:
        num_obs (int or list[int]): Observations dimentions  
        num_states (int or list[int]): Hidden states dimentions
    
    Returns:
        np.ndarray: NumPy object array of A matrices, one per modality
    """
    if isinstance(num_obs, int):
        num_obs = [num_obs]
    if isinstance(num_states, int):
        num_states = [num_states]

    num_modalities = len(num_obs)
    A = obj_array(num_modalities)

    for m, modality_obs in enumerate(num_obs):
        shape = (modality_obs, *num_states)
        A[m] = np.full(shape, 1.0 / modality_obs, dtype=np.float32)

    return A

def uniform_B_matrix(num_states, num_controls):
    """
    Creates a uniform B matrix (transition probabilities).

    Args:
        num_states (int or list[int]): Hidden states dimentions
        num_controls (int or list[int]): Number of actions per hidden state factor

    Returns:
        np.ndarray: NumPy object array of B matrices, one per hidden state factor
    """
    if isinstance(num_states, int):
        num_states = [num_states]
    if isinstance(num_controls, int):
        num_controls = [num_controls]

    num_factors = len(num_states)
    assert len(num_controls) == num_factors, "num_controls must match num_states length"

    B = obj_array(num_factors)

    for f in range(num_factors):
        shape = (num_states[f], num_states[f], num_controls[f])
        B[f] = np.full(shape, 1.0 / num_states[f], dtype=np.float32)

    return B

def uniform_D_matrix(shape_list):
    """
    Creates a NumPy object array whose sub-arrays are uniform categorical
    distributions with shapes given by num_states[i].

    Args:
        num_states (int or list[int]): Hidden states dimentions

    Returns:
        np.ndarray: Object array of uniform distributions
    """
    arr = obj_array(len(shape_list))

    for i, shape in enumerate(shape_list):
        if isinstance(shape, int):
            shape = (shape,)
        else:
            shape = tuple(shape)
        arr[i] = np.full(shape, 1.0 / shape[0], dtype=np.float32)

    return arr

def zero_C_matrix(shape_list, temp_horizon):
    """
    Creates a NumPy object array whose sub-arrays are zero-initialized
    distributions with an added time horizon dimension.

    Args:
        shape_list (list[int or tuple or list]): Shapes of sub-arrays
        temp_horizon (int): Number of time steps (extra dimension)

    Returns:
        np.ndarray: Object array of zeros distributions
    """
    arr = obj_array(len(shape_list))

    for i, shape in enumerate(shape_list):
        if isinstance(shape, int):
            shape = (shape,)
        else:
            shape = tuple(shape)
        
        shape_with_horizon = shape + (temp_horizon,)

        arr[i] = np.zeros(shape_with_horizon, dtype=np.float32)

    return arr

def construct_policies(num_states, num_controls=None, policy_len=1, control_fac_idx=None):
    """
    Generate all possible policies over a planning horizon.

    Parameters
    ----------
    num_states : list[int]
        Dimensionalities of each hidden state factor
    num_controls : list[int], optional
        Dimensionalities of control factors. If None, inferred from controllable factors
    policy_len : int, default 1
        Temporal depth of policies
    control_fac_idx : list[int], optional
        Indices of controllable hidden state factors

    Returns
    -------
    policies : list[np.ndarray]
        Each policy is a 2D array of shape (policy_len, num_factors)
    """
    num_factors = len(num_states)

    if control_fac_idx is None:
        if num_controls is not None:
            control_fac_idx = [f for f, n_c in enumerate(num_controls) if n_c > 1]
        else:
            control_fac_idx = list(range(num_factors))

    if num_controls is None:
        num_controls = [num_states[f] if f in control_fac_idx else 1 for f in range(num_factors)]

    per_factor_choices = [list(range(n_c)) for n_c in num_controls]
    per_timestep_choices = per_factor_choices * policy_len 

    policies = [np.array(p).reshape(policy_len, num_factors) 
                for p in itertools.product(*per_timestep_choices)]

    return policies




