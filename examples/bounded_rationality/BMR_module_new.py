import numpy as np
import copy
from PyAIF import utils, ActiveInfAgent
from environment import GridEnvironment
import matplotlib.pyplot as plt
import time

class BMRModule:
    def __init__(self, num_states, num_obs, num_controls, temp_horizon, controllable_factors,  controllable_modalities, A, B, C, D, minimum_dim = 5, E=None):

        self.master_A = copy.deepcopy(A)
        self.master_B = copy.deepcopy(B)
        self.master_C = copy.deepcopy(C)
        self.master_D = copy.deepcopy(D)
        if E is not None:
            self.master_E = copy.deepcopy(E)

        self.m_states_dim = num_states
        self.m_obs_dim = num_obs
        self.m_controls_dim = num_controls
        self.Temp_horizon = temp_horizon
        self.minimum_dim = minimum_dim
        self.master_dim = self.m_states_dim[0]
        self.control_factors = controllable_factors
        self.control_mod = controllable_modalities

    def _reduce_A_matrix(self, target_dim, n_states, n_obs):
        # Check cache first!
        #if target_dim in self.cache_A:
            #return self.cache_A[target_dim]

        A = utils.zeros_A_matrix(n_obs, n_states)
        # Only get Pi for changeable factors
        for i, T in enumerate(self.master_A):
            res = T.copy()
            
            # 1. Reduce Observations (Axis 0)
            #if self.control_mod[i] < 0:
                #pi = self._generate_pi(target_dim)
                #res = np.tensordot(pi, res, axes=([1], [0]))
                
            # 2. Reduce Factors (Axes 1, 2, 3...)
            for f_idx, ctrl in enumerate(self.control_factors):
                if ctrl < 0:
                    pi = self._generate_pi(target_dim)
                    # Contract along the specific factor axis
                    res = np.tensordot(res, pi, axes=([f_idx + 1], [1]))
                    # tensordot might mess with axis order; move it back
                    res = np.moveaxis(res, -1, f_idx + 1)

            A[i] = res
        
        #self.cache_A[target_dim] = new_A
        return A
    
    def _reduce_B_matrix(self, target_dim, n_states):
        # n_states is the new shape list, e.g., [5, 5, 50]
        B = utils.zeros_B_matrix(n_states, self.m_controls_dim)
        """
        # Get the mapping matrix Pi for the target dimension
        pi = self._generate_pi(target_dim)

        for f in range(len(self.master_B)):
            if self.control_factors[f] < 0:
                # This factor is changeable (e.g., Grid Position)
                for a in range(self.m_controls_dim[f]):
                    master_slice = self.master_B[f][:, :, a]
                    
                    # BMR formula for transitions: Pi @ Master @ Pi.T
                    # 1. Pi @ master_slice -> collapses rows (Target States)
                    # 2. (Result) @ pi.T   -> collapses columns (Source States)
                    B[f][:,:,a] = pi @ master_slice @ pi.T
            else:
                # This factor is fixed
                # We just copy the Master B-matrix for this factor
                B[f] = self.master_B[f].copy()

        """
        
                
        return B
    
    def _reduce_C_matrix(self, target_dim, n_obs):
        C = utils.zero_C_matrix(n_obs, self.Temp_horizon)
        pi = self._generate_pi(target_dim)

        for m in range(len(self.master_C)):
            if self.control_mod[m] < 0:
                # master_C[m] shape: (50, horizon)
                # pi shape: (5, 50)
                
                # This reduces the 50 observations to 5 for every time step simultaneously
                C[m] = pi @ self.master_C[m]
            else:
                C[m] = self.master_C[m].copy()
                
        return C
    
    def _reduce_D_matrix(self, target_dim, n_states):
        """
        n_states: List of the new dimensions, e.g., [5, 5, 50]
        """
        # Initialize the new D container (list of vectors)
        D = utils.uniform_D_matrix(n_states)
        pi = self._generate_pi(target_dim)

        for f in range(len(self.master_D)):
            if self.control_factors[f] < 0:
                # Factor is changeable: Reduce from 50 to 5
                # pi (5, 50) @ master_D (50,) -> (5,)
                D[f] = pi @ self.master_D[f]
                
            else:
                # Factor is fixed: Keep the master belief as is
                D[f] = self.master_D[f].copy()
                
        return D

    def decrease_resolution(self, curren_dim):
        """
        curren_dim: int 

        #@NOTE that target dimention is related only to the 
        # controllable factors and modalities

        """
        # Map 0-9 scale to a target dimension (e.g., 5, 10, 15... 50)
        # In this example, a resolution is define by both the factors and modalities
        # which are controllable.
        # for example target dimention is 5, it means both factors and modalities
        # which are controllable has dimention of 5.
        current_res = int(curren_dim/self.minimum_dim - 1)
        target_dim = max((current_res - 1) * 5 + 5, self.minimum_dim)


        # New states and obs modality dimentions 
        n_states = [target_dim if x < 0 else x for x in self.control_factors]
        n_obs =  self.m_obs_dim

        A = self._reduce_A_matrix(target_dim, n_states, n_obs)
        B = self._reduce_B_matrix(target_dim, n_states)
        C = self._reduce_C_matrix(target_dim, n_obs)
        D = self._reduce_D_matrix(target_dim, n_states)
        return A, B, D, n_states, n_obs
    
    def increase_resolution(self, curren_dim):
        """
        curren_dim: int 

        #@NOTE that target dimention is related only to the 
        # controllable factors and modalities

        """
        # Map 0-9 scale to a target dimension (e.g., 5, 10, 15... 50)
        # In this example, a resolution is define by both the factors and modalities
        # which are controllable.
        # for example target dimention is 5, it means both factors and modalities
        # which are controllable has dimention of 5.
        current_res = int(curren_dim/self.minimum_dim - 1)
        target_dim = min((current_res + 1) * 5 + 5, self.master_dim)

        # New states and obs modality dimentions 
        n_states = [target_dim if x < 0 else x for x in self.control_factors]
        n_obs = [target_dim if x < 0 else x for x in self.control_mod]

        A = self._reduce_A_matrix(target_dim, n_states, n_obs)
        B = self._reduce_B_matrix(target_dim, n_states)
        #C = self._reduce_C_matrix(target_dim, n_obs)
        D = self._reduce_D_matrix(target_dim, n_states)
        return A, B, D

    def _generate_pi(self, target_dim):
        pi = np.zeros((target_dim, self.master_dim))

        edges = np.linspace(0, self.master_dim, target_dim + 1)

        for i in range(target_dim):
            start = edges[i]
            end = edges[i + 1]

            # Find the original state indices that overlap with this target state
            left = int(np.floor(start))
            right = int(np.ceil(end))

            for j in range(left, right):
                # Overlap fraction of original state j
                overlap = min(end, j + 1) - max(start, j)
                pi[i, j] = overlap / (end - start)

        return pi




