from time import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp

class LikelihoodModels:
    def __init__(self, model_name, states_dim=None, obstacles_dic=None, obs_limits=None):
        self.states_dim = states_dim
        self.obstacles_dic = obstacles_dic
        self.obs_limits = obs_limits
        if model_name == "task":
            self.model = TaskLikelihoodModel(states_dim, obstacles_dic, obs_limits)
        elif model_name == "meta":
            df = pd.read_csv("resolution_signatures.csv")
            modalities = ['max_risk', 'max_ambiguity', 'inference_time_ms']
            signatures_df = df.groupby('resolution')[modalities].agg(['mean', 'std'])
            self.model = MetaLikelihoodModel(states_dim, signatures_df)

class MetaLikelihoodModel:
    def __init__(self, states_dim, signatures_df):
        self.states_dim = states_dim
        self.resolutions = sorted(signatures_df.index.unique())
        self.cpu_levels = np.arange(self.states_dim[1])  # Assuming the second factor's states represent CPU levels
        self.eps=1e-16 # small constant for numerical stability in log calculations
        # profiled signatures for the "Base" (Low CPU)
        self.base_signatures = self._parse_signatures(signatures_df)
        

    def log_likelihoods(self, obs_val, modality_idx):

        if modality_idx == 0 or modality_idx == 1: # Risk or Ambiguity
            lnA_grid = np.zeros(self.states_dim[0])
            for i, res in enumerate(self.resolutions):
                mu, sigma = self.base_signatures[res][modality_idx]
                
                diff = -0.5 * ((obs_val - mu) / sigma + self.eps) ** 2
                norm = -np.log(sigma * np.sqrt(2 * np.pi))
                lnA_grid[i] = diff + norm
            return lnA_grid

        elif modality_idx == 2: # Inference Time
            # Initialize grid: Rows = Resolutions, Cols = CPU Levels
            lnA_grid = np.zeros([self.states_dim[0], self.states_dim[1]])
            for i, res in enumerate(self.resolutions):
                for j, cpu_mult in enumerate(self.cpu_levels):
                    base_mu, base_sigma = self.base_signatures[res][modality_idx]
                    
                    # Adjust expectation based on CPU multiplier
                    current_mu = base_mu * cpu_mult
                    current_sigma = base_sigma * cpu_mult + self.eps # Uncertainty also scales
                    
                    diff = -0.5 * ((obs_val - current_mu) / current_sigma) ** 2
                    norm = -np.log(current_sigma * np.sqrt(2 * np.pi))
                    
                    lnA_grid[i, j] = diff + norm

            return lnA_grid
    
    def _parse_signatures(self, signatures_df):
        parsed = {}
        # Iterate through the unique resolutions in your CSV
        for res in signatures_df.index.get_level_values(0).unique():
            parsed[res] = {
                0: (signatures_df.loc[res, ('max_risk', 'mean')], 
                    signatures_df.loc[res, ('max_risk', 'std')] + self.eps),
                1: (signatures_df.loc[res, ('max_ambiguity', 'mean')], 
                    signatures_df.loc[res, ('max_ambiguity', 'std')] + self.eps),
                2: (signatures_df.loc[res, ('inference_time_ms', 'mean')], 
                    signatures_df.loc[res, ('inference_time_ms', 'std')] + self.eps)
            }
        return parsed

class TaskLikelihoodModel:
    def __init__(self, states_dim, obstacles_dict, obs_limits, sigma_x=10.0, sigma_y=10.0, sigma_s=2.0, alpha=0.01):
        """
        Initialize the likelihood model.
        
        Parameters:
        - states_dim: list of number of states for each factor [x_curr, y_curr, x_goal, y_goal]
        - sigma_x, sigma_y, sigma_s: observation noise for each modality
        - alpha: decay rate for signal
        - RSI: maximum signal value at the goal
        """
        
        self.states_dim = states_dim
        self.obstacles_dict = obstacles_dict
        # 1. Scale Sigmas for CM
        # 0.5 was too small for a 500cm map. 10.0 cm is a more realistic GPS error.
        self.sigma_x = sigma_x 
        self.sigma_y = sigma_y
        self.sigma_s = sigma_s # Signal noise stays relative to RSI (0-30)
        
        # 2. Update physical boundaries
        self.x_max = obs_limits['x_obs'][1]
        self.y_max = obs_limits['y_obs'][1]
        self.x_min = obs_limits['x_obs'][0]
        self.y_min = obs_limits['y_obs'][0]
        
        # 3. Calculate cell size (Scale Factor)
        self.cm_per_x_cell = (self.x_max - self.x_min) / states_dim[0] # e.g., 25.0
        self.cm_per_y_cell = (self.y_max - self.y_min) / states_dim[1] # e.g., 25.0
        
        self.alpha = alpha 
        self.RSI = obs_limits['rsi_obs'][1]
        self.rsi_min = obs_limits['rsi_obs'][0]
        # ... 

        self.pref_dep = [(0, 1)] # indices of obs modalities that have joint preferences (e.g., x-y)
        self.negative_pref = -50  # strong negative preference for obstacles
        self.eps=1e-8 # small constant for numerical stability in log calculations

        self.log_preferences = self._build_preferences()
        #self._plot_joint_preferences(self.log_preferences)
        self._precompute_signal_mean() 
        #self._plot_signal_expectation_map((4, 8))
        self._plot_signal_preferences(self.log_preferences)
        #self._plot_signal_preferences(self.log_preferences)
        
    
    def _convert_to_log_pref(self, pref_dic):
        log_pref_dic = {}
        for key, C in pref_dic.items():
            # Shift so the 'best' state is 0, others are negative
            log_pref_dic[key] = C - np.max(C) 
        return log_pref_dic
    
    def _softmax(self, x, axis = 0, gamma=1.0):
        exp_x = np.exp(gamma * x - np.max(gamma * x))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    
    def _build_preferences(self, goal_pos=None, sigma_goal=50.0, sigma_signal=10, scale = 1.0):
        """
        Generate preferences for each modality or joint modality.
        
        Parameters
        ----------
        o_grids : dict
            Observation grids for each modality, e.g., {0: x_grid, 1: y_grid, 2: signal_grid}
        goal_pos : tuple
            (x_goal, y_goal) position
        obstacles_dict : dict
            Keys = modality index or joint tuple, values = list of blocks (x_min, y_min, x_max, y_max)
        RSI : float
            Maximum signal strength for signal modality
        sigma_goal : float
            Width of goal preference Gaussian
        sigma_signal : float
            Width of signal preference Gaussian
        joint_mods : list of tuples
            List of modality indices that have joint preferences, e.g., [(0,1)]
        
        Returns
        -------
        preferences_dict : dict
            Keys = modality index or joint tuple, values = preference arrays over grids
        """
        preferences_dict = {}
        grids = [self.get_o_grid(m) for m in (0,1,2)]

        # --- Joint preferences (x-y) ---
        if self.pref_dep is not None:
            for joint in self.pref_dep:
                X, Y = np.meshgrid(grids[0], grids[1], indexing='ij')
                
                # Start with neutral preference (0 in log space)
                C = np.zeros_like(X) + -0.01 # small negative baseline to avoid ties

                # Goal preference (Log-Gaussian)
                if goal_pos is not None:
                    x_goal, y_goal = goal_pos
                    # Direct log-space calculation
                    C += -0.5 * ((X - x_goal)/sigma_goal)**2
                    C += -0.5 * ((Y - y_goal)/sigma_goal)**2

                # Obstacles (Apply negative penalty)
                if self.obstacles_dict is not None:
                    # Assuming obstacles are defined in CM coordinates (0-500)
                    for block_key in self.obstacles_dict.keys():
                        x_min, x_max, y_min, y_max = self.obstacles_dict[block_key]
                        mask = (X >= x_min) & (X <= x_max) & (Y >= y_min) & (Y <= y_max)
                        C[mask] = self.negative_pref

                #C_probs = self._softmax(C.flatten(), gamma=1.0).reshape(C.shape)
                preferences_dict[joint] = C

        # --- Single-modality preferences ---
        if 2 not in preferences_dict: # Signal Modality
            o_grid = np.linspace(0, 30, 100)
            def plateau_pref(x, center=18, steepness=0.8):
                # This creates a "S" curve that is very flat at the ends
                return 1 / (1 + np.exp(-steepness * (x - center)))

            # Apply a small offset so log(0) doesn't break the code
            # Multiply by a "Motivation" factor (e.g., 2.0) to control the depth
            C_signal = 0.01 * np.log(plateau_pref(o_grid) + 0.01)
            
            preferences_dict[2] = C_signal #np.log(C_sig_probs + self.eps)

        return preferences_dict
    
    def _plot_signal_expectation_map(self, curr_pos=(4, 18)):
        curr_x_idx, curr_y_idx = curr_pos

        # 1. Extract the slice
        # mu_signal[x_idx, y_idx, :, :] gives RSI expectations for all goals
        # indexing='ij' ensures that the first colon is Goal X and the second is Goal Y
        # However, imshow expects (rows, columns) which is (Y, X)
        # So we transpose the slice for correct visualization: .T
        goal_expectation_map = self.mu_signal[curr_x_idx, curr_y_idx, :, :].T

        plt.figure(figsize=(8, 6))
        
        # 2. Use 'extent' to map the 0-20 indices to 0-500 cm on the plot axes
        plt.imshow(goal_expectation_map, 
                origin='lower', 
                cmap='viridis',
                extent=[self.x_min, self.x_max, self.y_min, self.y_max])
        
        plt.colorbar(label='Expected RSI')
        plt.title(f"Expected Signal Map (Agent at Index {curr_pos})")
        plt.xlabel("Goal X (cm)")
        plt.ylabel("Goal Y (cm)")

        # 3. Mark the TRUE router position in PHYSICAL coordinates
        # If the router is at index (11, 1), its physical cm is:
        x_scale = (self.x_max - self.x_min) / self.states_dim[0]
        y_scale = (self.y_max - self.y_min) / self.states_dim[1]
        true_router_x_cm = (10 + 0.5) * x_scale
        true_router_y_cm = (1 + 0.5) * y_scale
        
        plt.scatter(true_router_x_cm, true_router_y_cm, 
                    color='red', marker='x', s=100, label='True Router (cm)')
        
        plt.legend()
        plt.show()
    
    def _plot_joint_preferences(self, preferences_dict, joint=(0,1)):
        """
        Visualize joint preference map as a 2D heatmap.

        Parameters
        ----------
        o_grids : dict
            Observation grids, e.g., {0: x_grid, 1: y_grid}
        preferences_dict : dict
            Dictionary of preferences returned from get_preferences()
        joint : tuple
            Modalities that are jointly dependent (default (0,1) for x-y)
        """
        o_grids = {m: self.get_o_grid(m) for m in (0,1)}
        X, Y = np.meshgrid(o_grids[joint[0]], o_grids[joint[1]], indexing='ij')
        C = preferences_dict[joint]

        plt.figure(figsize=(6,5))
        plt.pcolormesh(X, Y, C, shading='auto', cmap='coolwarm')
        plt.colorbar(label='Preference')
        plt.xlabel(f'Modality {joint[0]} (X)')
        plt.ylabel(f'Modality {joint[1]} (Y)')
        plt.title(f'Joint Preferences for modalities {joint}')
        plt.show()
    
    def _plot_signal_preferences(self, preferences_dict):
        # 1. Extract the data from your specific dictionaries
        # o_grids[2] contains the RSI values (e.g., 0 to 30)
        # preferences_dict[2] contains the log-preferences (the values peaking at 30)
        rsi_values = self.get_o_grid(2)
        log_preferences = self._convert_to_log_pref(preferences_dict)[2]

        # 2. Create the plot
        plt.figure(figsize=(12, 6))

        # Plot as a bar graph
        # 'width' should be adjusted based on the density of your grid
        bar_width = (rsi_values[1] - rsi_values[0]) * 0.8 if len(rsi_values) > 1 else 0.5

        plt.bar(rsi_values, log_preferences, width=bar_width, 
                color='skyblue', edgecolor='navy', alpha=0.7, 
                label='Log-Preference')

        # 3. Add styling and labels
        plt.title("Signal Modality Preferences from `preferences_dict[2]`", fontsize=14)
        plt.xlabel("Observation Value (RSI)", fontsize=12)
        plt.ylabel("Log-Probability Value", fontsize=12)

        # Mark the peak (the value the agent 'wants' to see)
        peak_idx = np.argmax(log_preferences)
        plt.axvline(x=rsi_values[peak_idx], color='red', linestyle='--', 
                    label=f'Target: {rsi_values[peak_idx]:.2f}')

        plt.grid(axis='y', linestyle=':', alpha=0.5)
        plt.legend()

        # 4. Save and Show
        plt.tight_layout()
        plt.savefig('signal_preferences_actual.png')
        plt.show()

        print(f"Plotted {len(rsi_values)} points.")
        print(f"Max Preference Value: {np.max(log_preferences)}")
        print(f"Min Preference Value: {np.min(log_preferences)}")

    
    def get_o_grid(self, modality_idx, N_grid=100):
        """
        Generate observation grid for a given modality.

        Parameters
        ----------
        modality_idx : int
            0 = x_obs, 1 = y_obs, 2 = signal
        N_grid : int
            Number of points in the observation grid

        Returns
        -------
        o_grid : np.ndarray
            1D array of observation values
        """
        if modality_idx == 0:  # x position
            return np.linspace(self.x_min, self.x_max, N_grid)

        elif modality_idx == 1:  # y position
            return np.linspace(self.y_min, self.y_max, N_grid)

        elif modality_idx == 2:  # signal strength
            o_min, o_max = self.rsi_min, self.RSI  # maximum signal at the goal
            return np.linspace(o_min, o_max, N_grid)

        else:
            raise ValueError(f"Unknown modality index: {modality_idx}")

    def _precompute_signal_mean(self):
        # 1. Determine the physical size of one grid cell
        # Assuming x_max=500 and states_dim=20, this is 25.0 cm/cell
        x_scale = (self.x_max - self.x_min) / self.states_dim[0]
        y_scale = (self.y_max - self.y_min) / self.states_dim[1]

        # 2. Shift to cell centers AND scale to physical centimeters
        x_curr_cm = (np.arange(self.states_dim[0]) + 0.5) * x_scale
        y_curr_cm = (np.arange(self.states_dim[1]) + 0.5) * y_scale
        x_goal_cm = (np.arange(self.states_dim[2]) + 0.5) * x_scale
        y_goal_cm = (np.arange(self.states_dim[3]) + 0.5) * y_scale

        # 3. Use 'ij' indexing to ensure Axis 0 = X and Axis 1 = Y
        Xc, Yc, Xg, Yg = np.meshgrid(
            x_curr_cm, y_curr_cm,
            x_goal_cm, y_goal_cm,
            indexing='ij'
        )

        # 4. d is now the physical distance in CM
        d = np.sqrt((Xc - Xg)**2 + (Yc - Yg)**2)

        # 5. Compute RSI expectation based on CM distance
        # alpha should now be tuned for cm (e.g., 0.01)
        self.mu_signal = self.RSI * np.exp(-self.alpha * d)

    
    def log_likelihoods(self, obs_val, modality_idx):
        if modality_idx == 0:  # x_obs
            # Calculate centimeters per cell
            x_scale = (self.x_max - self.x_min) / self.states_dim[0]
            # mu is now in centimeters: [12.5, 37.5, ..., 487.5]
            mu = (np.arange(self.states_dim[0]) + 0.5) * x_scale
            sigma = self.sigma_x

        elif modality_idx == 1:  # y_obs
            y_scale = (self.y_max - self.y_min) / self.states_dim[1]
            # mu is now in centimeters
            mu = (np.arange(self.states_dim[1]) + 0.5) * y_scale
            sigma = self.sigma_y

        elif modality_idx == 2:  # signal
            # This is already in the right "units" because precompute_signal_mean 
            # now uses cm distances to calculate RSI values (0-30).
            mu = self.mu_signal 
            sigma = self.sigma_s

        # Gaussian kernel calculation
        log_kernal = -0.5 * ((obs_val - mu) / sigma) ** 2

        return log_kernal

    def log_likelihoods_grid(self, o_grid, modality_idx, s_vals):
        """
        Evaluate log-likelihood of continuous observations over grid for a single state.
        s_vals: tuple of state indices for the factors this modality depends on.
        """
        # Define scaling factors
        x_scale = (self.x_max - self.x_min) / self.states_dim[0]
        y_scale = (self.y_max - self.y_min) / self.states_dim[1]

        if modality_idx == 0: # x_obs
            # s_vals[0] is the x_index
            mu = (s_vals[0] + 0.5) * x_scale
            return -0.5 * ((o_grid - mu) / self.sigma_x)**2 
            
        elif modality_idx == 1: # y_obs
            # s_vals[0] is the y_index (assuming factor 1 was passed)
            mu = (s_vals[0] + 0.5) * y_scale
            return -0.5 * ((o_grid - mu) / self.sigma_y)**2
            
        elif modality_idx == 2: # signal
            # s_vals = (x_curr_idx, y_curr_idx, x_goal_idx, y_goal_idx)
            x_c = (s_vals[0] + 0.5) * x_scale
            y_c = (s_vals[1] + 0.5) * y_scale
            x_g = (s_vals[2] + 0.5) * x_scale
            y_g = (s_vals[3] + 0.5) * y_scale
            
            d = np.sqrt((x_c - x_g)**2 + (y_c - y_g)**2)
            mu_signal = self.RSI * np.exp(-self.alpha * d)
            
            return -0.5 * ((o_grid - mu_signal) / self.sigma_s)**2
        