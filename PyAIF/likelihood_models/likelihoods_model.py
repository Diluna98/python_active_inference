import json
import scipy
from scipy.stats import t
from time import time
from matplotlib import scale
from matplotlib.pylab import beta
from numpy.char import array
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp
from scipy.ndimage import zoom
from scipy.signal import convolve2d
from itertools import product
from scipy.special import gammaln


class LikelihoodModels:
    def __init__(self, model_name, states_dim=None, obstacles_dic=None, obs_limits=None):
        self.states_dim = states_dim
        self.obstacles_dic = obstacles_dic
        self.obs_limits = obs_limits
        if model_name == "task":
            self.model = TaskLikelihoodModel(states_dim, obstacles_dic, obs_limits)
        elif model_name == "meta":
            #df = pd.read_csv("resolution_signatures.csv")
            #modalities = ['max_risk', 'max_ambiguity', 'inference_time_ms']
            #signatures_df = df.groupby('resolution')[modalities].agg(['mean', 'std'])
            self.model = MetaLikelihoodModel(states_dim)

    def update_likelihood_model(self, new_states_dim):
        self.states_dim = new_states_dim
        if isinstance(self.model, TaskLikelihoodModel):
            self.model = TaskLikelihoodModel(new_states_dim, self.obstacles_dic, self.obs_limits)
        elif isinstance(self.model, MetaLikelihoodModel):
            self.model = MetaLikelihoodModel(new_states_dim)
        return self.model

class MetaLikelihoodModel:
    def __init__(self, states_dim):
        self.states_dim = states_dim
        #self.resolutions = sorted(signatures_df.index.unique())
        cpu_scale = {
                        0: 4.0,   # low CPU
                        1: 3.0,   # medium CPU
                        2: 1.0    # high CPU (data source)
                    }
        self.eps=1e-16 # small constant for numerical stability in log calculations
        # profiled signatures for the "Base" (Low CPU)
        #self.base_signatures = self._parse_signatures(signatures_df)
        self.infog_proxy_min = 0
        self.infog_proxy_max = 2
        self.err_min = 2
        self.err_max = 10
        self.lat_min = 50
        self.lat_max = 9000
        self.avl_cpu_min = 0
        self.avl_cpu_max = 100
        K = 4 #num_models
        C = 4 #num_contexts
        with open("external_lm_params.json", "r") as f:
            data = json.load(f)

        self.mu_infog_proxy = np.array([0.02, 0.1, 0.5, 3.0], dtype=np.float64)
        self.sigma_infog_proxy = np.array([0.02, 0.015, 0.08, 0.4], dtype=np.float64)

        self.mu_err = np.ones((K, C))* 2.4
        
        self.kappa_err = np.ones((K, C)) * 1e-3 # Initial precision of the mean estimates

        self.alpha_err = np.ones((K, C)) * 1.0 # Varince in the data
        self.beta_err = np.ones((K, C)) * 1.0 # Varince scale
        
        self.mu_cpu = np.array([20.0, 57.5, 87.5], dtype=np.float64)
        self.sigma_cpu = np.array([8.0, 10.0, 8.0], dtype=np.float64)
        """
        mu_lat_1d = tables["mu_lat"]
        sigma_lat_1d = tables["sigma_lat"]

        num_res = self.states_dim[0]
        num_cpu = self.states_dim[2]

        self.mu_lat = np.zeros((num_res, num_cpu))
        self.sigma_lat = np.zeros((num_res, num_cpu))


        for r in range(num_res):
            for u in range(num_cpu):

                self.mu_lat[r, u] = mu_lat_1d[r] * cpu_scale[u]

                # uncertainty increases with degradation
                self.sigma_lat[r, u] = sigma_lat_1d[r]* cpu_scale[u]
        """
        
        self.mu_err = np.array(data["mu_err"])
        
        self.kappa_err = np.array(data["kappa_err"])
        self.alpha_err = np.array(data["alpha_err"])
        self.beta_err = np.array(data["beta_err"])
        
        self.mu_cpu = np.array([20.0, 57.5, 87.5], dtype=np.float64)
        self.sigma_cpu = np.array([8.0, 10.0, 8.0], dtype=np.float64)
        
        self.mu_lat = np.array(data["mu_lat"])
        self.sigma_lat = np.array(data["sigma_lat"])
        

        """        
        with open("external_lm_params.json", "r") as f:
            data = json.load(f)

        
        self.mu_div = np.array(data["mu_div"])
        self.sigma_div = np.array(data["sigma_div"])
        self.mu_err = np.array(data["mu_err"])
        self.sigma_err = np.array(data["sigma_err"])
        self.mu_lat = np.array(data["mu_lat"])
        self.sigma_lat = np.array(data["sigma_lat"])
        self.mu_cpu = np.array(data["mu_cpu"])
        self.sigma_cpu = np.array(data["sigma_cpu"])
        """
        self.pref_dep = [(0, 1)] # joint preference for info gain and accuracy
        self.log_preferences = self._build_preferences()

    def _softmax(self, x, axis = 0, gamma=1.0):
        exp_x = np.exp(gamma * x - np.max(gamma * x))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

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
        if modality_idx == 0:
            return np.linspace(self.infog_proxy_min, self.infog_proxy_max, N_grid)

        elif modality_idx == 1:
            return np.linspace(self.err_min, self.err_max, N_grid)

        elif modality_idx == 2:
            return np.linspace(self.lat_min, self.lat_max, N_grid)
        elif modality_idx == 3:
            return np.linspace(self.avl_cpu_min, self.avl_cpu_max, N_grid)

        else:
            raise ValueError(f"Unknown modality index: {modality_idx}")


    def _build_preferences(self, observations = None, scale = 1.0):
        if observations is not None:
            beta = self.compute_beta(observations)*0 + 1
        else:
            beta = 1
        preferences_dict = {}

        # Preference for divergence (modality 0)
        #infog_proxy_grid = np.linspace(self.infog_proxy_min, self.infog_proxy_max, 100)
        #normalized_div = (infog_proxy_grid - self.infog_proxy_min) / (self.infog_proxy_max - self.infog_proxy_min)  # Normalize to [0, 1]
        #midpoint = 0.03  # Set midpoint at 0.5 (500 ms), which is the critical threshold in the data 
        #steepness = 5.0  
        #C_infog_proxy = 1 / (1 + np.exp(-steepness * (normalized_div - midpoint)))

        # Scale it massively. When this gate opens, it must dominate everything else.
        #C_infog_proxy = C_infog_proxy * 50.0 

        #C_infog_proxy_probs = np.exp(C_infog_proxy - np.max(C_infog_proxy))
        #C_infog_proxy_probs /= C_infog_proxy_probs.sum()
        preferences_dict[0] = np.log(self.eps)

        if self.pref_dep is not None:

            joint = self.pref_dep[0]
            complexity = np.arange(100)
            pred_error = np.arange(100)
            C, P = np.meshgrid(complexity, pred_error, indexing='ij')
            C_norm = C / (C.max() + 1e-8)
            P_norm = P / (P.max() + 1e-8)
            P_centered = P_norm
            threshold = 0.4
            gain = np.tanh(5 * (C_norm - threshold))
            beta = 0.5
            C_joint = -beta * C_norm * P_centered

            C_joint_probs = self._softmax(C_joint.flatten(), gamma=1.0).reshape(C_joint.shape)
            preferences_dict[joint] = np.log(C_joint_probs + self.eps)
        else:
            # ------------------------------------------------------------------
            # 1. Prediction error: Scaled down initially
            # ------------------------------------------------------------------
            # Assuming err_grid is already defined (length 100)
            err_grid = np.linspace(self.err_min, self.err_max, 100)
            
            # 1. Create a steep drop-off centered around index 20
            # We map the 100-step grid to an index-based array to easily target "index 20"
            x = np.arange(100) 
            center_index = 5  # Centers the inflection point near index 20-25
            steepness = 0.1   # Controls how rapidly it plunges after index 20

            # 2. Use a negative sigmoid shape for C_err
            C_err = -1 / (1 + np.exp(-steepness * (x - center_index)))

            # 3. Scale C_err to control the overall vertical span of the log curve
            # A larger multiplier creates a larger gap between the start and end values
            if observations is not None:
                C_err = beta * C_err
            else:
                C_err = beta * C_err


            # 4. Your original exponentiation, normalization, and log steps
            #C_err_probs = np.exp(C_err - np.max(C_err))
            #C_err_probs /= C_err_probs.sum()

            C_err_probs = self._softmax(C_err.flatten(), gamma=1.0).reshape(C_err.shape)

            preferences_dict[1] = np.log(C_err_probs + self.eps)
        

        

        # ------------------------------------------------------------------
        # 3. Latency: Highly sensitive cost
        # ------------------------------------------------------------------
        # 1. Create a 100-step grid indices
        x = np.arange(100) 

        # 2. Define the shift point where it begins to fall faster
        delay_index = 5  

        # 3. Create a continuous, multi-slope curve
        # Start with a gentle, slight decrease from the very beginning
        initial_slope = -0.005
        C_lat = initial_slope * x

        # 4. After the delay index, add an extra steeper downward slope
        steep_slope = -0.045
        C_lat[delay_index:] += steep_slope * (x[delay_index:] - delay_index)

        # 5. Softmax with a low gamma to keep the linear behavior intact
        gamma = 0.15 
        C_lat_probs = self._softmax(C_lat * gamma, gamma=1.0)

        # 6. Compute the final log-probabilities
        preferences_dict[2] = np.log(C_lat_probs + self.eps)
        
        #plt.plot(preferences_dict[1])
        #plt.plot(preferences_dict[2])
        #plt.plot(preferences_dict[3])
        #plt.show()
        # Preference for cpu (modality 4)
        #cpu_grid = np.linspace(self.avl_cpu_min, self.avl_cpu_max, 100)
        #C_cpu = cpu_grid  # no preference
        #C_cpu_probs = np.exp(C_cpu - np.max(C_cpu))
        #C_cpu_probs /= C_cpu_probs.sum()
        #C_cpu_probs = self._softmax(C_cpu.flatten(), gamma=1.0).reshape(C_cpu.shape)
        preferences_dict[3] = np.log(self.eps)
        self.log_preferences = preferences_dict
        return preferences_dict
    
    def compute_beta(self, obs, max_latency_ms=3000.0, max_err=3.0, 
                    max_complexity=0.3, max_cpu=50.0) -> float:
        """
        obs: (context_proxy, prediction_error, inference_latency, cpu_availability)
        
        β = 0: very constrained → prefer low resolution
        β = 1: fully available  → prefer high resolution
        """
        
        # What we CAN spend (headroom)
        latency_headroom = 1.0 - (obs[2] / max_latency_ms)  # high latency → low headroom
        cpu_headroom     = 1.0 - (obs[3] / max_cpu)          # high cpu use → low headroom
        
        headroom = (np.clip(latency_headroom, 0.0, 1.0) + 
                    np.clip(cpu_headroom,     0.0, 1.0)) / 2.0
        
        # What we NEED to spend (pressure toward higher resolution)
        error_pressure   = np.clip(obs[1] / max_err,        0.0, 1.0)  # high error → need more
        context_pressure = np.clip(obs[0] / max_complexity, 0.0, 1.0)  # high complexity → need more
        
        pressure = (error_pressure + context_pressure) / 2.0
        
        # β: only spend if we both CAN and NEED to
        β = np.clip(headroom * pressure, 0.0, 1.0)
        
        return β


    def likelihoods(self, obs_val, modality_idx):
        if modality_idx == 0:

            exponent = np.exp(-0.5 * ((obs_val - self.mu_infog_proxy) / self.sigma_infog_proxy) ** 2)
            normalization = 1.0 / (self.sigma_infog_proxy * np.pi)
            return exponent * normalization

        elif modality_idx == 1:
            nu = 2.0 * self.alpha_err
            scale = np.sqrt(
                self.beta_err
                * (self.kappa_err + 1.0)
                / (self.alpha_err * self.kappa_err)
            )
            log_p_obs_given_state = scipy.stats.t.pdf(
                obs_val,
                df=nu,
                loc=self.mu_err,
                scale=scale
            )
            return log_p_obs_given_state

        elif modality_idx == 2:

            exponent = np.exp(-0.5 * ((obs_val - self.mu_lat) / self.sigma_lat) ** 2)
            normalization = 1.0 / (self.sigma_lat * np.pi)
            return exponent * normalization
        
        elif modality_idx == 3:

            exponent = np.exp(-0.5 * ((obs_val - self.mu_cpu) / self.sigma_cpu) ** 2)
            normalization = 1.0 / (self.sigma_cpu * np.pi)

            return exponent * normalization

       
    
    def likelihoods_grid_vec(self, o_grid, modality_idx, s_vals):
        """
        Vectorized likelihood evaluation of Likelihood_grid funtion.

        Parameters
        ----------
        o_grid : ndarray
            Observation grid

        modality_idx : int

        s_vals : ndarray or tuple of ndarrays
            Latent state samples

        Returns
        -------
        P : ndarray
            Shape:
                (num_samples, len(o_grid))
        """

        o_grid = np.asarray(o_grid, dtype=np.float32)
        dx = o_grid[1] - o_grid[0]

        if modality_idx == 0:

            mu = self.mu_infog_proxy[s_vals]
            sigma = self.sigma_infog_proxy[s_vals]
            diff = (
                o_grid[None, :] - mu[:, None]
            ) / sigma[:, None]

            exponent = np.exp(-0.5 * diff * diff)

            normalization = (
                1.0 /
                (sigma[:, None] * 2.50662827463)
            )

            P = exponent * normalization

            P *= dx

            P /= (
                P.sum(axis=1, keepdims=True)
                + self.eps
            )

            return P.astype(np.float32)

        elif modality_idx == 1:

            s0, s1 = s_vals

            m = self.mu_err[s0, s1]
            kappa = self.kappa_err[s0, s1]
            alpha = self.alpha_err[s0, s1]
            beta = self.beta_err[s0, s1]

            nu = 2.0 * alpha

            scale = np.sqrt(
                beta * (kappa + 1.0)
                / (alpha * kappa + self.eps)
            )

            diff = (o_grid[None, :] - m[:, None]) / (scale[:, None] + self.eps)

            base = 1.0 + (diff * diff) / (nu[:, None] + self.eps)

            log_coef = (
                gammaln((nu + 1.0) / 2.0)
                - gammaln(nu / 2.0)
                - 0.5 * (np.log(nu * np.pi) + 2.0 * np.log(scale + self.eps))
            )

            log_P = log_coef[:, None] - ((nu + 1.0) / 2.0)[:, None] * np.log(base)

            P = np.exp(log_P)

            P *= dx

            P /= (P.sum(axis=1, keepdims=True) + self.eps)

            return P.astype(np.float32)

        elif modality_idx == 2:

            s0, s2 = s_vals

            mu = self.mu_lat[s0, s2]
            sigma = self.sigma_lat[s0, s2]
            diff = (
                o_grid[None, :] - mu[:, None]
            ) / sigma[:, None]

            exponent = np.exp(-0.5 * diff * diff)

            normalization = (
                1.0 /
                (sigma[:, None] * 2.50662827463)
            )

            P = exponent * normalization

            P *= dx

            P /= (
                P.sum(axis=1, keepdims=True)
                + self.eps
            )

            return P.astype(np.float32)

        elif modality_idx == 3:

            mu = self.mu_cpu[s_vals]
            sigma = self.sigma_cpu[s_vals]
            diff = (
                o_grid[None, :] - mu[:, None]
            ) / sigma[:, None]

            exponent = np.exp(-0.5 * diff * diff)

            normalization = (
                1.0 /
                (sigma[:, None] * 2.50662827463)
            )

            P = exponent * normalization

            P *= dx

            P /= (
                P.sum(axis=1, keepdims=True)
                + self.eps
            )

            return P.astype(np.float32)

        else:
            raise ValueError("Invalid modality")

    def update_mu_sigma_vectorized(self, observations, qs, lr=0.1, threshold=1e-2):
        
        #qs_res = qs[0]
        qs_res = qs[0]#np.array([0., 0., 1., 0.])
        qs_con = qs[1]
        qs_cpu = qs[2]#np.array([0., 0., 1.])#qs[2]
        for modality_idx in range(4):
            if modality_idx in [0, 2, 3, 4]: # For these modalities, we do not perform learning.
                continue
            # 1. Prepare Observations and Beliefs
            obs = observations[0][modality_idx]
            
            if modality_idx == 0:
                err = obs - self.mu_infog_proxy
                """
                # mean update
                self.mu_infog_proxy = np.clip(
                                            self.mu_infog_proxy + lr * qs_con * err,
                                            self.infog_proxy_min,
                                            self.infog_proxy_max
                                            )
                """
                # variance update
                var = self.sigma_infog_proxy ** 2
                var = var + lr * qs_con * ((err ** 2) - var)

                self.sigma_infog_proxy = np.sqrt(np.clip(var, 1e-6, None))

            elif modality_idx == 1:
                gamma = np.outer(qs_res, qs_con)

                m_old = self.mu_err
                kappa_old = self.kappa_err
                alpha_old = self.alpha_err
                beta_old = self.beta_err

                kappa_new = kappa_old + gamma

                m_new = (
                    kappa_old * m_old
                    + gamma * obs
                ) / kappa_new

                alpha_new = alpha_old + 0.5 * gamma

                beta_new = beta_old + (
                    0.5
                    * (kappa_old * gamma / kappa_new)
                    * (obs - m_old) ** 2
                )

                self.mu_err = m_new
                self.kappa_err = kappa_new
                self.alpha_err = alpha_new
                self.beta_err = beta_new

                #print(f"Updated mu_err:\n{self.mu_err}")

            elif modality_idx == 2:
                err = obs - self.mu_lat
                gamma = np.outer(qs_res, qs_cpu)

                # mean
                self.mu_lat = np.clip(
                                            self.mu_lat + lr * gamma * err,
                                            self.lat_min,
                                            self.lat_max
                                        )

                # variance
                var = self.sigma_lat ** 2
                var += lr * gamma * ((err ** 2) - var)

                self.sigma_lat = np.sqrt(np.clip(var, 1e-6, None))
                #print(f"Updated mu_lat:\n{self.mu_lat}")

            elif modality_idx == 3:
                err = obs - self.mu_cpu

                # mean update
                self.mu_cpu = np.clip(
                    self.mu_cpu + lr * qs_cpu * err,
                    self.avl_cpu_min,
                    self.avl_cpu_max
                )

                # variance update
                var = self.sigma_cpu ** 2
                var = var + lr * qs_cpu * ((err ** 2) - var)

                self.sigma_cpu = np.sqrt(np.clip(var, 1e-6, None))
            

class TaskLikelihoodModel:
    def __init__(self, states_dim, obstacles_dict, obs_limits, sigma_x=1, sigma_y=1, sigma_s=2, alpha=0.01):
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

        self.negative_pref = -500  # strong negative preference for obstacles
        self.eps=1e-8 # small constant for numerical stability in log calculations
        self.log_kernel_over_time = []

        self.pref_fig = None
        self.pref_ax = None
        self.pref_mesh = None

        self.log_preferences = self._build_preferences()
        #self._plot_joint_preferences(self.log_preferences)
        #self.init_pref_plot()
        self._precompute_mean_sigma() 
        #self.compute_sensitivity_map()
        #self._plot_signal_expectation_map((7, 1))
        #self._plot_signal_preferences(self.log_preferences)
        #self._plot_signal_preferences(self.log_preferences)


    def update_pref_plot(self, joint=(0,1)):

        C = self.log_preferences[joint]

        self.pref_mesh.set_array(C.ravel())

        self.pref_fig.canvas.draw_idle()
        self.pref_fig.canvas.flush_events()
        
    
    def _convert_to_log_pref(self, pref_dic):
        log_pref_dic = {}
        for key, C in pref_dic.items():
            # Shift so the 'best' state is 0, others are negative
            log_pref_dic[key] = C - np.max(C) 
        return log_pref_dic
    
    def _softmax(self, x, axis = 0, gamma=1.0):
        exp_x = np.exp(gamma * x - np.max(gamma * x))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    
    def _build_preferences(self, goal_pos=None, sigma_goal=100.0, sigma_signal=0.01, scale = 1.0):
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
                self.X_points, self.Y_points = np.meshgrid(grids[0], grids[1], indexing='ij')
                
                # Start with neutral preference (0 in log space)
                self.C = np.zeros_like(self.X_points) + 0.01 # small negative baseline to avoid ties

                # Goal preference (Log-Gaussian)
                if goal_pos is not None:
                    x_goal, y_goal = goal_pos
                    # Direct log-space calculation
                    self.C += -0.5 * ((x_goal - self.X_points)/sigma_goal)**2
                    self.C += -0.5 * ((y_goal - self.Y_points)/sigma_goal)**2

                # Obstacles (Apply negative penalty)
                if self.obstacles_dict is not None:
                    # Assuming obstacles are defined in CM coordinates (0-500)
                    self.obst_mask = {}
                    for block_key in self.obstacles_dict.keys():
                        x_min, x_max, y_min, y_max = self.obstacles_dict[block_key]
                        self.obst_mask[block_key] = (self.X_points >= x_min) & (self.X_points <= x_max) & (self.Y_points >= y_min) & (self.Y_points <= y_max)
                        self.C[self.obst_mask[block_key]] += self.negative_pref

                C_probs = self._softmax(self.C.flatten(), gamma=1.0).reshape(self.C.shape)
                preferences_dict[joint] = np.log(C_probs + self.eps) # Add small value for numerical stability

        # --- Single-modality preferences ---
        if 2 not in preferences_dict: # Signal Modality
            o_grid = np.linspace(0, 30, 100)
            def plateau_pref(x, center=10, steepness=0.25):
                # This creates a "S" curve that is very flat at the ends
                return 1 / (1 + np.exp(-steepness * (x - center)))

            # Apply a small offset so log(0) doesn't break the code
            # Multiply by a "Motivation" factor (e.g., 2.0) to control the depth
            C_signal = np.log(plateau_pref(o_grid) + 0.1)
            C_signal = self._softmax(C_signal.flatten(), gamma=1.0).reshape(C_signal.shape)
            
            preferences_dict[2] = np.log(C_signal + self.eps)

        return preferences_dict

    
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

    def _precompute_mean_sigma(self):
        """
        # x_scale and y_scale stay the same
        x_scale = (self.x_max - self.x_min) / self.states_dim[0]
        y_scale = (self.y_max - self.y_min) / self.states_dim[1]

        # Centers
        # x_idx 0 is far left, y_idx 0 is far top
        x_coords = (np.arange(self.states_dim[0]) + 0.5) * x_scale
        y_coords = (np.arange(self.states_dim[1]) + 0.5) * y_scale

        self.x_coords_agent = x_coords
        self.y_coords_agent = y_coords

        block_size = int(self.states_dim[0]/np.sqrt(self.states_dim[2]))  # 20 -> 5
        sqrt_goal_dim = int(np.sqrt(self.states_dim[2]))

        self.x_coords_goal = x_coords.reshape(sqrt_goal_dim, block_size).mean(axis=1)
        self.y_coords_goal = y_coords.reshape(sqrt_goal_dim, block_size).mean(axis=1)
        """
        agent_cell_size = self.x_max / self.states_dim[0]  # = 25 cm
        
        sqrt_goal_dim = int(np.sqrt(self.states_dim[2]))

        goal_cell_size = self.x_max / sqrt_goal_dim

        self.x_coords_goal = (np.arange(sqrt_goal_dim) + 0.5) * goal_cell_size
        self.y_coords_goal = (np.arange(sqrt_goal_dim) + 0.5)* goal_cell_size
        self.x_coords_agent = (np.arange(self.states_dim[0]) + 0.5) * agent_cell_size
        self.y_coords_agent = (np.arange(self.states_dim[1]) + 0.5) * agent_cell_size

        Xc, Yc, Xg, Yg = np.meshgrid(
            self.x_coords_agent, self.y_coords_agent,
            self.x_coords_goal, self.y_coords_goal,
            indexing='ij'
        )

        d = np.sqrt((Xc - Xg)**2 + (Yc - Yg)**2)
        
        mu = self.RSI * np.exp(-self.alpha * d)

        mu = np.transpose(mu, (0, 1, 3, 2))

        self.mu_signal = mu.reshape(
            self.states_dim[0],
            self.states_dim[1],
            self.states_dim[2]
        )

        self.sigma_signal = np.full((self.states_dim[0], self.states_dim[1], self.states_dim[2]), self.sigma_s, dtype=np.float64)
        
        x_coords = self.x_coords_agent
        y_coords = self.y_coords_agent
        self.mu_x = x_coords
        self.sigma_x = np.full((self.states_dim[0]), self.sigma_x, dtype=np.float64)
        self.mu_y = y_coords
        self.sigma_y = np.full((self.states_dim[1]), self.sigma_y, dtype=np.float64)


        ### for master model (highest resolution) ###
        sqrt_goal_dim_master = int(np.sqrt(400))

        goal_cell_size_master = self.x_max / sqrt_goal_dim_master

        x_coords_goal_master = (np.arange(sqrt_goal_dim_master) + 0.5) * goal_cell_size_master
        y_coords_goal_master = (np.arange(sqrt_goal_dim_master) + 0.5) * goal_cell_size_master

        Xc_master, Yc_master, Xg_master, Yg_master = np.meshgrid(
            self.x_coords_agent, self.y_coords_agent,
            x_coords_goal_master, y_coords_goal_master,
            indexing='ij'
        )

        d_master = np.sqrt((Xc_master - Xg_master)**2 + (Yc_master - Yg_master)**2)
        
        mu_signal_master = (self.RSI * np.exp(-self.alpha * d_master))
        mu_signal_master = np.transpose(mu_signal_master, (0, 1, 3, 2))

        self.mu_signal_master = mu_signal_master.reshape(
            self.states_dim[0], self.states_dim[1], -1
        )
        self.sigma_signal_master = np.full((self.states_dim[0], self.states_dim[1], 400), self.sigma_s, dtype=np.float64)

        dx = np.gradient(self.mu_signal_master, axis=0)
        dy = np.gradient(self.mu_signal_master, axis=1)
        self.fisher_map_signal = (dx**2 + dy**2) / (self.sigma_signal_master**2 + self.eps)     
    
    def likelihoods(self, obs_val, modality_idx, master=False):
        if modality_idx == 0:  # x_obs

            exponent = np.exp(-0.5 * ((obs_val - self.mu_x) / self.sigma_x) ** 2)
            normalization = 1.0 / (self.sigma_x * 2.50662827463)

        elif modality_idx == 1:  # y_obs

            exponent = np.exp(-0.5 * ((obs_val - self.mu_y) / self.sigma_y) ** 2)
            normalization = 1.0 / (self.sigma_y * 2.50662827463)

        elif modality_idx == 2:  # signal
            if not master:
                exponent = np.exp(-0.5 * ((obs_val - self.mu_signal) / self.sigma_signal) ** 2)
                normalization = 1.0 / (self.sigma_signal * 2.50662827463)
            if master:
                exponent = np.exp(-0.5 * ((obs_val - self.mu_signal_master) / self.sigma_signal_master) ** 2)
                normalization = 1.0 / (self.sigma_signal_master * 2.50662827463)

        return exponent * normalization

    def likelihoods_grid(self, o_grid, modality_idx, s_vals):
        """
        Returns normalized likelihood over o_grid:
        p(o | s) such that sum_o p(o|s) = 1
        """

        if modality_idx == 0:  # x_obs

            exponent = np.exp(-0.5 * ((o_grid - self.mu_x[s_vals[0]]) / self.sigma_x[s_vals[0]]) ** 2)
            normalization = 1.0 / (self.sigma_x[s_vals[0]] * 2.50662827463)

        elif modality_idx == 1:  # y_obs

            exponent = np.exp(-0.5 * ((o_grid - self.mu_y[s_vals[0]]) / self.sigma_y[s_vals[0]]) ** 2)
            normalization = 1.0 / (self.sigma_y[s_vals[0]] * 2.50662827463)

        elif modality_idx == 2:  # signal

            exponent = np.exp(-0.5 * ((o_grid - self.mu_signal[s_vals[0], s_vals[1], s_vals[2]]) / self.sigma_signal[s_vals[0], s_vals[1], s_vals[2]]) ** 2)
            normalization = 1.0 / (self.sigma_signal[s_vals[0], s_vals[1], s_vals[2]] * 2.50662827463)

        L = exponent * normalization
        dx = o_grid[1] - o_grid[0]  # assume uniform grid
        P = L * dx
        P = P / (P.sum() + self.eps)
        return P
    
    def likelihoods_grid_vec(self, o_grid, modality_idx, s_vals):
        """
        Vectorized likelihood evaluation of Likelihood_grid funtion.

        Parameters
        ----------
        o_grid : ndarray
            Observation grid

        modality_idx : int

        s_vals : ndarray or tuple of ndarrays
            Latent state samples

        Returns
        -------
        P : ndarray
            Shape:
                (num_samples, len(o_grid))
        """

        o_grid = np.asarray(o_grid, dtype=np.float32)
        dx = o_grid[1] - o_grid[0]

        if modality_idx == 0:

            mu = self.mu_x[s_vals]
            sigma = self.sigma_x[s_vals]

        elif modality_idx == 1:

            mu = self.mu_y[s_vals]
            sigma = self.sigma_y[s_vals]

        elif modality_idx == 2:

            s0, s1, s2 = s_vals

            mu = self.mu_signal[s0, s1, s2]
            sigma = self.sigma_signal[s0, s1, s2]

        else:
            raise ValueError("Invalid modality")

        diff = (
            o_grid[None, :] - mu[:, None]
        ) / sigma[:, None]

        exponent = np.exp(-0.5 * diff * diff)

        normalization = (
            1.0 /
            (sigma[:, None] * 2.50662827463)
        )

        P = exponent * normalization

        P *= dx

        P /= (
            P.sum(axis=1, keepdims=True)
            + self.eps
        )

        return P.astype(np.float32)
    
    def update_C_vectorized(self, bayesian_mod_avg, lr=0.1, threshold=1e-2, sigma =500):
        Xg, Yg = np.meshgrid(self.x_coords_goal, self.y_coords_goal, indexing='ij')
        q_goal = np.mean(bayesian_mod_avg[:, 2], axis=0)
        sqrt_goal_dim = int(np.sqrt(len(q_goal)))
        q_goal_2d = q_goal.reshape(sqrt_goal_dim, sqrt_goal_dim)
        q_x = q_goal_2d.sum(axis=1)
        q_y = q_goal_2d.sum(axis=0)

        P_pref = np.zeros_like(self.X_points)

        block_size_x = len(self.get_o_grid(0)) // Xg.shape[0]
        block_size_y = len(self.get_o_grid(1)) // Yg.shape[1]

        kernel = np.array([
                    [2, 2, 2],
                    [2, 4, 2],
                    [2, 2, 2]
                ], dtype=float)

        kernel /= kernel.sum()

        for i in range(Xg.shape[0]):
            for j in range(Xg.shape[1]):

                weight = q_goal_2d[j, i]

                x_start = i * block_size_x
                x_end   = (i + 1) * block_size_x

                y_start = j * block_size_y
                y_end   = (j + 1) * block_size_y

                P_pref[x_start:x_end, y_start:y_end] = weight
        # if Obstacles, do not change preferences at those points
        if self.obstacles_dict is not None:
            for block_key in self.obstacles_dict.keys():
                mask = self.obst_mask[block_key]

                for _ in range(100):

                    P_num = convolve2d(P_pref, kernel, mode='same', boundary='fill', fillvalue=0)
                    P_den = convolve2d((~mask).astype(float), kernel, mode='same', boundary='fill', fillvalue=0)

                    P_new = P_num / (P_den + 1e-8)

                    P_new[mask] = 0.0  # Keep obstacle areas at zero preference
                    P_pref = P_new

        self.C = self.C + 1* lr * P_pref
        C_probs = self.C - np.max(self.C)
        C_probs = np.exp(C_probs) / np.sum(np.exp(C_probs))
        self.log_preferences[(0, 1)] = np.log(C_probs + 1e-16)
        self.update_pref_plot()

    def update_mu_sigma_vectorized(self, observations, bayesian_mod_avg, lr=0.1, threshold=1e-2):

        qs_x = np.array([b[0] for b in bayesian_mod_avg])
        qs_y = np.array([b[1] for b in bayesian_mod_avg])
        qs_goal = np.array([b[2] for b in bayesian_mod_avg])
        for modality_idx in range(3):
            # 1. Prepare Observations and Beliefs
            obs = np.array([o[modality_idx] for o in observations.values()])

            # 2. Compute Joint Belief: (T, 5, 5, 25)
            if modality_idx == 0:
                # For position modalities, we can directly use
                # qs_x and qs_y as they are independent
                joint_belief =  qs_x
                # 3. Compute Errors
                errors = obs[:, None] - self.mu_x[None, :]
                mask = joint_belief > threshold
                masked_update = errors * (joint_belief * mask)
                self.mu_x += lr * np.sum(masked_update, axis=0)
                # 6. Sigma Update (Only count errors where the mask was active)
                if np.any(mask):
                    weights = joint_belief * mask

                    # normalize weights
                    norm = weights.sum(axis=0) + self.eps

                    # sigma update (log-space)
                    log_sigma = np.log(self.sigma_x + self.eps)

                    grad = (errors**2 / (self.sigma_x**2)) - 1
                    grad_update = (weights * grad).sum(axis=0) / norm

                    log_sigma += lr * grad_update

                    self.sigma_x = np.exp(log_sigma)
                    self.sigma_x = np.clip(self.sigma_x, 0.1, 10.0)

            elif modality_idx == 1:
                joint_belief = qs_y
                # 3. Compute Errors
                errors = obs[:, None] - self.mu_y[None, :]
                mask = joint_belief > threshold
                masked_update = errors * (joint_belief * mask)
                self.mu_y += lr * np.sum(masked_update, axis=0)
                # 6. Sigma Update (Only count errors where the mask was active)
                if np.any(mask):
                    log_sigma = np.log(self.sigma_y + self.eps)  # Add small value for numerical stability
                    grad = (errors**2 / (self.sigma_y**2)) - 1
                    masked_grad = grad * (joint_belief * mask)
                    log_sigma += lr* np.mean(masked_grad, axis=0)
                    self.sigma_y = np.exp(log_sigma)
                    self.sigma_y = np.clip(self.sigma_y, 0.1, 10.0)
            else:
                # For signal modality, we need to consider the joint belief over (x, y, goal)
                joint_belief = np.einsum('ij,ik,il->ijkl', qs_x, qs_y, qs_goal)

                # 3. Compute Errors
                errors = obs[:, None, None, None] - self.mu_signal[None, :, :, :]

                # 4. Create the Mask
                # Only update where the agent's confidence in that (Pos, Goal) pair is > threshold
                mask = joint_belief > threshold


                # 5. Apply Mask to the Update
                # We multiply the update by the mask (True=1, False=0)
                masked_update = errors * (joint_belief * mask)
                
                # Sum across time and apply
                self.mu_signal += lr * np.sum(masked_update, axis=0)

                # 6. Sigma Update (Only count errors where the mask was active)
                if np.any(mask):
                    log_sigma = np.log(self.sigma_signal + self.eps)  # Add small value for numerical stability
                    grad = (errors**2 / (self.sigma_signal**2)) - 1
                    masked_grad = grad * (joint_belief * mask)
                    log_sigma += lr * np.mean(masked_grad, axis=0)
                    self.sigma_signal = np.exp(log_sigma)
                    self.sigma_signal = np.clip(self.sigma_signal, 0.1, 10.0)

    def plot_goal_likelihood_inference(self, obs, goal_pos=(237, 471)):
        """
        obs: [phys_x, phys_y, rsi_val]
        goal_pos: [true_goal_x, true_goal_y]
        """
        X_dim, Y_dim, _ = self.mu_signal.shape
        agent_x, agent_y, rsi_obs = obs[0], obs[1], obs[2]
        
        # 1. Find Agent's current grid position
        agent_idx_x = int(np.clip(agent_x / (500 / X_dim), 0, X_dim - 1))
        agent_idx_y = int(np.clip(agent_y / (500 / Y_dim), 0, Y_dim - 1))

        # 2. Get the expected RSI for EVERY possible goal location
        # given we are at (agent_idx_x, agent_idx_y)
        # This is a 1D vector of 25 values
        expected_rsis = self.mu_signal[agent_idx_x, agent_idx_y, :]
        
        # 3. Calculate Likelihood: How close is our observation to these expectations?
        # We use a small sigma (e.g., 0.5) to define how "strict" the matching is
        sigma = 0.5 
        likelihood_vec = np.exp(-0.5 * ((expected_rsis - rsi_obs) / sigma)**2)
        
        # 4. Reshape likelihoods back into a 5x5 grid of potential goals
        # Based on your meshgrid(indexing='ij') and reshape(-1)
        goal_likelihood_map = likelihood_vec.reshape(X_dim, Y_dim)

        # 5. Plotting
        plt.figure(figsize=(10, 8))
        
        # Note: We transpose (.T) if your Y is rows and X is columns in the plot
        im = plt.imshow(goal_likelihood_map.T, origin='upper', cmap='magma',
                        extent=[0, 500, 500, 0], interpolation='gaussian')
        plt.colorbar(im, label='Likelihood of Goal Being Here')

        # Marker for where the agent actually is
        plt.scatter(agent_x, agent_y, color='cyan', s=100, label='Agent Current Pos')
        # Marker for where the goal actually is
        plt.scatter(goal_pos[0], goal_pos[1], color='lime', marker='*', s=300, label='True Goal')

        plt.title(f"Inference: Where is the goal? (RSI Obs: {rsi_obs:.2f})")
        plt.xlabel("Physical X (cm)")
        plt.ylabel("Physical Y (cm)")
        plt.legend()
        plt.show()

    def _plot_signal_expectation_map(self, goal_pos):
        goal_x_idx, goal_y_idx = goal_pos

        # 1. Extract the slice
        # mu_signal[x_idx, y_idx, :, :] gives RSI expectations for all goals
        # indexing='ij' ensures that the first colon is Goal X and the second is Goal Y
        # However, imshow expects (rows, columns) which is (Y, X)
        # So we transpose the slice for correct visualization: .T
        goal_expectation_map = self.mu_signal[:, :, goal_x_idx, goal_y_idx].T

        plt.figure(figsize=(8, 6))
        
        # 2. Use 'extent' to map the 0-20 indices to 0-500 cm on the plot axes
        plt.imshow(goal_expectation_map, 
                origin='lower', 
                cmap='viridis',
                extent=[self.x_min, self.x_max, self.y_min, self.y_max])
        
        plt.colorbar(label='Expected RSI')
        plt.title(f"Expected Signal Map (Agent at Index {goal_pos})")
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
        plt.gca().invert_yaxis()
        plt.show()

    def init_pref_plot(self, joint=(0,1)):

        o_grids = {m: self.get_o_grid(m) for m in (0,1)}
        X, Y = np.meshgrid(o_grids[joint[0]], o_grids[joint[1]], indexing='ij')

        C = self.log_preferences[joint]

        self.pref_fig, self.pref_ax = plt.subplots(figsize=(6,5))

        self.pref_mesh = self.pref_ax.pcolormesh(
            X, Y, C,
            shading='auto',
            cmap='coolwarm'
        )

        plt.colorbar(self.pref_mesh, ax=self.pref_ax, label='Preference')

        self.pref_ax.set_xlabel(f'Modality {joint[0]} (X)')
        self.pref_ax.set_ylabel(f'Modality {joint[1]} (Y)')
        self.pref_ax.set_title(f'Joint Preferences {joint}')
        self.pref_ax.invert_yaxis()

        plt.show(block=False)
    
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

    def compute_sensitivity(self, observation):
        #### only consider signal modality for sensitivity,
        # since it's the only one that tells about the env complexity
        likelihood = self.likelihoods(observation[2], 2, master=True)
        normalized_likelihood = likelihood / (likelihood.sum() + self.eps)
        S_m = np.sum(normalized_likelihood * self.fisher_map_signal)
        return S_m






