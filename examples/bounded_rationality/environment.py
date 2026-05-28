import random
from matplotlib import patches
import numpy as np
import matplotlib.pyplot as plt
import math

class GridEnvironment:
    def __init__(self, size, workspace_length=500, workspace_width=500, s_x=25/2, s_y=25/2, g_x=237.5, g_y=462.5, RSI=30, alpha=0.01):
        self.grid_size = size
        self.workspace_length = workspace_length
        self.workspace_width = workspace_width
        self.cell_length = self.workspace_length / self.grid_size
        self.cell_width = self.workspace_width / self.grid_size
        self.RSI = RSI
        self.alpha = alpha # signal decay rate for RSI
        self.start_position = (s_x, s_y)
        self.goal_position = (g_x, g_y)
        self.current_position = self.start_position
        
        self.current_position_grid = self._world_to_grid_continuous(*self.current_position)
        self.obstacles = None#self._get_obstacles_dict()


        self.visualize = True  # Set to True to enable visualization

    def get_obs_limits(self):
        # Return the limits for each observation modality
        return {
            'x_obs': (0, self.workspace_length),
            'y_obs': (0, self.workspace_width),
            'rsi_obs': (0, self.RSI)
        }
    
    def _get_obstacles_dict(self):
        obstacles_dict = {
                0: [0, 395, 255, 295],    # obstacle 1: [x_min, x_max, y_min, y_max]
                #1: [0, 395, 305, 345],    # obstacle 2: [x_min, x_max, y_min, y_max]
                #2: [455, 500, 255, 345]    # obstacle 3: [x_min, x_max, y_min, y_max]
            }
        return obstacles_dict
    
    def _in_obstacle(self, x, y):
        for corners in self.obstacles.values():
            x_min, x_max, y_min, y_max = corners
            if x_min <= x <= x_max and y_min <= y <= y_max:
                return True
        return False
    
    def _get_obstacle_corners(self, coords):
        # coords = [x_min, x_max, y_min, y_max]
        x_min, x_max, y_min, y_max = coords
        return [
            (x_min, y_min), # Bottom-Left
            (x_min, y_max), # Top-Left
            (x_max, y_max), # Top-Right
            (x_max, y_min)  # Bottom-Right
        ]
    
    def update_plot(self, stat=None):
            if len(self.trajectory) < 1:
                return

            traj = np.array(self.trajectory)
            
            # Update the Agent's current position marker
            current_pos = traj[-1]
            self.agent_marker.set_offsets([current_pos])

            # --- Add stat annotation ---
            if stat is not None:
                txt = self.ax.text(
                    current_pos[0], current_pos[1],
                    f"{stat:.2f}",
                    fontsize=8,
                    color='black'
                )
                self.stat_texts.append(txt)

            # If we have at least 2 points, we can show the path and the latest move
            if len(traj) >= 2:
                # Update historical trajectory (all points up to the second to last)
                self.traj_line.set_data(traj[:-1, 0], traj[:-1, 1])
                
                # Update the latest move segment (from second-to-last to current)
                self.latest_move_line.set_data(traj[-2:, 0], traj[-2:, 1])
            else:
                # Only one point exists (at start)
                self.traj_line.set_data(traj[:, 0], traj[:, 1])

            # Redraw
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

    def _setup_plot(self):
        self.stat_texts = []
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        
        # Plot the static RSI map
        self.im = self.ax.imshow(
            self.RSI_map.T, 
            origin='upper', 
            extent=[0, self.workspace_length, self.workspace_width, 0], # Y is [top, bottom]
            aspect='auto'
        )
        self.fig.colorbar(self.im, ax=self.ax, label="RSI Signal Strength")

        # --- DRAW OBSTACLES ---
        if self.obstacles is not None:
            for obs_id, limits in self.obstacles.items():
                # Create a polygon patch from the corners
                # corners example: [(0, 200), (0, 250), (400, 250), (400, 200)]
                corners = self._get_obstacle_corners(limits)
                polygon = patches.Polygon(corners, closed=True, 
                                        linewidth=2, edgecolor='black', 
                                        facecolor='black', alpha=0.8, label='Obstacle')
                self.ax.add_patch(polygon)

        # 1. Historical Path (Past moves)
        self.traj_line, = self.ax.plot([], [], c='green', linestyle='-', alpha=0.5, label='Past Path', zorder=8)
        
        # 2. Highlight Latest Move (Segment between last two points)
        self.latest_move_line, = self.ax.plot([], [], c='lime', linewidth=3, label='Latest Move', zorder=9)

        # 3. The Agent (Current actual position)
        self.agent_marker = self.ax.scatter([], [], c='yellow', edgecolors='black', 
                                            marker='*', s=200, label='Agent', zorder=12)

        # Plot start and goal
        self.ax.scatter(*self.start_position, c='blue', label='Start', zorder=5)
        self.ax.scatter(*self.goal_position, c='red', marker='X', s=100, label='Goal (Router)', zorder=5)

        self.ax.legend(loc='upper right', fontsize='small')
        plt.ion()
        plt.show()

    def _world_to_grid_discrete(self, x, y):
        grid_x = int(x / self.cell_length)
        grid_y = int(y / self.cell_width)

        grid_x = max(0, min(self.grid_size - 1, grid_x))
        grid_y = max(0, min(self.grid_size - 1, grid_y))

        return grid_x, grid_y

    def _grid_to_world(self, grid_x, grid_y):
        x = (grid_x + 0.5) * self.cell_length
        y = (grid_y + 0.5) * self.cell_width
        return x, y
    
    def _world_to_grid_continuous(self, x_real, y_real):
        grid_x = (x_real / self.workspace_length) * self.grid_size
        grid_y = (y_real / self.workspace_width) * self.grid_size
        return (grid_x, grid_y)

    def _get_rsi_feedback(self, pos):
        """
        Compute raw RSI signal based on distance to router (goal position).

        Parameters
        ----------
        pos : tuple
            Current position (x, y) in continuous space
        alpha : float
            Signal decay rate

        Returns
        -------
        float
            RSI value (max at goal, decays with distance)
        """
        dx = pos[0] - self.goal_position[0]
        dy = pos[1] - self.goal_position[1]
        distance = math.sqrt(dx * dx + dy * dy)

        # Exponential decay model
        rsi = self.RSI * math.exp(-self.alpha * distance)

        return rsi
    
    def _initialize_visualization(self):
        if self.visualize:
            self.trajectory = [self.start_position]  # store visited positions
            # Precompute RSI map
            # x_scale and y_scale stay the same
            x_scale = (500 - 0) / self.grid_size
            y_scale = (500 - 0) / self.grid_size

            # Centers
            # x_idx 0 is far left, y_idx 0 is far top
            self.x_vals = (np.arange(self.grid_size) + 0.5) * x_scale
            self.y_vals = (np.arange(self.grid_size) + 0.5) * y_scale
            
            X, Y = np.meshgrid(self.x_vals, self.y_vals, indexing='ij')
            dx = X - self.goal_position[0]
            dy = Y - self.goal_position[1]
            distance = np.sqrt(dx**2 + dy**2)
            self.RSI_map = self.RSI * np.exp(-self.alpha * distance)
            self._setup_plot()


    def reset(self, max_attempts=20, random_start=False):
        if random_start:
            for _ in range(max_attempts):
                x = random.uniform(0, self.workspace_length)
                y = random.uniform(0, self.workspace_width)

                if not self._in_obstacle(x, y):
                    self.current_position = (x, y)
                    self.start_position = self.current_position
                    if self.visualize:
                        self.trajectory = [self.start_position]  # reset trajectory
                    rsi = self._get_rsi_feedback(self.current_position)
                    done = False
                    self._initialize_visualization()
                    return (self.current_position[0], self.current_position[1], rsi), done

            raise RuntimeError("Could not find a free start position")
        else:
            self.current_position = self.start_position
        self.current_position_grid = self._world_to_grid_continuous(*self.current_position)
        if self.visualize:
            self.trajectory = [self.start_position]  # reset trajectory
        rsi = self._get_rsi_feedback(self.current_position)
        done = False
        self._initialize_visualization()
        return (self.current_position[0], self.current_position[1], rsi), done

    def _is_goal(self, pos, threshold=12.5):
        dx = pos[0] - self.goal_position[0]
        dy = pos[1] - self.goal_position[1]
        distance = math.sqrt(dx**2 + dy**2)
        return distance <= threshold

    def is_neighbor(self, pos, goal):
        px, py = pos
        gx, gy = goal
        return abs(px - gx) <= 1 and abs(py - gy) <= 1 and pos != goal

    def step(self, action, stat=None):
        xa, ya = action
        # Validate action components
        if xa == 0:
            dx = 0
        elif xa == 1:
            dx = -self.cell_length
        elif xa == 2:
            dx = self.cell_length
        
        if ya == 0:
            dy = 0
        elif ya == 1:
            dy = -self.cell_width
        elif ya == 2:
            dy = self.cell_width

        # Apply movement
        new_x = self.current_position[0] + dx
        new_y = self.current_position[1] + dy

        if not (new_x < 0.0 or new_x > self.workspace_length or new_y < 0.0 or new_y > self.workspace_width):
            self.current_position = (new_x, new_y)

        rsi = self._get_rsi_feedback(self.current_position)

        

        if self._is_goal(self.current_position):
            done = True
        else:
            done = False

        if self.visualize:
            self.trajectory.append(self.current_position)
            self.update_plot(stat)

        self.current_position_grid = self._world_to_grid_continuous(*self.current_position)

        #print(f"After action {action}, current position is {self.current_position} and RSI is {rsi}.")
        return (self.current_position[0], self.current_position[1], rsi), done
"""
# Example usage:
if __name__ == "__main__":
    env = GridEnvironment()
    print("Initial position:", env.current_position)

    # Example action: move down (+x), right (+y), no mark
    action1 = (2, 2, 0)
    obs = env.step(action1)

    # Example action: move up (-x), left (-y), mark
    action2 = (1, 1, 1)
    obs = env.step(action2)
    print(obs)
"""