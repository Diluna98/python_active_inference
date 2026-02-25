import math

class GridEnvironment:
    def __init__(self, size, workspace_length=420, workspace_width=297, s_x=12.6, s_y=270.27, g_x=382.2, g_y=14.85):
        self.grid_size = size
        self.workspace_length = workspace_length
        self.workspace_width = workspace_width
        self.cell_length = self.workspace_length / self.grid_size
        self.cell_width = self.workspace_width / self.grid_size
        self.perfect_score_r = 5.94
        self.start_position = (s_x, s_y)
        self.goal_position = (g_x, g_y)
        self.current_position = self.start_position

    def world_to_grid(self, x, y):
        grid_x = int(x / self.cell_length)
        grid_y = int(y / self.cell_width)

        grid_x = max(0, min(self.grid_size - 1, grid_x))
        grid_y = max(0, min(self.grid_size - 1, grid_y))

        return grid_x, grid_y

    def grid_to_world(self, grid_x, grid_y):
        x = (grid_x + 0.5) * self.cell_length
        y = (grid_y + 0.5) * self.cell_width
        return x, y



    def reset(self):
        self.current_position = self.start_position
        gridx, gridy = self.world_to_grid(self.current_position[0], self.current_position[1])
        feedback_id = 0
        

        return (gridx, gridy, feedback_id)


    def is_neighbor(self, pos, goal):
        px, py = pos
        gx, gy = goal
        return abs(px - gx) <= 1 and abs(py - gy) <= 1 and pos != goal

    def step(self, action):
        xa, ya, mark_flag = action
        # Validate action components
        if xa == 0:
            dx = 0
        elif xa == 1:
            dx = -self.cell_length
        elif xa == 2:
            dx = self.cell_length
        else:
            raise ValueError(f"dx must be 0, 1, 2, or 3. current dx is {xa}")
        
        if ya == 0:
            dy = 0
        elif ya == 1:
            dy = -self.cell_width
        elif ya == 2:
            dy = self.cell_width
        else:
            raise ValueError(f"dy must be 0, 1, or 2. current dy is {ya}")
        if mark_flag not in [0, 1]:
            raise ValueError("mark_flag must be 0 (do_nothing) or 1 (mark)")

        # Apply movement
        new_x = max(0.0, min(self.workspace_length, self.current_position[0] + dx))
        new_y = max(0.0, min(self.workspace_width, self.current_position[1] + dy))

        self.current_position = (new_x, new_y)

        if mark_flag == 1:
            dx = self.current_position[0] - self.goal_position[0]
            dy = self.current_position[1] - self.goal_position[1]
            distance = math.sqrt(dx * dx + dy * dy)

            if distance <= self.perfect_score_r:
                feedback = 'perfect'
            elif distance <= 2 * self.perfect_score_r:
                feedback = 'good'
            else:
                feedback = 'bad'
        else:
            feedback = 'nothing'

        if feedback == 'nothing':
            feedback_id = 0
        elif feedback == 'perfect':
            feedback_id = 1
        elif feedback == 'good':
            feedback_id = 2
        elif feedback == 'bad':
            feedback_id = 3

        print(f"After action {action}, current position is {self.current_position} and feedback is {feedback}.")

        new_gridx, new_gridy = self.world_to_grid(new_x, new_y)
        return (new_gridx, new_gridy, feedback_id)
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