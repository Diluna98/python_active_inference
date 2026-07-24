import numpy as np
import random

class SortingEnv:
    
    def __init__(self):
        self.slot_assignment = {
            'safe': ['slot1', 'slot3'],
            'hazardous': ['slot2', 'slot3']
        }
        self.object_types = ['safe', 'hazardous']
        self.slots = ['slot1', 'slot2', 'slot3']
        self.pickup_object = None


    def reset(self):
        self.obs_noise = {
            'picking_slot': 0.01,
            'slot1': 0.01,
            'slot2': 0.01,
            'slot3': 0.01,
            'feedback': 0.01,
        }
        self.slot_status = {s: 'empty' for s in self.slots}
        #if self.pickup_object is None:
        self.pickup_object = random.choice(self.object_types)
        self.feedback = 'positive'
        self.endeff_position = 'ideal'
        self.obj_placement = False  # Flag to indicate if object is placed
        return self._obs_to_index_vector(self._get_observation())

    def _generate_valid_object(self):
        if self.obj_placement == True:
            # Determine all currently empty slots
            free_slots = [s for s, status in self.slot_status.items() if status == 'empty']

            # Identify which objects can go into at least one free slot
            valid_objects = []
            for obj, slots in self.slot_assignment.items():
                if any(slot in free_slots for slot in slots):
                    valid_objects.append(obj)

            # Sample from valid objects if any
            if valid_objects:
                return random.choice(valid_objects)
            else:
                return random.choice(self.object_types)
        else:
            return self.pickup_object


    def _get_valid_slot(self):
        valid = [s for s in self.slot_assignment[self.pickup_object] if self.slot_status[s] == 'empty']
        return random.choice(valid) if valid else 'ideal'

    def _get_invalid_slot(self):
        invalid = [s for s in self.slots if s not in self.slot_assignment[self.pickup_object] or self.slot_status[s] != 'empty']
        if invalid:
            return random.choice(invalid)
        return self._get_valid_slot()

    def _observations(self):
        observations = {
            'picking_slot': self.pickup_object,
            'slot_status': self.slot_status.copy(),
            'feedback': self.feedback,
            'endeff_pos': self.endeff_position
        }
        print(f"observations")
        return observations

    def _get_observation(self):
        def noisy(label, slot_name):
            noise = self.obs_noise.get(slot_name, 0.0)
            return 'not_clear' if np.random.rand() < noise else label

        obs = {
            'picking_slot': noisy(self.pickup_object, 'picking_slot'),
            'slot_status': {
                s: noisy(self.slot_status[s], s)
                for s in self.slots
            },
            'feedback': noisy(self.feedback, 'feedback'),
            'endeff_pos': self.endeff_position
        }
        print(f"obs: {obs}")
        return obs
    
    def _obs_to_index_vector(self, obs):
        # Define categorical mappings
        pickup_cat = ['safe', 'hazardous', 'not_clear']
        vision_cat = ['safe', 'hazardous', 'not_clear', 'empty']
        h_cmd_cat = ['slot1', 'slot2', 'slot3', 'ideal', 'not_clear']
        endeff_pos_cat = ['slot1', 'slot2', 'slot3', 'ideal']
        voice_fb_cat = ['positive', 'negative', 'not_clear']
        metacog_signal_cat = ['obeyed', 'not_obeyed']

        def _get_index(value, categories):
            if value not in categories:
                raise ValueError(f"Value '{value}' not in categories {categories}")
            return categories.index(value)

        # Map each observation entry
        picking_slot_idx = _get_index(obs['picking_slot'], pickup_cat)
        slot1_idx = _get_index(obs['slot_status']['slot1'], vision_cat)
        slot2_idx = _get_index(obs['slot_status']['slot2'], vision_cat)
        slot3_idx = _get_index(obs['slot_status']['slot3'], vision_cat)
        h_fb_idx = _get_index(obs['feedback'], voice_fb_cat)
        endeff_pos_idx = _get_index(obs['endeff_pos'], endeff_pos_cat)

        return [
            picking_slot_idx,
            slot1_idx,
            slot2_idx,
            slot3_idx,
            h_fb_idx,
            endeff_pos_idx
        ]

    
    def step(self, agent_action):
        # Apply agent action
        """
        if agent_action == 'ideal':
            self.endeff_position = 'ideal'
            if any(self.slot_status[slot] == 'empty' for slot in self.slot_assignment[self.pickup_object]):
                self.h_feedback = 'negative'
                self.obj_placement = False
            else:
                self.h_feedback = 'positive'
                self.obj_placement = False

        elif agent_action in self.slots and self.slot_status[agent_action] == 'empty':
            self.endeff_position = agent_action
            if agent_action not in self.slot_assignment[self.pickup_object]:
                self.h_feedback = 'negative'
                self.obj_placement = False
            else:
                self.slot_status[agent_action] = self.pickup_object
                self.h_feedback = 'positive'
                self.obj_placement = True

        else:
            self.endeff_position = agent_action
            self.h_feedback = 'negative'
            self.obj_placement = False
        """
        if agent_action == 'ideal':
            self.endeff_position = agent_action
            self.feedback = 'not_clear'
            self.obj_placement = False


        elif agent_action in self.slot_assignment[self.pickup_object]:
            if self.slot_status[agent_action] == 'empty':
                self.endeff_position = agent_action
                self.feedback = 'positive'
                self.obj_placement = True
                self.slot_status[agent_action] = self.pickup_object
            else:
                self.endeff_position = agent_action
                self.feedback = 'negative'
                self.obj_placement = False

        else:
            self.endeff_position = agent_action
            self.feedback = 'negative'
            self.obj_placement = False

        # Update next object and human command
        self.pickup_object = self._generate_valid_object()

        obs = self._get_observation()

        return self._obs_to_index_vector(obs)
    
""" 
env = SortingEnv()
obs = env.reset()

for t in range(5):
    print(f"\nStep {t}")
    print("Observation:", obs)

    # Get user input for action
    try:
        action_str = input("Enter action index (0:ideal, 1:slot1, 2:slot2, 3:slot3, 4:slot4): ")
        action = int(action_str)
        if action not in [0, 1, 2, 3, 4]:
            print("Invalid action index. Using '0' (ideal) as default.")
            action = 0
    except ValueError:
        print("Invalid input. Using '0' (ideal) as default.")
        action = 0

    obs = env.step(env.slots[action - 1] if action != 0 else 'ideal')  # map index to string
"""  