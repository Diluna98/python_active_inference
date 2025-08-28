import numpy as np
import random

class SortingEnv:
    
    def __init__(self, reliability=100):
        self.reliability = reliability  # human reliability percentage
        self.slot_assignment = {
            'safe': ['slot1', 'slot3'],
            'hazardous': ['slot2', 'slot3']
        }
        self.object_types = ['safe', 'hazardous']
        self.slots = ['slot1', 'slot2', 'slot3']
        self.pickup_object = None
        self.metacog_sig = 'obeyed'  # Initialize metacognition signal
        self.h_command_memory = 'ideal'  # Initialize human command memory
        #self.reset()


    def reset(self):
        self.obs_noise = {
            'picking_slot': 0.01,
            'slot1': 0.01,
            'slot2': 0.01,
            'slot3': 0.01,
            'voice_command': 0.00,
            'feedback_command': 0.10,
            'endeff_pos': 0.00,
            'h_command_memory': 0.00
        }
        self.slot_status = {s: 'empty' for s in self.slots}
        self.h_trustworthiness = self.reliability
        if self.pickup_object is None:
            self.pickup_object = random.choice(self.object_types)
        self.h_command = self._generate_human_command()
        self.h_feedback = 'positive'
        self.endeff_position = 'ideal'
        self.h_command_memory = 'ideal'
        #self.slot_status['slot1'] = 'fragile'
        #self.slot_status['slot2'] = 'safe'
        #self.slot_status['slot3'] = 'hazardous'
        #self.slot_status['slot4'] = 'fragile'
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

    
    def _generate_human_command(self):
        if np.random.rand() <= self.reliability / 100.0:
            return self._get_valid_slot()
        else:
            return self._get_invalid_slot()


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
            'h_command': self.h_command,
            'h_trustworthiness': self.h_trustworthiness,
            'h_feedback': self.h_feedback,
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
            'h_command': self.h_command,
            'h_feedback': noisy(self.h_feedback, 'feedback_command'),
            'endeff_pos': self.endeff_position,
            'h_command_memory': noisy(self.h_command_memory, 'h_command_memory'),
            'metacog_signal': self.metacog_sig
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
        h_cmd_idx = _get_index(obs['h_command'], h_cmd_cat)
        h_fb_idx = _get_index(obs['h_feedback'], voice_fb_cat)
        endeff_pos_idx = _get_index(obs['endeff_pos'], endeff_pos_cat)
        h_command_memory_idx = _get_index(obs['h_command_memory'], h_cmd_cat)    
        metacog_idx = _get_index(obs['metacog_signal'], metacog_signal_cat)

        return [
            picking_slot_idx,
            slot1_idx,
            slot2_idx,
            slot3_idx,
            h_cmd_idx,
            h_fb_idx,
            endeff_pos_idx,
            h_command_memory_idx,
            metacog_idx
        ]
    
    def get_metacog_signal(self, action, command):
        if action == command:
            self.metacog_sig = 'obeyed'
        else:  
            self.metacog_sig = 'not_obeyed'

    
    def step(self, agent_action):
        # Apply agent action
        self.get_metacog_signal(agent_action, self.h_command)
        self.h_command_memory = self.h_command
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
        if agent_action == self.h_command:
            self.endeff_position = agent_action
            self.h_feedback = 'positive'
            self.slot_status[agent_action] = self.pickup_object
            self.obj_placement = True
        else:
            self.endeff_position = agent_action
            self.h_feedback = 'negative'
            if agent_action is not 'ideal' and self.slot_status[agent_action] == 'empty':
                self.slot_status[agent_action] = self.pickup_object
                self.obj_placement = True
            else:
                self.obj_placement = False

        
        # Update next object and human command
        self.pickup_object = self._generate_valid_object()
        self.h_command = self._generate_human_command()

        obs = self._get_observation()

        return self._obs_to_index_vector(obs)
    
"""  
env = SortingEnv(reliability=100)
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