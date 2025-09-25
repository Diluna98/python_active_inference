### PyAIF: A Python Package for Active Inference Agents

**PyAIF** is a Python package designed for building and running **Active Inference agents**. It provides a modular and extensible set of tools to define generative models, construct policies, run simulations, and easily integrate with custom environments.

The goal of PyAIF is to make Active Inference research practical, modular, and easily extensible to tasks such as robotics and AI, with a specific focus on **multi-state factors** and **deep temporal applications**.

---

## Installation

### Clone the Repository and Install
First, clone the project repository:
```bash
git clone https://github.com/Diluna98/python_active_inference.git
cd active_inference
pip install .
````

> **NOTE:** This package is under active development. Please expect errors and report any debug information to the maintainer at `dawarn@utu.fi`.

-----

## Core Concepts

Active Inference agents in PyAIF are defined by the following key components: the **Generative Model**, **Policies**, the **ActiveInfAgent** class, and the **Environment**.

### 1\. Generative Model

The generative model defines how an agent **believes** the world works. It is composed of the following key matrices (or arrays) that define the system's dynamics and agent's preferences:

| Parameter | Description |
|:----------|:------------|
| **A** | Likelihood mapping from hidden states to observations. |
| **B** | Transition probabilities between states given actions. |
| **C** | Prior preferences over outcomes (observations). |
| **D** | Prior beliefs about initial states. |

To create a custom generative model, implement a function that returns these matrices and related metadata:

```python
def create_generative_model():
    A = ...
    B = ...
    C = ...
    D = ...
    num_states = [...]       # Dimensions of hidden states
    num_obs = [...]          # Dimensions of observations
    num_controls = [...]     # Dimensions of control factors
    control_fac_idx = [...]  # Indices of controllable state factors
    trial_horizon = 10       # The length of a single trial/simulation run

    return A, B, C, D, num_states, num_obs, num_controls, control_fac_idx, trial_horizon
```

### 2\. Policies

**Policies** are candidate action sequences that the agent evaluates during inference (specifically, when calculating the Expected Free Energy).

You can generate all possible policies using the built-in utility function:

```python
from PyAIF import utils

policies = utils.construct_policies(
    num_states=[...],
    num_controls=[...],
    horizon=trial_horizon - 1,
    control_fac_idx=[...]
)
```

You can optionally write a custom filter to prune invalid, redundant, or impossible action sequences before passing the policies to the agent.

### 3\. ActiveInfAgent

The `ActiveInfAgent` class is the core of the package. It implements the key Active Inference steps:

1.  **Inference** over hidden states (`infer_states`).
2.  **Evaluation** of policies (`infer_policies`).
3.  **Action selection** (`choose_action`).
4.  **Learning** of model parameters (`perform_learning`).

Example initialization:

```python
from PyAIF import ActiveInfAgent

agent = ActiveInfAgent(
    A=A,
    B=B,
    states_dim=num_states,
    obs_dim=num_obs,
    controls_dim=num_controls,
    controlable_states=control_fac_idx,
    trial_length=trial_horizon,
    number_of_msg_passing=int, # Number of message passing iterations for state inference
    trials=trials,             # Total number of trials
    D=D,
    C=C,
    policies=policies,
    policy_pruning=False,      # Option to enable custom policy pruning
    learning_A=False,          # Enable/disable learning for A matrix
    learning_B=False,          # Enable/disable learning for B matrix
    learning_D=True            # Enable/disable learning for D vector
)
```

### 4\. Environments

Environments define the external world the agent interacts with. To use PyAIF, your custom environment must implement a class with a `reset` and a `step` method:

```python
class CustomEnv:
    def reset(self):
        '''Reset environment state and return initial observation.'''
        # ... implementation
        return initial_observation

    def step(self, action):
        '''Execute an action (chosen by the agent) and return a new observation.'''
        # ... implementation
        return new_observation
```

-----

## Typical Workflow

This is the general procedure for setting up and running a simulation with PyAIF:

1.  **Create a generative model**

    ```python
    A, B, C, D, num_states, num_obs, num_controls, control_fac_idx, trial_horizon = create_generative_model()
    ```

2.  **Construct and filter policies**

    ```python
    policies = utils.construct_policies(num_states, num_controls, trial_horizon - 1, control_fac_idx)
    # Optional: custom policy filtering goes here
    ```

3.  **Initialize the agent**

    ```python
    agent = ActiveInfAgent(
        A=A, B=B, states_dim=num_states, obs_dim=num_obs, controls_dim=num_controls,
        controlable_states=control_fac_idx, trial_length=trial_horizon,
        number_of_msg_passing=int, trials=trials, D=D, C=C, policies=policies
    )
    ```

4.  **Set up the environment**

    ```python
    from environment import EnvClass # Assuming EnvClass is your custom environment
    env = EnvClass()
    ```

5.  **Run the simulation loop**

The core simulation is a loop over trials and then a loop over the time horizon (`trial_horizon`):

```python
for trial in range(trials):
    agent.store_parameters()         # Computes complexity of generative model parameters after learning
    agent.normalize_columns()        # Normalizes columns after updating concentration parameters
    agent.initialize_variables()     # Initializes internal variables for the next trial
    obs = env.reset()                # Get the initial observation

    for t in range(trial_horizon):
        # The agent's action is executed in the environment *before* getting the observation for t > 0
        if t > 0:
            obs = env.step(action)

        agent.observations[trial, t, :] = obs      # Store the new observation
        agent.infer_states(trial, t)               # Perform state inference
        _, _ = agent.infer_policies(trial, t)      # Calculate EFE and policy inference
        agent.perform_modal_average(trial, t)      # Perform model average (if applicable)
        chosen_action, _ = agent.choose_action(trial, t) # Sample an action

        if chosen_action is not None:
            # Extract the actions of controllable state factors for the environment
            action = chosen_action[control_fac_idx]

    _,_,_,_ = agent.perform_learning(trial) # Update generative model parameters at the end of the trial
```

-----

## Example Script

An example script is included in the project structure, demonstrating a full simulation:

```
active_inference/
├── examples/
│   └── example_01
│       └── generative_model.py          # Generative model definitions
│       └── main.py                      # Main file running the simulation
│       └── environment.py               # Simulation environment
```

Run the provided example with:

```bash
python examples/handover_task_single_agent/main.py
```

```
```