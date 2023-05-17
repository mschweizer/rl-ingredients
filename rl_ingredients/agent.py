import yaml
from sacred import Ingredient
from stable_baselines3 import PPO, A2C, DQN

agent_ingredient = Ingredient("agent")
ALGORITHMS = {"dqn": DQN, "a2c": A2C, "ppo": PPO}


# noinspection PyUnusedLocal
@agent_ingredient.config
def cfg():
    algorithm = "dqn"  # the reinforcement learning algorithm used by the agent
    try:
        hyperparams = read_hyperparameter_defaults(algorithm)
    except FileNotFoundError:
        hyperparams = {"policy": "MlpPolicy"}


@agent_ingredient.pre_run_hook
def assert_algorithm(algorithm):
    assert algorithm in ALGORITHMS, \
        f"RL algorithm {algorithm} is not supported. Please choose any of {ALGORITHMS}."


@agent_ingredient.capture
def create_agent(env, algorithm, hyperparams):
    agent = ALGORITHMS[algorithm](env=env, **hyperparams)
    return agent


@agent_ingredient.capture
def get_algorithm_id(algorithm):  # TODO: remove
    return algorithm


def read_hyperparameter_defaults(algorithm, defaults_path="hyperparams_defaults"):
    filename = f"{defaults_path}/{algorithm}.yml"
    with open(filename) as f:
        print(f"Loading hyperparameters from: {filename}")
        hyperparams = yaml.safe_load(f)
    return hyperparams
