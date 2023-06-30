import copy

import gym
from sacred import Ingredient
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

env_ingredient = Ingredient("environment")


# noinspection PyUnusedLocal
@env_ingredient.config
def cfg():
    name = "CartPole-v0"  # the name of the task environment (must be registered with gym)
    num_envs = 2  # the number of environments (vectorized environments are used)


@env_ingredient.capture
def create_env(name, env_kwargs=None):
    return gym.make(name, **env_kwargs if env_kwargs else {})


@env_ingredient.capture
def create_vec_env(num_envs, vec_env_start_method="spawn"):
    env = create_env()
    vec_env = vectorize_env(env, num_envs, vec_env_start_method)
    return vec_env, env


def vectorize_env(env, num_envs, vec_env_start_method="spawn"):
    if num_envs > 1:
        env = SubprocVecEnv(env_fns=[lambda: copy.deepcopy(env) for _ in range(num_envs)],
                            start_method=vec_env_start_method)
    else:
        env = DummyVecEnv(env_fns=[lambda: copy.deepcopy(env)])
    env = VecMonitor(env)

    return env
