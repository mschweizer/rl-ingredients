import os

import pkg_resources
import randomname
from sacred import Ingredient

from rl_ingredients.agent import agent_ingredient
from rl_ingredients.environment import env_ingredient

utilities_ingredient = Ingredient("utilities", ingredients=[agent_ingredient, env_ingredient])

RESULTS_DIR = "results"


# noinspection PyUnusedLocal
@utilities_ingredient.config
def cfg():
    results_dir = RESULTS_DIR
    sb3_log_subdir = f"{RESULTS_DIR}/sb3"


@utilities_ingredient.pre_run_hook
def log_package_dependencies():
    for package, version in get_dependencies():
        utilities_ingredient.add_package_dependency(package_name=package, version=version)


def get_dependencies():
    return list(tuple(str(ws).split()) for ws in pkg_resources.working_set)


@utilities_ingredient.capture
def create_results_path(agent, environment, training_steps, sb3_log_subdir, _run, nickname=None):
    nickname = nickname if nickname else randomname.generate('a/character', 'n/apex_predators')
    log_path = f"{sb3_log_subdir}/" \
               f"{create_prefix(environment=environment, agent=agent, training_steps=training_steps)}_{nickname}"
    log_path_with_sacred(_run, log_path)
    return log_path


def log_path_with_sacred(_run, log_path):
    _run.info["sb3_logs"] = log_path


@utilities_ingredient.capture
def create_prefix(sb3_log_subdir, environment, agent, training_steps):
    run_name = f"env={environment['name']}_algo={agent['algorithm']}_nsteps={training_steps}"

    if os.path.exists(sb3_log_subdir):
        run_number = len([name for name in os.listdir(sb3_log_subdir) if name.startswith(run_name + "_")]) + 1
    else:
        run_number = 1

    return f"{run_name}_{run_number}"


@utilities_ingredient.capture
def log_wrappers(agent, eval_callback, _run):
    _run.info["env_wrappers"] = {}
    _run.info["env_wrappers"]["training"] = get_wrappers(agent.env)
    _run.info["env_wrappers"]["evaluation"] = get_wrappers(eval_callback.eval_env)


def get_wrappers(env):
    env_str_repr = str(env)
    wrapper_names = env_str_repr.split("<")[1:-1]
    return wrapper_names
