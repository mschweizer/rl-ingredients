import os

import pkg_resources
import randomname
from sacred import Ingredient

from rl_ingredients.agent import agent_ingredient
from rl_ingredients.environment import env_ingredient
from rl_ingredients.observer import CustomFileStorageObserver

utilities_ingredient = Ingredient("utilities", ingredients=[agent_ingredient, env_ingredient])

RESULTS_DIR = "results"


@utilities_ingredient.pre_run_hook
def log_package_dependencies():
    for package, version in get_dependencies():
        utilities_ingredient.add_package_dependency(package_name=package, version=version)


def get_dependencies():
    return list(tuple(str(ws).split()) for ws in pkg_resources.working_set)


@utilities_ingredient.capture
def get_or_create_log_path(agent, env_id, training_steps, _run, _log, base_log_dir="results", nickname=None):
    observer = get_custom_file_storage_observer(_run)
    if observer:
        base_log_dir = observer.basedir

    log_dir_name = create_log_dir_name(agent, env_id, nickname, training_steps, base_log_dir)

    if observer:
        log_path = observer.get_log_path_with_new_name(new_name=log_dir_name, logger=_log)
    else:
        log_path = f"{base_log_dir}/{log_dir_name}"
        log_path_with_sacred(_run, log_path)

    return log_path


def log_path_with_sacred(_run, log_path):
    _run.info["sb3_logs"] = log_path


def get_custom_file_storage_observer(_run):
    registered_file_storage_observers = \
        [observer for observer in _run.observers if isinstance(observer, CustomFileStorageObserver)]
    if len(registered_file_storage_observers) > 0:
        return registered_file_storage_observers[0]


def create_log_dir_name(agent, env_id, nickname, training_steps, base_log_dir):
    prefix = create_prefix(base_log_dir=base_log_dir, env_id=env_id, agent=agent,
                           training_steps=training_steps)
    nickname = nickname if nickname else randomname.generate('a/character', 'n/apex_predators')
    return f"{prefix}_{nickname}"


@utilities_ingredient.capture
def create_prefix(base_log_dir, env_id, agent, training_steps):
    run_name = f"env={env_id}_algo={agent['algorithm']}_nsteps={training_steps}"

    if os.path.exists(base_log_dir):
        run_number = len([name for name in os.listdir(base_log_dir) if name.startswith(run_name + "_")]) + 1
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
