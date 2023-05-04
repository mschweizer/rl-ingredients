from sacred import Experiment
from sacred.observers import FileStorageObserver

from rl_ingredients.agent import agent_ingredient, create_agent, get_algorithm_id
from rl_ingredients.agent_logger import logger_ingredient, create_logger
from rl_ingredients.environment import env_ingredient, create_vec_env
from rl_ingredients.evaluation_callback import create_eval_callback, eval_callback_ingredient
from rl_ingredients.utils import utilities_ingredient, create_results_path, log_wrappers, RESULTS_DIR

experiment = Experiment("experiment",
                        ingredients=[agent_ingredient, env_ingredient, logger_ingredient,
                                     eval_callback_ingredient, utilities_ingredient])
experiment.observers.append(FileStorageObserver(basedir=f"{RESULTS_DIR}/sacred"))


# noinspection PyUnusedLocal
@experiment.config
def cfg():
    training_steps = 10_000  # the number of agent training steps


@experiment.pre_run_hook
def add_observer(utilities):
    experiment.observers.append(FileStorageObserver(basedir=f"{utilities['results_dir']}/sacred"))


@experiment.automain
def run(training_steps):
    log_path = create_results_path(training_steps=training_steps)

    env = create_vec_env()
    agent = create_agent(env)

    agent.set_logger(create_logger(log_path=log_path))

    eval_callback = create_eval_callback(results_path=log_path,
                                         num_vectorized_training_steps=training_steps // agent.n_envs,
                                         algorithm=get_algorithm_id())  # TODO: get algo directly from agent object

    log_wrappers(agent, eval_callback)

    agent.learn(total_timesteps=training_steps, callback=[eval_callback])

    agent.save(log_path + "/" + "final_model.zip")
