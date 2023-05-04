from sacred import Ingredient
from stable_baselines3.common.callbacks import EvalCallback

from rl_ingredients.evaluation_environment import create_eval_env, eval_env_ingredient

eval_callback_ingredient = Ingredient("evaluation_process", ingredients=[eval_env_ingredient])


# noinspection PyUnusedLocal
@eval_callback_ingredient.config
def cfg():
    save_best_model = True  # whether to save model weights of best model
    log_files = True  # whether to log agent performance evaluation to file
    num_evals = 10  # the number of evaluations over the course of agent training
    num_eval_episodes = 20  # the number of episodes for each evaluation run
    std_out = True  # whether to log agent performance evaluation to standard out (console)


@eval_callback_ingredient.capture
def create_eval_callback(results_path, num_vectorized_training_steps, algorithm,
                         save_best_model, log_files, num_evals, num_eval_episodes, std_out):
    return EvalCallback(
        eval_env=create_eval_env(results_path=results_path, num_eval_episodes=num_eval_episodes),
        best_model_save_path=results_path if save_best_model else None,
        log_path=results_path if log_files else None,
        eval_freq=max(num_vectorized_training_steps // num_evals, 1),
        n_eval_episodes=num_eval_episodes,
        deterministic=False if algorithm in ("a2c", "ppo") else True,
        render=False,
        verbose=1 if std_out else 0,
    )
