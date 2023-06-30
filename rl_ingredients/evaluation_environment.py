from gym.wrappers import RecordVideo
from sacred import Ingredient

from rl_ingredients.environment import vectorize_env

eval_env_ingredient = Ingredient("evaluation_environment")


# noinspection PyUnusedLocal
@eval_env_ingredient.config
def cfg():
    num_envs = 1  # the number of environments (vectorized environments are used)
    videos = False  # whether to save videos of agent behavior during evaluation
    videos_per_eval = 5  # the number of videos recorded in each evaluation run
    video_dir = "evaluation_videos"  # video directory, relative to root for other sb3 logs/artifacts


@eval_env_ingredient.capture
def create_eval_env(env, num_envs, videos, num_eval_episodes, results_path, video_dir=None, videos_per_eval=None):
    if videos:
        recording_freq = max(num_eval_episodes // videos_per_eval, 1)
        env = RecordVideo(env,
                          video_folder=f"{results_path}/{video_dir}",
                          episode_trigger=lambda episode_count: episode_count % recording_freq == 0
                          )
    vec_env = vectorize_env(env, num_envs)
    return vec_env
