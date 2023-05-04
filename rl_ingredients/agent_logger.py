from sacred import Ingredient
from stable_baselines3.common.logger import configure

logger_ingredient = Ingredient("agent_logger")


# noinspection PyUnusedLocal
@logger_ingredient.config
def cfg():
    std_out = False  # whether to log agent training to standard out (console)
    log_files = True  # whether to log agent training to file (csv & tensorboard)


@logger_ingredient.capture
def create_logger(std_out, log_path, log_files):
    output_formats = []
    if std_out:
        output_formats.append("stdout")
    if log_files:
        output_formats.extend(["csv", "tensorboard"])
    logger = configure(
        folder=log_path if log_files else None,
        format_strings=output_formats
    )
    return logger
