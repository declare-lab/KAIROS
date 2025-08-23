import os
import sys
from logging import Logger
from typing import Optional

import colorlog
import datasets
import transformers
from MAS.utils.arguments import ScriptArguments


def setup_logger(training_args: Optional[transformers.TrainingArguments] = None, script_args: Optional[ScriptArguments] =  None) -> Logger:
    """
    Setup the logger. If training_args is None, use default log level.
    """
    fmt_string = '%(log_color)s %(asctime)s - %(levelname)s - %(message)s'
    log_colors = {
        'DEBUG': 'white',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'purple'
    }

    if training_args is not None:
        log_level = training_args.get_process_log_level()
        datasets.utils.logging.set_verbosity(log_level)
        transformers.utils.logging.set_verbosity(log_level)
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
        
        # Configure wandb if enabled and training_args is not None
        if "wandb" in training_args.report_to:
            if script_args is not None:
                init_wandb_training(script_args)
            else:
                print("Warning: script_args is None, cannot initialize wandb.")
    else:
        log_level = "INFO"

    colorlog.basicConfig(
        log_colors=log_colors,
        format=fmt_string,
        handlers=[colorlog.StreamHandler(sys.stdout)],
        level=log_level
    )

    logger = colorlog.getLogger(__name__)

    return logger


def init_wandb_training(script_args) -> None:
    """
    Helper function for setting up Weights & Biases logging tools.
    """
    if script_args.wandb_entity is not None:
        os.environ["WANDB_ENTITY"] = script_args.wandb_entity
    if script_args.wandb_project is not None:
        os.environ["WANDB_PROJECT"] = script_args.wandb_project