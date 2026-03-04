from loguru import logger
import sys
import torch
from importlib import import_module
import random
import numpy as np


def attr_from_module(qualified_name: str):
    if "." not in qualified_name:
        raise ValueError("Invalid module name/path name.")
    module_name, attribute_name = qualified_name.rsplit(".", 1)
    module_to_import = import_module(module_name)
    cls_attribute = getattr(module_to_import, attribute_name)
    return cls_attribute


def attr_from_py_path(path: str, endswith: str | None = None) -> type:
    from importlib import import_module

    module_name = path.replace("/", ".")
    # Strip ending
    if module_name.endswith(".py"):
        module_name = module_name[:-3]  # Remove last 3 characters (".py")

    try:
        module = import_module(module_name)
    except ModuleNotFoundError:
        logger.exception(f"Module not found {module_name}")
        raise ValueError(f"Module not found: {module_name}")

    attr_list = [m for m in dir(module) if not endswith or m.endswith(endswith)]
    if len(attr_list) != 1:
        raise ValueError(f"Expected 1 class with endswith={endswith}, got {len(attr_list)}")

    return getattr(module, attr_list[0])


def seed_everything(seed: int = 42, deterministic: bool = True) -> int:
    logger.debug(f"Setting global seed to {seed}...")
    import os

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    return seed

def loguru_excepthook(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logger.opt(exception=(exc_type, exc_value, exc_traceback)).error(
        f"Uncaught exception: {exc_value}"
    )

def setup_global_logger():
    #pprint errors
    sys.excepthook = loguru_excepthook
    from webdataset.utils import pytorch_worker_info
    rank, world_size, *_ = pytorch_worker_info()
    logger.remove()
    if rank == 0:
        # Make the logger with this format the default for all loggers in this package
        logger.configure(
            handlers=[
                {
                    "sink": sys.stdout,
                    "format": "<fg #FFA903>(X-ARES-LLM)</fg #FFA903> [<yellow>{time:YYYY-MM-DD HH:mm:ss}</yellow>] "
                    "<level>{message}</level>",
                    "level": "DEBUG",
                    "colorize": True,
                }
            ]
        )
        logger.level("ERROR", color="<red>")
        logger.level("INFO", color="<white>")
    else:
        logger.disable(__name__)

