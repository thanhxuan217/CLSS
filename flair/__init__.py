import torch
import os

# global variable: cache_root
cache_root = os.path.expanduser(os.path.join("~", ".flair"))

# global variable: device
device = None
if torch.cuda.is_available():
    try:
        # Test that CUDA kernels actually work on this GPU
        _test = torch.zeros(1, device=torch.device("cuda:0"))
        del _test
        device = torch.device("cuda:0")
    except Exception as _e:
        import logging as _logging
        _logging.warning(
            f"CUDA is available but not functional (kernel incompatibility): {_e}. "
            f"Falling back to CPU."
        )
        device = torch.device("cpu")
else:
    device = torch.device("cpu")

from . import data
from . import models
from . import visual
from . import trainers
from . import nn

import logging.config

__version__ = "0.4.3"

logging.config.dictConfig(
    {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {"standard": {"format": "%(asctime)-15s %(message)s"}},
        "handlers": {
            "console": {
                "level": "INFO",
                "class": "logging.StreamHandler",
                "formatter": "standard",
                "stream": "ext://sys.stdout",
            }
        },
        "loggers": {
            "flair": {"handlers": ["console"], "level": "INFO", "propagate": False}
        },
        "root": {"handlers": ["console"], "level": "WARNING"},
    }
)

logger = logging.getLogger("flair")
