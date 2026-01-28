"""Queens Solver - A Graph Neural Network approach to solving Queens puzzles."""

import logging

__version__ = "1.0.0"

# Set up library logging with NullHandler (let users configure their own handlers)
logging.getLogger(__name__).addHandler(logging.NullHandler())

from queens_solver.config import Config, ModelConfig, TrainingConfig, DataConfig
