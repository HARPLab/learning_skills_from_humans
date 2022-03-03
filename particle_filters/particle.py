from __future__ import annotations

import logging
from logging import handlers
from pathlib import Path
from typing import Tuple, Union

import numpy as np

from utils.utils import mean_normalize, feature_function

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler_1 = logging.StreamHandler()
handler_2 = handlers.RotatingFileHandler(
    Path(__file__).parent / Path("../logs/particle.log")
)

formatted_output = "{asctime}|{name}|{levelname}|{message}"
formatter_1 = logging.Formatter(formatted_output, style="{")

handler_1.setLevel(logging.WARNING)
handler_2.setLevel(logging.DEBUG)
handler_1.setFormatter(formatter_1)
handler_2.setFormatter(formatter_1)

logger.addHandler(handler_1)
logger.addHandler(handler_2)

"""
At the place of logging:
    logger.<message type>("Something happened here.")

    e.g.

    logger.debug("Something happened here.")
"""


class Particle:
    """Particle-as-model; to be used in a particle filter."""

    def __init__(
        self, id_tag: int, param_vals: np.ndarray, weight_vals: np.ndarray
    ):
        """Instantiate a particle as a reward function configuration."""
        self._id = id_tag
        self._params = param_vals.reshape(1, -1)
        self._weights = weight_vals.reshape(1, -1)
        self._weights = self._weights / np.sum(self._weights)
        self._param_count = self._params.shape[1]

    @classmethod
    def from_params(cls, incoming_params: np.ndarray, id_tag: int) -> Particle:
        """Instantiate a particle based on pre-computed params."""
        incoming_weights = np.random.rand(1, incoming_params.shape[0])
        particle = cls(
            id_tag, param_vals=incoming_params, weight_vals=incoming_weights
        )
        return particle

    @classmethod
    def from_particle(cls, sample_particle: Particle, id_tag: int) -> Particle:
        """Instantiate a particle based on sample_particle."""
        particle = cls(
            id_tag,
            param_vals=sample_particle.params,
            weight_vals=sample_particle.weights,
        )
        return particle

    @property
    def params(self) -> np.ndarray:
        """Return this particle's feature parameters."""
        return np.array(self._params, copy=True)

    @params.setter
    def params(self, new_values: np.ndarray) -> None:
        """Reset the param values."""
        self._params = np.array(new_values, copy=True).reshape(1, -1)

    @property
    def weights(self) -> np.ndarray:
        """Return this particle's importance weights."""
        return np.array(self._weights, copy=True)

    @weights.setter
    def weights(self, new_values: np.ndarray) -> None:
        """Reset the weight values."""
        new_values = new_values / np.sum(new_values)
        self._weights = np.array(new_values, copy=True).reshape(1, -1)

    @property
    def param_count(self) -> int:
        """Return this particle's param count."""
        return self._param_count

    @param_count.setter
    def param_count(self, count: int) -> int:
        """Return this particle's param count."""
        self._param_count = count

    def get_id(self) -> int:
        """Return this particle's ID."""
        return self._id

    def compute_reward(self, input_params: np.ndarray) -> float:
        """Compute this particle's reward."""
        # 1.) Compute the params:
        params = feature_function(input_params, self._params)
        # 2.) Compute the linear combo. of this particle's importance weights
        #     and these features:
        rwd = self._weights.dot(params.T)
        return rwd
