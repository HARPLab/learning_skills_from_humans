from typing import Tuple
from logging import handlers

import numpy as np
import logging

# from utils.utils import feature_function
from utils.param_function import ParamFunction

logger = logging.getLogger(__name__)


class Human:
    """Representation of relevant human desires/preferences."""

    def __init__(
        self,
        desired_params: list,
        importance_weights: list,
        basis_functions: list,
        beta_coefficient: float,
    ):
        """Initialize human representation."""
        self.desired_params = np.array(desired_params, copy=True).reshape(
            1, -1
        )
        self.importance_weights = np.array(
            importance_weights, copy=True
        ).reshape(1, -1)
        self.coeff = beta_coefficient
        self.param_fn = ParamFunction(basis_functions)

    @classmethod
    def from_list(
        cls,
        param_vals: np.ndarray,
        importance_vals: np.ndarray,
        basis_funcs: list,
        beta: float,
    ):
        """Initialize a human representation w/ random values."""
        human = cls(param_vals, importance_vals, basis_funcs, beta)
        return human

    def choose_from_options(self, to_choose_from: np.ndarray) -> int:
        """Select from list of options based on human's preference."""
        # 1.) Compute the differences between each option and
        #     the human's desired features:
        option_params = np.array(
            list(map(self.param_fn.compute_params, to_choose_from))
        )
        features = self.feature_function(option_params)
        # 2.) Normalize the features relative across the options:
        normalized_features = features / np.sum(features, axis=0)
        if len(normalized_features.shape) <= 1:
            normalized_features = normalized_features.reshape(-1, 1)
        # 3.) Take into account the human's importance weights:
        weighted_features_compared = self.importance_weights.dot(
            normalized_features.T
        )
        # 4.) Exponentiate the negation of the results:
        relative_preference = np.exp(-self.coeff * weighted_features_compared)
        logger.info(f"Human's relative preference:\n{relative_preference}.")
        # 5.) Choose the option which maximizes human's reward:
        choice = np.argmax(relative_preference)
        # 6.) See if the option is above or below some standard
        #     of goodness:
        # meets_standard = check_good_enough(features_compared)
        return choice

    def get_params(self) -> np.ndarray:
        """Return the human's desired feature params."""
        return np.array(self.desired_params, copy=True)

    def get_weights(self) -> np.ndarray:
        """Return the human's importance weights."""
        return np.array(self.importance_weights, copy=True)

    def feature_function(self, query_params: np.ndarray) -> np.ndarray:
        """Subtract and combine.

        ::return:
            ::difference_mats: Per-option differences in particle params.
        """
        difference_mats = np.zeros(
            (
                query_params.shape[0],
                self.desired_params.shape[0],
                query_params.shape[1],
            )
        )
        # 1.) Separate each option's feature params
        #     and stack them for efficient computation:
        for i in range(query_params.shape[0]):
            q_option = np.repeat(
                np.reshape(
                    query_params[i, :],
                    (1, query_params.shape[1]),
                ),
                self.desired_params.shape[0],
                axis=0,
            )
            # 1a.) Compute the element-wise magnitudes between each
            #      option in a query and each possible reward
            #      function model:
            difference_mats[i, :, :] = np.abs(q_option - self.desired_params)
        difference_mats = difference_mats.squeeze()
        return difference_mats

    # @staticmethod
    # def show_desired_pizza(desired_features: dict):
    #    """Create ideal pizza from parameters."""
    #    # 1.) Get the topping count and centroid:
    #    topping_count = desired_features["object_count"]
    #    pizza_array = np.zeros((2, topping_count))
    #    x_centroid = 0
    #    y_centroid = 0
    #    if np.isin(desired_features.keys(), "x_centroid"):
    #        x_centroid = desired_features["x_centroid"]
    #        y_centroid = desired_features["y_centroid"]
    #    # 2.) Now check what features remain for which
    #    #    we must account:
    #    for k in desired_features.keys():
    #        if k == "x_variance":
    #            while (
    #                np.abs(
    #                    pizza_array[0, :].std() ** 2
    #                    - desired_features["x_variance"]
    #                )
    #                > 10e-4
    #            ):
    #                pizza_array[0, :] = np.random.normal(
    #                    loc=x_centroid,
    #                    scale=np.sqrt(desired_features["x_variance"]),
    #                    size=(1, topping_count),
    #                )
    #        if k == "y_variance":
    #            while (
    #                np.abs(
    #                    pizza_array[1, :].std() ** 2
    #                    - desired_features["y_variance"]
    #                )
    #                > 10e-4
    #            ):
    #                pizza_array[1, :] = np.random.normal(
    #                    loc=y_centroid,
    #                    scale=np.sqrt(desired_features["y_variance"]),
    #                    size=(1, topping_count),
    #                )
    #    return pizza_array

    def check_good_enough(pizza_features: np.ndarray) -> bool:
        """Compare this pizza to the human's desired pizza."""
        pass
