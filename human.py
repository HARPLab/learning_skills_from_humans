from __future__ import annotations
from typing import Tuple

import numpy as np

from pizza import Pizza
from utils.param_function import ParamFunction


class Human:
    """Representation of relevant human desires/preferences."""

    def __init__(
        self,
        desired_params: list,
        importance_weights: list,
        basis_functions: list,
        beta_coefficient: float,
    ) -> Human:
        """Initialize human representation."""
        self._desired_params = np.array(desired_params).reshape(1, -1)
        self._importance_weights = np.array(importance_weights).reshape(1, -1)
        self._coeff = beta_coefficient
        self.param_fn = ParamFunction(basis_functions)

    @classmethod
    def from_dicts(
        cls, params: dict, importances: dict, basis_funcs: list, beta: float
    ):
        """Initialize a human representation from dictionaries."""
        param_vals = []
        importance_vals = []
        for b in basis_funcs:
            param_vals.append(params[b])
            importance_vals.append(importances[b])

        human = cls(param_vals, importance_vals, basis_funcs, beta)
        return human

    @property
    def desired_params(self) -> np.ndarray:
        """Return the human's desired feature params."""
        return np.array(self._desired_params, copy=True)

    @property
    def importance_weights(self) -> np.ndarray:
        """Return the human's importance weights."""
        return np.array(self._importance_weights, copy=True)

    @property
    def coeff(self) -> np.ndarray:
        """Return the human's desired feature params."""
        return self._coeff

    def choose_option(
        self, to_choose_from: np.ndarray
    ) -> Tuple[int, np.ndarray]:
        """Select from list of options based on human's preference."""
        # 1.) Compute the differences between each option and
        #     the human's desired features:
        if type(to_choose_from[0]) == Pizza:
            option_params = np.array(
                list(map(self.param_fn.compute_params, to_choose_from))
            )
        else:
            option_params = to_choose_from
        features = self.feature_function(option_params)
        ## 2.) Normalize the features relative across the options:
        # normalized_features = features / np.sum(features, axis=0)
        # if len(normalized_features.shape) <= 1:
        #    normalized_features = normalized_features.reshape(-1, 1)
        ## 3.) Take into account the human's importance weights:
        # weighted_features_compared = self.importance_weights.dot(
        #    normalized_features.T
        # )
        ## 4.) Exponentiate the negation of the results:
        # relative_preference = np.exp(-1 * weighted_features_compared)
        rewards = np.array(list(map(self.compute_reward, features)))
        # 5.) Choose the option which maximizes human's reward:
        choice = np.argmax(rewards)
        # 6.) See if the option is above or below some standard of goodness:
        # meets_standard = check_good_enough(features_compared)
        return choice, rewards

    def compute_reward(self, option: Tuple[Pizza, np.ndarray]) -> float:
        """Compute human's liking given some features."""
        if type(option) == Pizza:
            option_params = self.param_fn(option)
            features = self.feature_function.compute_params(
                option_params
            ).reshape(1, -1)
        else:
            features = option.reshape(1, -1)
        weighted_features = self.importance_weights.dot(features.T)
        reward = np.exp(-self.coeff * weighted_features)
        return reward[0, 0]

    def feature_function(self, query_params: np.ndarray) -> np.ndarray:
        """Subtract and combine.

        ::return:
            ::feature_mats: Per-option features in particle params.
        """
        feature_mats = np.zeros(
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
                np.reshape(query_params[i, :], (1, query_params.shape[1])),
                self.desired_params.shape[0],
                axis=0,
            )
            # 1a.) Compute the element-wise magnitudes between each option
            #      in a query and each possible reward function model:
            feature_mats[i, :, :] = np.abs(q_option - self.desired_params)
        if len(feature_mats.shape) > 2:
            feature_mats = feature_mats.squeeze()
        return feature_mats

    def check_good_enough(pizza_features: np.ndarray) -> bool:
        """Compare this pizza to the human's desired pizza."""
        pass
