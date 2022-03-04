import time
from pathlib import Path
from typing import Tuple, Union

import numpy as np

from pizza import Pizza
from utils.utils import boltzmann_likelihood as bl
from utils.utils import feature_function, expected_KL_D


class QueryGenerator:
    """A generic robot representation."""

    def __init__(
        self,
        random_number_generator,
        seed: int,
        sample_pizza: Pizza,
        query_candidate_count: int,
        query_options: int,
        option_feature_max: int,
        confidence_coefficient: float,
        query_method: str = "uniform",
        preference_type: str = "complete",
    ):
        """Initialize agent.

        ::inputs:
            ::query_candidate_count: How many candidate queries to generate.
            ::query_options: The number of options per query candidate.
            ::basis_functions: The list of basis function keys to pass to a
                               a param-function.

        ::attributes:
            ::self.importance_density: A 2-D column vector which represents
                                       the approximating distribution.

            ::self.posterior_belief: The distribution representing the latest
                                     belief over a human's preferred model.
        """
        # 1.) Define the space from which candidate queries are drawn:
        self.sample_pizza = sample_pizza
        self.preference_type = preference_type
        self._seed = seed
        self.rng = np.random.default_rng(self._seed)
        crust_and_topping_factor = (self.sample_pizza.crust_thickness) + (
            self.sample_pizza.topping_size / 2.0
        )
        self.query_space = np.array(
            [
                -(self.sample_pizza.diameter / 2.0 - crust_and_topping_factor),
                (self.sample_pizza.diameter / 2.0 - crust_and_topping_factor),
            ]
        )

        # 2.) Set the remaining user-defined attributes:
        self.query_options = np.arange(query_options)
        self.option_feature_max = option_feature_max
        self.query_candidate_count = query_candidate_count
        self.confidence_coeff = confidence_coefficient
        self.querying_method = query_method

        self.query_count = 0

    def generate_query(
        self,
        all_particle_params: np.ndarray,
        all_particle_param_weights: np.ndarray,
        each_particles_importance_weights: np.ndarray,
        param_function: callable,
    ) -> np.ndarray:
        """Generate a query of options to present user."""
        # 1.) Generate a list of queries with random options:
        if self.preference_type == "next_slice":
            candidates = self.generate_query_candidates_next_slice(
                self.query_options.shape[0], self.query_candidate_count
            )
        else:
            candidates = self.generate_query_candidates_complete(
                self.query_options.shape[0], self.query_candidate_count
            )

        expected_KL_D_array = np.zeros((1, candidates.shape[0]))
        # 3.) Compute the likelihood of each option within the ith query:
        for i in range(candidates.shape[0]):
            # 4.) Extract param values of each option within the
            #     query candidate:
            query_option_params = np.array(
                list(map(param_function.compute_params, candidates[i]))
            )
            # 5.) Find the difference between each option's computed
            #     params and the params within each model. Then
            #     weight the results by each model's weights:
            weighted_features = feature_function(
                query_option_params,
                all_particle_params,
            )
            option_likelihoods = bl(self.confidence_coeff, weighted_features)
            expected_KL_divergence = expected_KL_D(
                option_likelihoods, each_particles_importance_weights
            )

            expected_KL_D_array[0, i] = expected_KL_divergence

        # 5.) Select the query which yielded the greatest expected
        #     KL divergence and return:
        query_options = candidates[np.argmax(expected_KL_D_array[0, :]), :]
        return query_options

    def generate_query_candidates_next_slice(
        self, option_count: int, query_count: int
    ) -> np.ndarray:
        """Create a set of potential queries.

        ::inputs:
            ::option_count: Number of choices presented to human in a query
            ::query_count: Number of queries to evaluate for proffering
        """
        candidate_array = np.empty((query_count, option_count), dtype=object)

        for i in range(query_count):
            query_candidate = np.array([])
            # 1.) Generate tuples of options such that each option has
            #     THE SAME (feature_count-1) feature-pairs:
            pizza_feature_count = self.rng.integers(
                low=1, high=self.option_feature_max
            )
            q = self.rng.uniform(
                self.query_space[0],
                self.query_space[1],
                size=(2, pizza_feature_count - 1),
            )
            for j in range(option_count):
                # 2.) Distinguish each option with a distinct "next" topping:
                next_top = self.rng.uniform(
                    self.query_space[0], self.query_space[1], size=(2, 1)
                )
                q = np.append(q, next_top, axis=1)
                # 3.) Convert feature lists to pizzas:
                p = Pizza.from_sample(self.sample_pizza, topping_placements=q)
                query_candidate = np.append(query_candidate, p)
                q = np.delete(q, -1, axis=1)
            candidate_array[i, :] = query_candidate

        return candidate_array

    def generate_query_candidates_complete(
        self, option_count: int, query_count: int
    ) -> np.ndarray:
        """Create a set of potential queries.

        ::inputs:
            ::option_count: Number of choices presented to human in a query
            ::query_count: Number of query candidates to create
        """
        candidate_array = np.empty((query_count, option_count), dtype=object)

        for i in range(query_count):
            query_candidate = np.array([])
            pizza_feature_count = self.rng.integers(
                low=1, high=self.option_feature_max
            )
            # 1.) Generate tuples of options such that each option has
            #     pizza_feature_count feature-pairs:
            for j in range(option_count):
                q = self.rng.uniform(
                    self.query_space[0],
                    self.query_space[1],
                    size=(2, pizza_feature_count),
                )
                # 2.) Convert feature lists to pizzas:
                p = Pizza.from_sample(self.sample_pizza, topping_placements=q)
                query_candidate = np.append(query_candidate, p)
            candidate_array[i, :] = query_candidate

        return candidate_array

    def query(
        self,
        all_particle_params: np.ndarray,
        all_particle_param_weights: np.ndarray,
        each_particles_importance_weights: np.ndarray,
        param_function: callable,
        get_human_input=False,
    ) -> Union[Tuple[np.ndarray, int], np.ndarray]:
        """Prompt user for preference."""
        self.query_count += 1
        t_start = time.perf_counter()
        choice_options = self.generate_query(
            all_particle_params,
            all_particle_param_weights,
            each_particles_importance_weights,
            param_function,
        )
        t_stop = time.perf_counter()
        query_gen_time = t_stop - t_start
        print(f"Generated query {self.query_count} in {query_gen_time:.5f}.")
        if get_human_input:
            # Interacting with real human
            humans_choice = input(
                f"Which of these is best in your mind?: {choice_options}"
            )
            return choice_options, humans_choice
        # Simulating interaction with real human
        return choice_options
