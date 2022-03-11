import time
from pathlib import Path
from typing import Tuple, Union

import numpy as np

from human import Human
from particle_filters.bayes_filter import BootstrapFilter
from particle_filters.particle import Particle
from pizza import Pizza
from query_generator import QueryGenerator
from utils.param_function import ParamFunction
from utils.utils import boltzmann_likelihood as bl
from utils.utils import feature_function, expected_KL_D


class Agent:
    """A generic robot representation."""

    def __init__(
        self,
        sample_pizza: Pizza,
        desired_particle_count: int,
        query_candidate_count: int,
        basis_functions: list,
        query_options: int,
        option_feature_max: int,
        confidence_coefficient: float,
        resampling_threshold: float,
        param_noise_level: float,
        weight_noise_level: float,
        axes_count: int,
        seed: int,
        preference_type: str = "complete",
        query_method: str = "uniform",
    ):
        """Initialize agent.

        ::inputs:
            ::query_candidate_count: How many candidate queries to generate.
            ::query_options: The number of options per query candidate.

        """
        # 1.) Define the space from which candidate queries are drawn:
        self.sample_pizza = sample_pizza
        crust_and_topping_factor = (self.sample_pizza.crust_thickness) + (
            self.sample_pizza.topping_size / 2.0
        )
        self.query_space = [
            -(self.sample_pizza.diameter / 2.0 - crust_and_topping_factor),
            self.sample_pizza.diameter / 2.0 - crust_and_topping_factor,
        ]

        # 2.) Set the remaining user-defined attributes:
        self.param_function = ParamFunction(basis_functions)
        self.query_options = np.arange(query_options)
        self.option_feature_max = option_feature_max
        self.query_candidate_count = query_candidate_count
        self.confidence_coeff = confidence_coefficient
        self.querying_method = query_method
        self.rng = np.random.default_rng(seed)

        # 1.) Instantiate the filter's particles-as-models. NOTE the agent's
        #     posterior belief is maintained within BootstrapFilter:
        try:
            assert sample_pizza is not None

            # 2.) Create the particles:
            start = time.perf_counter()
            particle_set = np.array([])
            for i in range(desired_particle_count):
                pizza_feature_count = self.rng.integers(
                    low=1, high=option_feature_max
                )
                q = self.rng.uniform(
                    self.query_space[0],
                    self.query_space[1],
                    size=(2, pizza_feature_count),
                )
                # 3.) Convert feature lists to pizzas:
                pie = Pizza.from_sample(self.sample_pizza, q)
                pies_params = self.param_function.compute_params(pie)
                p = Particle.from_params(pies_params, id_tag=i)
                particle_set = np.append(particle_set, p)
            stop = time.perf_counter()
            elapsed = stop - start
            print(f"It took {elapsed:.4f}s to generate particles.")

            start = time.perf_counter()
            self.filter = BootstrapFilter(
                particle_set,
                params_noise=param_noise_level,
                weights_noise=weight_noise_level,
                resampling_threshold=resampling_threshold,
            )
            stop = time.perf_counter()
            elapsed = stop - start
            print(f"It took {elapsed:.4f}s to create filter.")
        except AssertionError:
            print(
                "A pizza example is missing; can't instantiate\
                    agent's Filter."
            )
            return
        start = time.perf_counter()
        self.query_generator = QueryGenerator(
            self.rng,
            seed,
            self.sample_pizza,
            query_candidate_count,
            query_options,
            option_feature_max,
            confidence_coefficient,
            query_method,
            preference_type,
        )
        stop = time.perf_counter()
        elapsed = stop - start
        print(f"It took {elapsed:.4f}s to create query generator.")
        # 4.) Set attributes for data:
        self.query_count = 0

        self.ideal_pizza_params = self.param_function.compute_params(
            sample_pizza
        )
        print(
            f"The agent computed the ideal params to be:\n"
            f"{self.ideal_pizza_params}."
        )
        self.ideal_pizza_params = self.ideal_pizza_params.reshape(1, -1)
        self.best_particle = np.nan
        self.best_particle_reward = -1
        self.highest_accumulated_reward = -1
        self.accumulated_reward_found_at = 0
        self.particle_found_at = 0
        self.humans_choices = np.array([])
        self.agents_choices = np.array([])

    def get_human_feedback(self, get_human_input: bool, human_object: Human):
        """Run the loop to acquire human feedback."""
        self.query_count += 1
        particles_params = self.filter.get_particles_params()
        particles_weights = self.filter.get_particles_weights()
        # 1.) Query the human (or simulate human response):
        if get_human_input:
            query_options, chosen_option = self.query_generator.query(
                particles_params,
                particles_weights,
                self.filter.importance_weights,
                self.param_function,
            )
        else:
            query_options = self.query_generator.query(
                particles_params,
                particles_weights,
                self.filter.importance_weights,
                self.param_function,
            )
            agents_choice, agents_rewards = self.choose_from_options(
                query_options
            )
            humans_choice, humans_rewards = human_object.choose_option(
                query_options
            )
            self.agents_choices = np.append(self.agents_choices, agents_choice)
            self.humans_choices = np.append(self.humans_choices, humans_choice)

        # 2.) Compute the query options' respective param values:
        query_params = np.array(
            list(map(self.param_function.compute_params, query_options))
        )
        self.filter.update_belief(
            self.confidence_coeff, humans_choice, query_params
        )

        self.filter.check_degeneracy(self.query_count)

    def choose_from_options(
        self,
        query: Tuple[object, np.ndarray],
        expected_weights: np.ndarray = None,
        expected_parameters: np.ndarray = None,
    ) -> Tuple[int, np.ndarray]:
        """Choose the option which returns the highest reward."""
        q_to_eval = np.array(query, copy=True)
        if type(q_to_eval[0]) == Pizza:
            q_params = np.array(
                list(map(self.param_function.compute_params, q_to_eval))
            )
        else:
            q_params = q_to_eval
        rewards = self.compute_reward(
            q_params, expected_weights, expected_parameters
        )
        choice = np.argmax(rewards)
        return choice, rewards

    def compute_reward(
        self,
        input_params: np.ndarray,
        expected_weights: np.ndarray = None,
        expected_parameters: np.ndarray = None,
    ) -> float:
        """Compute the reward given by some input parameters.

        ::inputs:
            ::input_params: s-by-1-by-m with s the number of objects
                            being evaluated and m the number of
                            parameters each object has.
        """
        # 1.) Get the currently-believed weights and parameters
        #     and compute the features:
        if expected_weights is None:
            expected_weights = self.filter.get_instantaneous_expected_weights()
            expected_parameters = (
                self.filter.get_instantaneous_expected_params()
            )
        # Features are row-wise  w.r.t. the input objects:
        features = feature_function(
            input_params, expected_parameters, normalize=False, combine=False
        ).squeeze()

        # 2.) Now compute the reward:
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        reward = np.exp(
            -self.confidence_coeff * expected_weights.dot(features.T)
        )
        if len(reward.shape) > 2:
            return reward.squeeze()
        else:
            return reward
