import logging
import time
from logging import handlers
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

logger = logging.getLogger(__name__)
handler_1 = logging.StreamHandler()
handler_2 = handlers.RotatingFileHandler(
    Path(__file__).parent / Path("logs/agent.log")
)

formatted_output = "{asctime}|{name}|{levelname}|{message}"
formatter_1 = logging.Formatter(formatted_output, style="{")

handler_1.setLevel(logging.WARNING)
handler_2.setLevel(logging.DEBUG)
handler_1.setFormatter(formatter_1)
handler_2.setFormatter(formatter_1)

logger.setLevel(logging.DEBUG)
logger.addHandler(handler_1)
logger.addHandler(handler_2)

"""
At the place of logging:
    logger.<message type>("Something happened here.")

    e.g.

    logger.debug("Something happened here.")
"""


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
        importance_density: str = "uniform",
        query_method: str = "uniform",
        debug: bool = True,
    ):
        """Initialize agent."""
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
        self.resampling_threshold = resampling_threshold
        self.querying_method = query_method
        self.importance_density_type = importance_density
        self.rng = np.random.default_rng(seed)

        # 1.) Instantiate the filter's particles-as-models.
        #     NOTE the agent's posterior belief is maintained
        #     within this BootstrapFilter object:
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

            # To test if the model updates as expected:
            # 1.) un-stub the proceeding block
            # 2.) Stub the resampling procedure at the end of this
            #     file

            # ideal_particle = Particle(
            #    id_tag=particle_set.shape[0] - 1,
            #    param_vals=np.array([[0.55, 2.54]]),
            #    weight_vals=np.array([[0.4, 0.6]]),
            # )
            # particle_set[-1] = ideal_particle

            start = time.perf_counter()
            self.filter = BootstrapFilter(
                particle_set,
                params_noise=param_noise_level,
                weights_noise=weight_noise_level,
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

    def get_human_feedback(
        self,
        i: int,
        get_human_input: bool,
        human_object: Human,
        stepwise: bool,
    ):
        """Run the loop to acquire human feedback."""
        self.query_count += 1
        particles_params = self.filter.get_particles_params()
        particles_weights = self.filter.get_particles_weights()
        # 1.) Query the human (or simulate human response) and get a
        #     visual depiction of that query:
        if get_human_input:
            query_options, chosen_option = self.query_generator.query(
                particles_params,
                particles_weights,
                self.filter.importance_weights,
                self.param_function,
            )
        else:
            # query_options = self.query()
            query_options = self.query_generator.query(
                particles_params,
                particles_weights,
                self.filter.importance_weights,
                self.param_function,
            )
            agents_choice = self.choose_from_options(query_options)
            humans_choice = human_object.choose_from_options(query_options)
            self.agents_choices = np.append(self.agents_choices, agents_choice)
            self.humans_choices = np.append(self.humans_choices, humans_choice)

        # 2.) Compute the query options' respective param values:
        query_params = np.array(
            list(map(self.param_function.compute_params, query_options))
        )
        # 4.) Update robot's belief of human's reward function:
        self.filter.update_belief(
            self.confidence_coeff, humans_choice, query_params
        )

        # 5.) Update the visual:
        resampling_count = None

        latest_belief_plot = None
        latest_query_visual = None
        # 6.) Check if we have enough particles with sufficient
        #     probabilities. If the answer is 'no', resample:
        n_eff = 1 / (np.sum(self.filter.importance_weights**2))
        if n_eff < self.resampling_threshold * self.filter.particle_count:
            print(f"Resampling after query {self.query_count}.")
            resampling_count = self.filter.systematic_resample()

        if stepwise:
            input("Press enter to continue\n")
        return (latest_belief_plot, latest_query_visual)

    def choose_from_options(self, query: Tuple[object, np.ndarray]) -> int:
        """Choose the option which returns the highest reward."""
        q_to_eval = np.array(query, copy=True)
        # if type(q_to_eval[0]) != np.ndarray:
        #    q_to_eval = np.array([p.toppings for p in q_to_eval]).squeeze()
        q_params = np.array(
            list(map(self.param_function.compute_params, q_to_eval))
        )
        rewards = self.compute_reward(q_params)
        choice = np.argmax(rewards)
        return choice

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
        reward = np.exp(
            -self.confidence_coeff * expected_weights.dot(features.T)
        )
        return reward
