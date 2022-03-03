import numpy as np
from utils.utils import boltzmann_likelihood as bl
from utils.utils import feature_function, mean_normalize, rescale

from particle_filters.particle import Particle


class BootstrapFilter:
    """A generic particle filter."""

    def __init__(
        self, particles: np.ndarray, params_noise: float, weights_noise: float
    ):
        """Instantiate a BoostrapFilter."""
        # 1.) Basic attributes:
        self.particle_models = particles
        self.particle_count = particles.shape[0]
        self.importance_weights = np.full(
            (1, self.particle_count),
            fill_value=(1.0 / particles.shape[0]),
            dtype=float,
        )
        self.params_noise = params_noise
        self.weights_noise = weights_noise
        self.expected_params = np.zeros(
            (1, self.particle_models[0].param_count)
        )
        self.expected_weights = np.zeros(
            (1, self.particle_models[0].param_count)
        )
        self.particle_param_count = self.particle_models[0].param_count
        # 4.) Instantiate data-tracking attributes:
        self.update_count = 0
        self.m_most_probable = int(self.particle_count)  # / 5.0)
        self.expectation_every_n = 5

    def update_belief(
        self,
        confidence_coeff: float,
        query_choice: int,
        input_param_values: np.ndarray,
    ) -> None:
        """Compute importance weights.

        inputs::
            ::input_param_values: The param values to compare
                                    against each particle. Can be
                                    a 2D array or larger depending
                                    upon the observation model at
                                    play.
        """
        # 1.) Do some preprocessing:
        models_params = self.get_particles_params()

        # 2.) Separate each option's param values
        #      and stack them for efficient computation:
        weighted_param_diffs = feature_function(
            input_param_values, models_params
        )
        # 3.) Compute the likelihood: P(query_choice|particle_model):
        observation_model = bl(confidence_coeff, weighted_param_diffs)
        new_weights = np.array([observation_model[query_choice, :]])
        # 4.) Update the agent's posterior belief as likelihood * prior:
        self.importance_weights = new_weights * self.importance_weights
        # 5.) Normalize the posterior:
        self.importance_weights = self.importance_weights / np.sum(
            self.importance_weights
        )
        if np.any(np.isnan(self.importance_weights)):
            print(f"NaN at: {np.argwhere(np.isnan(self.importance_weights))}.")
        # 6.) Compute the expected params and weights given the
        #     latest particle models:
        expected_params = self.get_instantaneous_expected_params(
            count=self.m_most_probable
        )
        self.expected_params = np.vstack(
            (self.expected_params, expected_params)
        )
        expected_weights = self.get_instantaneous_expected_weights(
            count=self.m_most_probable
        )
        self.expected_weights = np.vstack(
            (self.expected_weights, expected_weights)
        )
        if (
            self.update_count > self.expectation_every_n
            and self.update_count % self.expectation_every_n == 0
        ):
            expected_params = self.expected_params_last_n(
                self.expectation_every_n
            )
            expected_weights = self.expected_weights_last_n(
                self.expectation_every_n
            )
        self.update_count += 1

    def systematic_resample(self, with_noise: bool = True) -> np.ndarray:
        """Adjust agent's posterior_belief via resampling."""
        # 1.) Collect n=particle_count samples from prior
        #     weighted by importance.
        resampled_particle_set = np.array([])
        resampled_count = np.zeros((1, self.particle_count))
        c_s = np.cumsum(self.importance_weights)
        u = np.random.uniform(low=0, high=1 / self.particle_count, size=1)
        i = 0
        for j in range(self.particle_count):
            while u > c_s[i]:
                i += 1
            new_part = Particle.from_particle(
                self.particle_models[i], id_tag=j
            )
            resampled_particle_set = np.append(
                resampled_particle_set, new_part
            )
            resampled_count[0, new_part.get_id()] = (
                resampled_count[0, new_part.get_id()] + 1
            )
            u += 1 / self.particle_count

        # 2.) Delete old particles:
        del self.particle_models

        # 3.) Set new particles-as-models and add
        #     some noise:
        self.particle_models = resampled_particle_set
        if with_noise:
            self.add_normed_noise()
        # 4.) Reset the weights with uniform value:
        self.importance_weights = np.full_like(
            self.importance_weights, 1.0 / self.particle_count, dtype=float
        )
        return resampled_count

    def multinomial_resample(self) -> np.ndarray:
        """Adjust agent's posterior_belief via resampling."""
        # 1.) Collect n=particle_count samples from prior
        #     weighted by importance.
        resampled_particle_set = np.array([])
        resampled_count = np.zeros((1, self.particle_count))
        for i in range(self.particle_count):
            part = np.random.choice(
                a=self.particle_models,
                size=1,
                replace=True,
                p=self.importance_weights[0, :],
            )[0]
            new_part = Particle.from_particle(part, id_tag=i)
            resampled_particle_set = np.append(
                resampled_particle_set, new_part
            )
            resampled_count[0, part.get_id()] = (
                resampled_count[0, part.get_id()] + 1
            )

        # 2.) Delete old particles:
        del self.particle_models

        # 3.) Set new particles-as-models and add
        #     some noise:
        self.particle_models = resampled_particle_set
        self.add_normed_noise()
        # 4.) Reset the weights with uniform value:
        self.importance_weights = np.full_like(
            self.importance_weights, 1.0 / self.particle_count, dtype=float
        )
        return resampled_count

    def get_instantaneous_expected_params(
        self, count: int = None
    ) -> np.ndarray:
        """Get E[param values] at this time-step."""
        if count is None:
            count = self.particle_count
        # 1.) Get the params:
        particles_params = self.get_particles_params().T
        # 2.) Get the indices of the n most probable particles:
        n_inds = np.argpartition(self.importance_weights[0, :], -count)[
            -count:
        ]
        # 3.) Get their weights and normalize them w.r.t. each other:
        normed_probs = self.importance_weights[0, n_inds] / np.sum(
            self.importance_weights[0, n_inds]
        )
        # 4.) Compute the expected values of these n most-probable particles:
        expectation = np.sum(
            normed_probs * particles_params[:, n_inds], axis=1
        ).reshape(1, -1)
        return expectation

    def get_instantaneous_expected_weights(
        self, count: int = None
    ) -> np.ndarray:
        """Get E[param importance weights] at this time-step."""
        if count is None:
            count = self.particle_count
        # 1.) Get the weights:
        particles_weights = self.get_particles_weights().T
        # 2.) Get the indices of the n most probable particles:
        n_inds = np.argpartition(self.importance_weights[0, :], -count)[
            -count:
        ]
        # 3.) Get their weights and normalize them w.r.t. each other:
        normed_probs = self.importance_weights[0, n_inds] / np.sum(
            self.importance_weights[0, n_inds]
        )
        # 4.) Compute the expected values of these n most-probable particles:
        expectation = np.sum(
            normed_probs * particles_weights[:, n_inds], axis=1
        ).reshape(1, -1)
        return expectation

    def expected_params_last_n(self, n: int) -> np.ndarray:
        """Return E[params] based on the last 'n' samples."""
        summed_expectation = np.sum(self.expected_params[-n::, :], axis=0)
        expectation = summed_expectation / n
        return expectation

    def expected_weights_last_n(self, n: int) -> np.ndarray:
        """Return E[weights] based on the last 'n' samples."""
        summed_expectation = np.sum(self.expected_weights[-n::, :], axis=0)
        expectation = summed_expectation / n
        return expectation

    def get_particles_params(self) -> np.ndarray:
        """Return each particle's parameters."""
        particles_params = np.array(
            [part.params for part in self.particle_models]
        ).squeeze()
        if len(particles_params.shape) <= 1:
            particles_params = particles_params.reshape(-1, 1)

        return particles_params

    def get_particles_weights(self) -> np.ndarray:
        """Return each particle's param weights."""
        particles_weights = np.array(
            [part.weights for part in self.particle_models]
        ).squeeze()
        if len(particles_weights.shape) <= 1:
            particles_weights = particles_weights.reshape(-1, 1)

        return particles_weights

    def add_normed_noise(self) -> None:
        """Mean-normalize the params before adding noise.

        Note we don't rescale the weights; performance degrades.
        """
        noisy_params = np.zeros(
            (self.particle_count, self.particle_param_count)
        )
        noisy_weights = np.zeros(
            (self.particle_count, self.particle_param_count)
        )
        # 1.) Get each particle's param values and param weights:
        particles_params = self.get_particles_params()
        particles_weights = self.get_particles_weights()
        for i in range(self.particle_param_count):
            # 2.) Mean-normalize the ith param (by considering all particles):
            normed_param, param_mean, param_std = mean_normalize(
                particles_params[:, i]
            )
            noise = (
                np.random.normal(
                    loc=0, scale=1, size=particles_params[:, i].shape
                )
                * self.params_noise
            )
            noisy_params[:, i] = normed_param + noise
            noisy_params[:, i] = rescale(
                noisy_params[:, i], param_mean, param_std
            )
            # 3.) Clip the params at 0:
            noisy_params[:, i] = np.where(
                noisy_params[:, i] < 0, 0, noisy_params[:, i]
            )
            # 4.) Mean-normalize the ith weight (by considering
            #     all particles):
            normed_weight, weight_mean, weight_std = mean_normalize(
                particles_weights[:, i]
            )
            noise = (
                np.random.normal(
                    loc=0, scale=1, size=particles_weights[:, i].shape
                )
                * self.weights_noise
            )
            noisy_weights[:, i] = particles_weights[:, i] + noise
            # 5.) Clip the weights at 0:
            noisy_weights[:, i] = np.where(
                noisy_weights[:, i] < 0, 0, noisy_weights[:, i]
            )
        for i in range(self.particle_count):
            self.particle_models[i].params = noisy_params[i, :]
            self.particle_models[i].weights = noisy_weights[i, :]
