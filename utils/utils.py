import logging
import time
from logging import handlers
from pathlib import Path
from typing import Tuple, Union

import matplotlib.animation as animate
import numpy as np
from pizza import Pizza


def KL_D(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Compute KL-Divergence.

    ::inputs:
        ::Q: Reference/'true' distribution we want to approximate.
        ::P: Distribution(s) used to approximate Q.

    ::return:
        ::results: 1-D array of numeric KL divergence(s)
                   (ie. relative entropy of Q to P).
    """
    results = np.array([])

    for i in range(Q.shape[0]):
        # 3.) Compute the cross-entropy between P and Q:
        # logger.info(f"P[:,0]=\n{P[:,0]}\nQ[:,i]:\n{Q[:,i]}")
        result = np.sum(Q[i, :] * np.log(Q[i, :] / P[0, :]), axis=0)

        results = np.append(results, result)
    return results


def expected_KL_D(
    query_option_likelihoods: np.ndarray, current_posterior: np.ndarray
) -> np.ndarray:
    """Compute the expected KL divergence.

    ::inputs:

    ::return:
    """
    posterior = current_posterior
    posterior = np.repeat(
        current_posterior, query_option_likelihoods.shape[0], axis=0
    )
    # 1.) Compute the hypothesized posteriors for the ith query:
    hypothesized_posteriors = query_option_likelihoods * posterior

    # 2.) Compute each choice's probability of being selected
    choice_probabilities = np.sum(query_option_likelihoods, axis=1)
    choice_probabilities = choice_probabilities / np.sum(choice_probabilities)

    # 3.) Normalize the hypotheses:
    hypothesized_posteriors = hypothesized_posteriors / np.expand_dims(
        np.sum(hypothesized_posteriors, axis=1), axis=1
    )

    # 3.) Compute the KL divergence FROM each hypothesized
    #     posterior TO the current posterior:
    KL_Ds = KL_D(current_posterior, hypothesized_posteriors)

    # 4.) Multiply each divergence by the pertinent choice's
    #     probability and sum those products:
    expected_KL_D = np.sum(KL_Ds * choice_probabilities)

    return expected_KL_D


def boltzmann_likelihood(
    coefficient_magnitude: float, numerator_array: np.ndarray
) -> np.ndarray:
    """Compute likelihood via a Boltzmann-rational distribution.

    Distribution is aka softmax, Gibbs measure, and multinomial
    logit (probably) among others.

    ::return:
        ::likelihood_array: A #choices-by-#particles dimension
                            numpy array.
    """
    # 1.) Ensure the values are non-negative:
    # assert np.all(numerator_array >= 0), logger.critical(
    #    "Not all values in the numerator array are non-negative."
    # )
    # 2.) Account for the coefficient of stochasticity and the
    #     exponential operation:
    exponential_numerator_array = np.exp(
        -coefficient_magnitude * numerator_array
    )

    # 3.) Compute the likelihood's denominator by either. . .
    if numerator_array.shape[0] > 1:
        # 3a.) summing across each column's (ie. each model's) rows
        #      (ie. likelihoods for each option within a query):
        total_prob_denominator = np.sum(exponential_numerator_array, axis=0)
        total_prob_denominator = np.expand_dims(total_prob_denominator, axis=0)
    else:
        # 3b.) summing to a single normalizing scalar (e.g. for trajectory
        #      observations):
        total_prob_denominator = np.sum(exponential_numerator_array, axis=1)

    # 4.) Compute the likelihood for each element:
    likelihood_array = exponential_numerator_array / total_prob_denominator

    return likelihood_array


def good_or_bad(
    coefficient_magnitude: float, decision_threshold: float
) -> bool:
    """Determine if an input meets some goodness threshold."""
    pass


def feature_function(
    query_param_values: np.ndarray,
    model_params: np.ndarray,
    normalize: bool = True,
    combine: bool = True,
) -> np.ndarray:
    """Compute the features from sets of parameters."""
    feature_mats = np.zeros(
        (
            query_param_values.shape[0],
            model_params.shape[0],
            query_param_values.shape[1],
        )
    )
    # 1.) Separate and stack each option's params for efficient computation:
    for i in range(query_param_values.shape[0]):
        q_option = np.repeat(
            np.reshape(query_param_values[i, :], (1, -1)),
            model_params.shape[0],
            axis=0,
        )
        # 1a.) Compute the element-wise magnitudes between each option in
        #      a query and each possible reward function model:
        feature_mats[i, :, :] = np.abs(model_params - q_option)

    # 2.) Min-normalize each feature w.r.t. that feature:
    if normalize:
        for j in range(feature_mats.shape[2]):
            normed_feature = min_normalize(feature_mats[:, :, j])
            for i in range(feature_mats.shape[0]):
                feature_mats[i, :, j] = normed_feature[i, :]

    # 3.) If we want an unweighted linear combination of features for each
    #     particle, sum each particle's normalized features:
    if combine:
        feature_list = np.zeros((feature_mats.shape[0], feature_mats.shape[1]))
        for i in range(feature_mats.shape[0]):
            feature_list[i] = np.sum(feature_mats[i, :, :], axis=1)
        return feature_list
    # 3b.) Otherwise, return the set of particle features computed for
    #      each query:
    else:
        return feature_mats


def mean_normalize(data: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Normalize the data to a common scale."""
    mean = np.mean(data)
    std_dev = np.nanstd(data)

    normed_data = (data - mean) / std_dev
    return normed_data, mean, std_dev


def min_normalize(data: np.ndarray) -> Tuple:
    """Shift and scale the data."""
    # 0,) If the data is 0, just return:
    if np.all(data == 0):
        return data
    # 1.) Get the min and max:
    d_min = np.min(data)
    d_max = np.max(data)

    # 2.) Get the range:
    d_range = d_max - d_min

    # 3.) Subtract the min from each data-point:
    shifted_data = data - d_min

    # 4.) Divide by the range:
    if d_range > 0:
        normed_data = shifted_data / d_range
    else:
        normed_data = data

    return normed_data


def rescale(data: np.ndarray, mean: float, std_dev: float) -> np.ndarray:
    """Rescale the data from its normalized state."""
    rescaled_data = data * std_dev + mean

    return rescaled_data


def generate_hypothesized_pizza(
    pizza_form: dict,
    desired_param_dict: dict,
    param_function: callable,
    error_threshold: float,
) -> np.ndarray:
    """Generate a pizza from learned desired_params."""
    viable = False
    goal_value = None
    goal_index = None
    try:
        goal_value = desired_param_dict["coverage"]
        goal_index = np.argwhere(param_function.basis_fns == "coverage")[0, 0]
    except KeyError:
        goal_value = desired_param_dict["object_count"]
        goal_index = np.argwhere(param_function.basis_fns == "object_count")[
            0, 0
        ]

    desired_param_values = list(desired_param_dict.values())
    while not viable:
        start = time.perf_counter()
        _, hypothesized_params, hypothesized_pizza = stepwise_pizza_generator(
            goal_value,
            goal_index,
            pizza_form,
            desired_param_dict,
            desired_param_values,
            param_function,
        )
        viable = check_if_viable(
            goal_index,
            hypothesized_pizza,
            desired_param_values,
            hypothesized_params,
            param_function,
            error_threshold,
        )
        elapsed = time.perf_counter() - start
        if elapsed > 180:
            print(f"Stopping generation after {elapsed:.3f} seconds.")
            return None

    return hypothesized_pizza


def stepwise_pizza_generator(
    sufficiency_value: float,
    sufficiency_value_index: int,
    pizza_attributes: dict,
    desired_parameter_dict: dict,
    desired_parameter_values: np.ndarray,
    parameter_function: callable,
) -> Pizza:
    """Sequentially build a pizza w.r.t. desired_parameters.

    ::inputs:
        ::sufficiency_value: Defines when this pizza is 'finished.'
        ::sufficiency_value_index: To ensure we index appropriate parameters.
        ::pizza_attributes: The high-level pizza description.
        ::desired_parameters: The learned feature-parameters.
    """
    c = pizza_attributes["crust_thickness"]
    topping_size = pizza_attributes["topping_size"]
    topping_area = (topping_size / 2.0) ** 2  # * np.pi
    d = pizza_attributes["diameter"]
    viable_surface_radius = (d / 2.0) - (c + (topping_size / 2.0))
    viable_surface_area = viable_surface_radius ** 2  # * np.pi

    pizza_array = np.random.uniform(
        low=-viable_surface_radius, high=viable_surface_radius, size=(2, 1)
    )
    proposed_pizza = None
    proposed_pizzas_params = None
    current_value = 0
    # TODO Change 'coverage' to incorporate object count:
    coverage_error = np.abs(current_value - sufficiency_value)
    coverage_error_threshold = topping_area / (2 * viable_surface_area)

    start = time.perf_counter()
    while coverage_error > coverage_error_threshold:
        # print(f"The latest coverage error is: {coverage_error:.4f}")

        viable_placement = False
        while not viable_placement:
            proposed_next_topping = np.random.uniform(
                low=-viable_surface_radius,
                high=viable_surface_radius,
                size=(2, 1),
            )
            proposed_pizza_array = np.append(
                pizza_array, proposed_next_topping, axis=1
            )
            # TODO fix compute_params so we don't have to create this pizza:
            proposed_pizza = Pizza.from_dict(
                pizza_attributes, proposed_pizza_array
            )
            proposed_pizzas_params = parameter_function.compute_params(
                proposed_pizza
            ).reshape(1, -1)
            acceptable_param_count = 0
            # Check the parameters we computed:
            for i in range(parameter_function.basis_fns.shape[0]):
                if i == sufficiency_value_index:
                    continue
                if parameter_function.optimize_as[i] == "equal":
                    if ~np.isclose(
                        proposed_pizzas_params[0, i],
                        desired_parameter_values[i],
                    ):
                        del proposed_pizza_array
                        break
                elif parameter_function.optimize_as[i] == "max":
                    if (
                        proposed_pizzas_params[0, i]
                        > desired_parameter_values[i]
                    ):
                        del proposed_pizza_array
                        break
                elif parameter_function.optimize_as[i] == "min":
                    if (
                        proposed_pizzas_params[0, i]
                        < desired_parameter_values[i]
                    ):
                        del proposed_pizza_array
                        break
                acceptable_param_count += 1
            # Did all params pass or did we break early?:
            if acceptable_param_count == proposed_pizzas_params.shape[1] - 1:
                pizza_array = proposed_pizza_array
                viable_placement = True
                current_value = proposed_pizzas_params[
                    0, sufficiency_value_index
                ]
            coverage_error = np.abs(current_value - sufficiency_value)
            elapsed = time.perf_counter() - start
            if elapsed > 180:
                print(f"Stopping generation after {elapsed:.3f} seconds.")
                return None
        # print(f"Placed another topping after {elapsed:.4f} seconds.")
        # print(f"The latest coverage is {current_value:.4f}")
        pizza_array = proposed_pizza_array

    return pizza_array, proposed_pizzas_params, proposed_pizza


def check_if_viable(
    target_value_index: int,
    proposed_object: object,
    desired_param_values: np.ndarray,
    proposed_params: np.ndarray,
    parameter_function: callable,
    error_threshold: float,
) -> bool:
    """Check if tested_object's parameters satisfy error_threshold."""
    residuals = np.abs(proposed_params - desired_param_values)
    error_ratios = residuals / desired_param_values
    # print(
    #    f"The residuals were:\n{residuals}.\nThe corresponding error "
    #    f"ratios were:\n{error_ratios}."
    # )
    for i in range(parameter_function.basis_fns.shape[0]):
        if i == target_value_index:
            continue
        if parameter_function.optimize_as[i] == "equal":
            if error_ratios[0, i] > error_threshold:
                return False
        elif parameter_function.optimize_as[i] == "max":
            if (
                proposed_params[0, i] > desired_param_values[i]
                or error_ratios[0, i] > error_threshold
            ):
                return False
        elif parameter_function.optimize_as[i] == "min":
            if (
                proposed_params[0, i] < desired_param_values[i]
                or error_ratios[0, i] > error_threshold
            ):
                return False
    return True


def rotate(vector: np.ndarray, rads: float) -> np.ndarray:
    """Rotate a vector by (+/-)radians amount."""
    rotated_x = np.cos(vector[0]) - np.sin(vector[1])
    rotated_y = np.sin(vector[0]) + np.cos(vector[1])
    return np.array([[rotated_x, rotated_y]])


def cosine_sim(truth: np.ndarray, approx: np.ndarray) -> float:
    """Compute the cosine-similarity of two vectors."""
    numerator = truth.dot(approx.T)
    denom = np.linalg.norm(truth) * np.linalg.norm(approx)
    if np.isnan(numerator) or np.isnan(denom):
        print("Encountered a nan in cosine sim. Returning 0.")
        return 0

    return numerator / denom


def new_find_best_result(
    learned_weights: np.ndarray,
    learned_params: np.ndarray,
    ideal_params: np.ndarray,
) -> int:
    """Find the learned model which returns the greatest reward."""
    pass


def find_best_result(
    true_weights, true_params, hypoth_weights, hypoth_params
) -> int:
    """ID the result most similar to the true values."""
    # 1.) Normalize the param values:
    normed_params = np.append(hypoth_params, true_params, axis=0)
    for i in range(normed_params.shape[1]):
        normed_params[:, i] = min_normalize(normed_params[:, i])
    # 2.) Make appropriate truth-arrays:
    t_weights_repeated = np.repeat(
        true_weights, hypoth_weights.shape[0], axis=0
    )
    t_params_repeated = np.repeat(
        normed_params[-1, :].reshape(1, -1), hypoth_params.shape[0], axis=0
    )

    # 3.) Compute the norms:
    l2_params = np.linalg.norm(
        t_params_repeated - normed_params[:-1, :], axis=1
    )
    l2_weights = np.linalg.norm(t_weights_repeated - hypoth_weights, axis=1)

    # 4.) Sum the L2 norms and return the index of the smallest value:
    sums = l2_params + l2_weights
    return np.argmin(sums)


def arrange_params_and_weights(
    feature_params: dict, feature_weights: dict
) -> Tuple[np.ndarray, np.ndarray]:
    """Create arrays of feature parameters and weights that are aligned."""
    params_array = np.array(list(feature_params.values()))
    param_labels = np.array(list(feature_params.keys()))
    weights_array = np.array([])
    # Loop to ensure the weights and params instantiate in the same order:
    for k in param_labels:
        weights_array = np.append(weights_array, feature_weights[k])

    return params_array, weights_array
