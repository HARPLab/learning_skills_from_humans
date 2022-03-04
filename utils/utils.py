from pathlib import Path
from typing import Tuple, Union

import matplotlib.animation as animate
import numpy as np


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
    assert np.all(numerator_array >= 0), logger.critical(
        "Not all values in the numerator array are non-negative."
    )
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
    pizza_form: dict, desired_params: dict
) -> np.ndarray:
    """Sequentially build a pizza w.r.t. feature-parameters.

    ::inputs:
        ::pizza_form: The higher-level pizza attributes.
        ::desired_params: The learned feature-parameters.
    """
    # 1.) Get the topping count:
    covered = 0
    c = pizza_form["crust_thickness"]
    t_s = pizza_form["topping_size"]
    d = pizza_form["diameter"]
    viable_surface_radius = (d / 2.0) - (c + (t_s / 2.0))
    topping_count = 0
    if np.any(np.isin(list(desired_params.keys()), "coverage")):
        while covered < desired_params["coverage"]:
            topping_count += 1
            covered = (
                topping_count
                * np.pi
                * (t_s / 2.0) ** 2
                / (np.pi * viable_surface_radius**2)
            )
    elif np.any(np.isin(list(desired_params.keys()), "object_count")):
        topping_count = desired_params["object_count"]
    pizza_array = np.zeros((2, topping_count))
    # 2.) Get centroid:
    x_centroid = 0
    y_centroid = 0
    if np.any(np.isin(list(desired_params.keys()), "x_centroid")):
        x_centroid = desired_params["x_centroid"]
        y_centroid = desired_params["y_centroid"]
    # 3.) Generate topping positions:
    if np.any(np.isin(list(desired_params.keys()), "average_vector_change")):
        # 3b.) Follow the trajectory implied by the approximated
        #      average vector change. First, randomly place the
        #      first topping:
        pizza_array[0, 0] = np.random.normal(
            loc=x_centroid,
            scale=np.sqrt(desired_params["x_variance"]),
            size=(1, 1),
        )
        pizza_array[1, 0] = np.random.normal(
            loc=y_centroid,
            scale=np.sqrt(desired_params["y_variance"]),
            size=(1, 1),
        )
        t_ct = 1
        while t_ct < topping_count:
            # 3c.) Then, until we've placed as many toppings that abide
            #     by surface_coverage, translate each subsequent topping
            #     along x until the magnitude between it and the last
            #     topping is of the desired value, and then rotate the
            #     second topping until the angle between it and the last
            #     topping is of the desired value:
            new_y = 0
            new_x = (
                desired_params["average_vector_change"][1]
                - pizza_array[0, t_ct - 1]
            )
            pizza_array[:, t_ct] = rotate(
                np.array([[new_x, new_y]]),
                desired_params["average_vector_change"][0],
            )

    elif np.any(np.isin(list(desired_params.keys()), "x_variance")):
        pizza_array[0, :] = np.random.normal(
            loc=x_centroid,
            scale=np.sqrt(desired_params["x_variance"]),
            size=(1, topping_count),
        )
        pizza_array[1, :] = np.random.normal(
            loc=y_centroid,
            scale=np.sqrt(desired_params["y_variance"]),
            size=(1, topping_count),
        )
    elif np.any(np.isin(list(desired_params.keys()), "max_dist_from_origin")):
        object_space = np.linspace(
            -desired_params["max_dist_from_origin"],
            desired_params["max_dist_from_origin"],
            20000,
        )
        pizza_array = np.random.choice(object_space, size=(2, topping_count))
    elif np.any(np.isin(list(desired_params.keys()), "avg_distance")):
        object_space = np.linspace(
            -viable_surface_radius, viable_surface_radius, 20000
        )
        pizza_array = np.random.choice(object_space, size=(2, topping_count))
        while (
            avg_distance(pizza_array) - desired_params["avg_distance"] > 10e-4
        ):
            pizza_array = np.random.choice(
                object_space, size=(2, topping_count)
            )
    else:
        object_space = np.linspace(
            -viable_surface_radius, viable_surface_radius, 20000
        )
        pizza_array = np.random.choice(object_space, size=(2, topping_count))

    # 4.) Now check what params remain for which we must account:
    for k in desired_params.keys():
        if k == "max_dist_from_origin":
            # If there are no toppings, break:
            if topping_count == 0:
                continue
            else:
                # Pull the furthest point from the surface-origin (0,0)
                # closer to it until all toppings are within the maximum
                # distance allowed:
                while (
                    np.nanmax(np.linalg.norm(pizza_array, axis=0))
                    > desired_params["max_dist_from_origin"]
                ):
                    pizza_array = np.random.choice(
                        object_space, size=(2, topping_count)
                    )
                    # argmax_dist = np.nanargmax(
                    #    np.linalg.norm(pizza_array, axis=0)
                    # )
                    # x_sign = np.sign(pizza_array[0, argmax_dist])
                    # y_sign = np.sign(pizza_array[1, argmax_dist])
                    # pizza_array[0, argmax_dist] = (
                    #    pizza_array[0, argmax_dist] - x_sign * 0.1
                    # )
                    # pizza_array[1, argmax_dist] = (
                    #    pizza_array[1, argmax_dist] - y_sign * 0.1
                    # )
        if k == "x_variance":
            # If variance is too high, pull the objects toward the centroid,
            # but do so in a manner proportional to how distal each point is
            # from the centroid:
            while (
                np.abs(
                    pizza_array[0, :].std() ** 2 - desired_params["x_variance"]
                )
                > 10e-4
            ):
                pizza_array[0, :] = np.where(
                    pizza_array[0, :] < x_centroid,
                    pizza_array[0, :] + 0.01 * pizza_array[0, :],
                    pizza_array[0, :] - 0.01 * pizza_array[0, :],
                )
        if k == "y_variance":
            # If variance is too high, pull the objects toward the centroid,
            # but do so in a manner proportional to how distal each point is
            # from the centroid:
            while (
                np.abs(
                    pizza_array[1, :].std() ** 2 - desired_params["y_variance"]
                )
                > 10e-4
            ):
                pizza_array[1, :] = np.where(
                    pizza_array[1, :] < y_centroid,
                    pizza_array[1, :] + 0.01 * pizza_array[1, :],
                    pizza_array[1, :] - 0.01 * pizza_array[1, :],
                )
    return pizza_array


def rotate(vector: np.ndarray, rads: np.float) -> np.ndarray:
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


# def feature_function(
#    query_param_values: np.ndarray,
#    model_params: np.ndarray,
#    model_weights: np.ndarray,
# ) -> np.ndarray:
#    """Compute the features from sets of parameters."""
#    feature_mats = np.zeros(
#        (
#            query_param_values.shape[0],
#            model_params.shape[0],
#            query_param_values.shape[1],
#        )
#    )
#    # 1.) Separate and stack each option's params for efficient computation:
#    for i in range(query_param_values.shape[0]):
#        q_option = np.repeat(
#            np.reshape(query_param_values[i, :], (1, -1)),
#            model_params.shape[0],
#            axis=0,
#        )
#        # 1a.) Compute the element-wise magnitudes between each option in
#        #      a query and each possible reward function model:
#        feature_mats[i, :, :] = np.abs(model_params - q_option)
#
#        # 1b.) Incorporate the weights via element-wise multiplication. Note
#        #      that including the weights before normalizing is mathematically
#        #      the same as including them after normalizing the differences:
#        # feature_mats[i, :, :] = np.abs(
#        #    feature_mats[i, :, :]  * model_weights
#        # )
#    # 2.) Min-normalize each feature-difference
#    for j in range(feature_mats.shape[2]):
#        normed_feature = min_normalize(feature_mats[:, :, j])
#        for i in range(feature_mats.shape[0]):
#            feature_mats[i, :, j] = normed_feature[i, :]
#
#    # 3.) Sum the linear combos for each particle:
#    weighted_feature_combos = np.zeros(
#        (feature_mats.shape[0], feature_mats.shape[1])
#    )
#    for i in range(feature_mats.shape[0]):
#        weighted_feature_combos[i] = np.sum(feature_mats[i, :, :], axis=1)
#
#    return weighted_feature_combos
