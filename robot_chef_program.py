#!/usr/bin/env python3
import time
from pathlib import Path

import numpy as np
import pandas as pd
import logging
from logging import handlers

output_path = Path.cwd() / Path("data/")
if not output_path.exists():
    output_path.mkdir(parents=True)
    Path(Path.cwd() / "logs").mkdir(parents=True)

from agent import Agent
from human import Human
from particle_filters.particle import Particle
from pizza import Pizza
from utils.arguments import get_args
from utils.utils import (
    arrange_params_and_weights,
    find_best_result,
    generate_hypothesized_pizza,
)
from utils.visualizer import Visualizer


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

formatted_output = "{asctime}|{name}|{levelname}|{message}"
formatter_1 = logging.Formatter(formatted_output, style="{")

handler_1 = logging.StreamHandler()
handler_1.setLevel(logging.INFO)
handler_1.setFormatter(formatter_1)

handler_2 = handlers.RotatingFileHandler(
    Path(__file__).parent / Path("logs/robot_chef_program.log")
)
handler_2.setLevel(logging.DEBUG)
handler_2.setFormatter(formatter_1)

logger.addHandler(handler_1)
logger.addHandler(handler_2)


def main() -> None:
    """Run the program."""
    user_inputs = get_args()

    params_array, weights_array = arrange_params_and_weights(
        user_inputs.humans_ideal_params, user_inputs.humans_importance_weights
    )
    pizza_form = user_inputs.pizza_form
    human = Human(
        params_array,
        weights_array,
        list(user_inputs.humans_ideal_params.keys()),
        user_inputs.oracle_confidence,
    )

    logger.info(
        "The human's desired params are:\n "
        f"{params_array}\nand her "
        f"importance weights are:\n{weights_array}"
    )

    start = time.perf_counter()
    ideal_pizza_sample = generate_hypothesized_pizza(
        pizza_form=pizza_form,
        desired_param_dict=user_inputs.humans_ideal_params,
        param_function=human.param_fn,
        error_threshold=user_inputs.error_threshold * 0.2,
    )
    elapsed = time.perf_counter() - start
    print(f"It took {elapsed:.4f}s to generate an ideal pizza.")

    plot_forms = {"titles": ["Belief"], "subplot_titles": [["User's pizza"]]}

    viz = Visualizer(plot_forms)
    robot = Agent(
        sample_pizza=ideal_pizza_sample,
        desired_particle_count=user_inputs.particle_count,
        query_candidate_count=user_inputs.query_can_ct,
        basis_functions=list(user_inputs.humans_ideal_params.keys()),
        query_options=user_inputs.query_option_count,
        option_feature_max=user_inputs.option_feature_max,
        confidence_coefficient=user_inputs.agent_confidence,
        resampling_threshold=user_inputs.resampling_threshold,
        param_noise_level=user_inputs.params_noise,
        weight_noise_level=user_inputs.weights_noise,
        axes_count=2,
        seed=user_inputs.seed,
        preference_type=user_inputs.preference_type,
    )
    expected_param_values = np.zeros(
        (user_inputs.query_count, params_array.shape[0])
    )
    expected_param_weights = np.zeros(
        (user_inputs.query_count, params_array.shape[0])
    )
    for i in range(user_inputs.query_count):
        robot.get_human_feedback(False, human)
        expected_param_values[
            i, :
        ] = robot.filter.get_instantaneous_expected_params(
            # user_inputs.particle_count
            # 10
            5
        )
        expected_param_weights[
            i, :
        ] = robot.filter.get_instantaneous_expected_weights(
            # user_inputs.particle_count
            # 10
            5
        )
        print(
            f"Expected param values at iteration {i}:\n"
            f"{expected_param_values[i,:]}"
        )
        print(
            f"Expected param importance weights at iteration {i}:\n"
            f"{expected_param_weights[i,:]}"
        )
    best_result = find_best_result(
        human.importance_weights,
        human.desired_params,
        expected_param_weights,
        expected_param_values,
    )
    d = {}
    best_pizza_config_dict = {}
    keys = list(user_inputs.humans_ideal_params.keys())
    for i in range(params_array.shape[0]):
        best_pizza_config_dict[keys[i]] = expected_param_values[best_result, i]
        d[f"E[{keys[i]}]"] = expected_param_values[:, i]
        d[f"True {keys[i]}"] = params_array[i]
        d[f"E[w_{i}]"] = expected_param_weights[:, i]
        d[f"True w_{i}"] = weights_array[i]

    df = pd.DataFrame(d)
    t = time.localtime()
    t = time.strftime("%Y:%m:%d:%H:%M:%S", t)
    if not Path(output_path / "expected_values/").exists():
        Path(output_path / "expected_values/").mkdir(parents=True)
    df.to_csv(
        output_path
        / f"expected_values/{t}_{params_array.shape[0]}_params_{user_inputs.query_count}_queries.csv"
    )
    query_choices = {}
    query_choices["Human"] = robot.humans_choices
    query_choices["Agent"] = robot.agents_choices
    df = pd.DataFrame(query_choices)

    if not Path(output_path / "choice_comparisons/").exists():
        Path(output_path / "choice_comparisons/").mkdir(parents=True)
    df.to_csv(
        output_path
        / f"choice_comparisons/{t}_query_choices_{params_array.shape[0]}_params_{user_inputs.query_count}_queries.csv"
    )
    print(f"Generating a pizza using parameters:\n{best_pizza_config_dict}")
    best_pizza_config = generate_hypothesized_pizza(
        pizza_form,
        best_pizza_config_dict,
        robot.param_function,
        user_inputs.error_threshold,
    )
    if best_pizza_config is None:
        print("Didn't generate a pizza in time; can't create the visual.")
    else:
        viz.visualize_expected_configuration(
            best_pizza_config, f"Best model found at iteration {best_result})"
        )


if __name__ == "__main__":
    main()
