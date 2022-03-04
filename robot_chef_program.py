#!/usr/bin/env python3
import time
from pathlib import Path

import matplotlib.animation as animate
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from agent import Agent
from human import Human
from particle_filters.particle import Particle
from pizza import Pizza
from utils.arguments import get_args
from utils.utils import (
    find_best_result,
    generate_hypothesized_pizza,
    min_normalize,
    arrange_params_and_weights,
)
from utils.visualizer import Visualizer

output_path = Path.cwd() / Path("data/")
if not output_path.exists():
    output_path.mkdir(parents=True)


def main() -> None:
    """Run the program."""
    # 1.) Get user arguments:
    user_inputs = get_args()

    # 2.) Set up scenario and models
    params_array, weights_array = arrange_params_and_weights(
        user_inputs.humans_ideal_params, user_inputs.humans_importance_weights
    )
    # 3.) Set up the pizza form:
    pizza_form = user_inputs.pizza_form
    start = time.perf_counter()
    ideal_pizza_toppings = generate_hypothesized_pizza(
        pizza_form=pizza_form, desired_params=user_inputs.humans_ideal_params
    )
    elapsed = time.perf_counter() - start
    print(f"It took {elapsed:.4f}s to generate an ideal pizza.")
    ideal_pizza_sample = Pizza.from_dict(pizza_form, ideal_pizza_toppings)

    # plot_forms = {
    #    "titles": ["Particles"],
    #    "subplot_titles": [["Importance weights", "Reward"]],
    # }
    # plot_forms = {
    #    "titles": ["Queries"],
    #    "subplot_titles": [["Option 1", "Option 2"]],
    # }
    plot_forms = {"titles": ["Belief"], "subplot_titles": [["User's pizza"]]}
    # plot_forms = {
    #    "titles": ["Particles", "Queries"],
    #    "subplot_titles": [
    #        ["Importance weights", "Reward"],
    #        ["Option 1", "Option 2"],
    #    ],
    # }
    #    "x_axes_labels": [[]],
    #    "y_axes_labels": [[]],
    # }

    viz = Visualizer(plot_forms)
    robot = Agent(
        sample_pizza=ideal_pizza_sample,
        desired_particle_count=user_inputs.particle_count,
        query_candidate_count=user_inputs.query_can_ct,
        basis_functions=list(user_inputs.humans_ideal_params.keys()),
        query_options=user_inputs.option_count,
        option_feature_max=user_inputs.option_feature_max,
        confidence_coefficient=user_inputs.confidence,
        resampling_threshold=user_inputs.resampling_threshold,
        param_noise_level=user_inputs.params_noise,
        weight_noise_level=user_inputs.weights_noise,
        axes_count=2,
        seed=user_inputs.seed,
        preference_type=user_inputs.preference_type,
    )
    human = Human.from_list(
        params_array,
        weights_array,
        list(user_inputs.humans_ideal_params.keys()),
        user_inputs.confidence,
    )

    # 3.) Create a pause-when-clicked event:
    def toggle_pause(*args, **kwargs):
        global paused
        if paused:
            anim1.resume()
            anim2.resume()
        else:
            anim1.pause()
            anim2.pause()
        paused = not paused

    # 4.) Run experiment using an animation of the get_human_feedback function.
    #     Note fargs=() are the get_human_feedback input arguments:
    expected_param_values = np.zeros(
        (user_inputs.query_count, params_array.shape[0])
    )
    expected_param_weights = np.zeros(
        (user_inputs.query_count, params_array.shape[0])
    )
    for i in range(user_inputs.query_count):
        robot.get_human_feedback(i, False, human, user_inputs.step_plot)
        expected_param_values[
            i, :
        ] = robot.filter.get_instantaneous_expected_params(
            user_inputs.particle_count
        )
        expected_param_weights[
            i, :
        ] = robot.filter.get_instantaneous_expected_weights(
            user_inputs.particle_count
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
        human.get_weights(),
        human.get_params(),
        expected_param_weights,
        expected_param_values,
    )
    d = {}
    final_pizza_config_dict = {}
    best_pizza_config_dict = {}
    keys = list(user_inputs.humans_ideal_params.keys())
    for i in range(params_array.shape[0]):
        final_pizza_config_dict[keys[i]] = expected_param_values[-1, i]
        best_pizza_config_dict[keys[i]] = expected_param_values[best_result, i]
        final_pizza_config_dict[f"E[param {i}]"] = expected_param_values[-1, i]
        d[f"E[{keys[i]}]"] = expected_param_values[:, i]
        d[f"True {keys[i]}"] = params_array[i]
        d[f"E[w_{i}]"] = expected_param_weights[:, i]
        d[f"True w_{i}"] = weights_array[i]

    df = pd.DataFrame(d)
    t = time.localtime()
    t = time.strftime("%Y:%m:%d:%H:%M:%S", t)
    data_path = output_path / Path("expected_values")
    if not data_path.exists():
        data_path.mkdir(parents=True)
    df.to_csv(
        output_path
        / f"expected_values/{t}_{params_array.shape[0]}_params_{user_inputs.query_count}_queries.csv"
    )
    query_choices = {}
    query_choices["Human"] = robot.humans_choices
    query_choices["Agent"] = robot.agents_choices
    df = pd.DataFrame(query_choices)

    data_path = output_path / Path("choice_comparisons")
    if not data_path.exists():
        data_path.mkdir(parents=True)

    df.to_csv(
        output_path
        / f"choice_comparisons/{t}_query_choices_{params_array.shape[0]}_params_{user_inputs.query_count}_queries.csv"
    )
    final_guessed_topping_placements = generate_hypothesized_pizza(
        pizza_form, final_pizza_config_dict
    )
    best_guessed_topping_placements = generate_hypothesized_pizza(
        pizza_form, best_pizza_config_dict
    )
    final_pizza_config = Pizza.from_dict(
        pizza_form, final_guessed_topping_placements
    )
    best_pizza_config = Pizza.from_dict(
        pizza_form, best_guessed_topping_placements
    )
    viz.visualize_expected_configuration(
        final_pizza_config, "Guess at final iteration."
    )
    viz.visualize_expected_configuration(
        best_pizza_config, f"Best model found at iteration {best_result})"
    )
    # anim1 = animate.FuncAnimation(
    #    robot.viz.figs["queries"],
    #    robot.get_human_feedback,
    #    fargs=(False, human, user_inputs.step_plot),
    #    save_count=500,
    #    interval=1,
    # )
    # anim2 = animate.FuncAnimation(
    #    robot.viz.figs["particles"],
    #    robot.get_human_feedback,
    #    fargs=(False, human, user_inputs.step_plot),
    #    save_count=500,
    #    interval=1,
    # )
    # robot.viz.figs["queries"].canvas.mpl_connect(
    #    "button_press_event", toggle_pause
    # )
    # robot.viz.figs["particles"].canvas.mpl_connect(
    #    "button_press_event", toggle_pause
    # )

    # 5.) Optionally--save a gif of the animation(s):
    # anim_title = (
    #    str(user_inputs.particle_count)
    #    + "_particles_"
    #    + str(user_inputs.params_noise)
    #    + "_variance_"
    #    + str(user_inputs.option_count)
    #    + "_qoptions.gif"
    # )
    # anim.save(anim_title, writer="pillow", fps=40)


if __name__ == "__main__":
    main()
