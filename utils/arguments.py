import argparse


def get_args():
    """Read from terminal or bash script."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preference_type",
        type=str,
        default="next_slice",
        help="Preference type is for: A comparison of two completely different\
                pizzas ('complete') or a comparison of a pizza which differs\
                only in the placement of the NEXT topping ('next_slice')",
    )
    parser.add_argument(
        "--pizza_form",
        type=dict,
        default={
            "diameter": 35,
            "slices": 8,
            "crust_thickness": 2.54,
            "topping_size": 3.54,
        },
        help="The diameter (cm), crust thickness (cm), and average topping size (cm),\
                as well as the pizza's slice count.",
    )
    parser.add_argument(
        "--humans_importance_weights",
        type=dict,
        default={
            "coverage": 0.6,
            "closest_neighbor": 0.3,
            "furthest_neighbor": 0.1,
        },
        help="[object_count x_centroid y_centroid x_variance y_variance density\
                coverage max_dist_from_origin]",
    )
    parser.add_argument(
        "--humans_ideal_params",
        type=dict,
        default={
            "coverage": 0.3,
            "closest_neighbor": 3.54,
            "furthest_neighbor": 24.00,
        },
        help="[topping_count x_centroid y_centroid x_variance y_variance density\
                coverage(percentage) max_dist_from_origin]",
    )
    parser.add_argument(
        "--error_threshold",
        "-e",
        type=float,
        default=0.15,
        help="The permissible percent-error in parameter values when "
        "generating a pizza. Error computed as the difference between "
        "learned parameter values and desired parameter values, "
        "divided by desired parameter values.",
    )
    parser.add_argument(
        "--query_count",
        type=int,
        default=30,
        help="Number of queries to pose.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--params_noise", type=float, default=0.40)
    parser.add_argument("--weights_noise", type=float, default=0.05)
    parser.add_argument("--agent_confidence", type=float, default=5.0)
    parser.add_argument("--oracle_confidence", type=float, default=5.0)
    parser.add_argument("--resampling_threshold", type=float, default=0.5)
    parser.add_argument("--particle_count", type=int, default=100)
    parser.add_argument("--query_option_count", type=int, default=2)
    parser.add_argument("--option_feature_max", type=int, default=200)
    parser.add_argument("--query_can_ct", type=int, default=500)
    parser.add_argument(
        "--get_human_input", action="store_true", default=False
    )
    parser.add_argument("--query_method", type=str, default="uniform")
    parser.add_argument("--bin_count", type=int, default=1000)

    return parser.parse_args()
