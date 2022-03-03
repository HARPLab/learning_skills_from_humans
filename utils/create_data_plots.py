import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_input():
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=str, default="expected_values")
    parser.add_argument("file", type=str, default=None)

    return parser.parse_args()


def get_data(data_path):
    df = pd.read_csv(data_path, sep=",")
    return df


def plot_expected_values(
    labels: np.ndarray, expectation_data: np.ndarray
) -> None:
    """Plot the expected feature parameters and importance weights.

    ::inputs:
        ::data_frame: A Pandas dataframe with data in the column-wise order:
                         E[feature]:true feature:E[weight]:true weight
    """
    # 1.) Arrange subplots s.t. the first row is of feature parameters
    #   and the second row is of the corresponding importance weights:
    feature_count = expectation_data.shape[1] // 4
    query_index = np.arange(expectation_data.shape[0])
    fig, axs = plt.subplots(nrows=2, ncols=feature_count)
    axs[0, 0].set_ylabel("Features")
    axs[1, 0].set_ylabel("Weights")
    for i in range(feature_count):
        # 2a.) Plot the expected feature:
        axs[0, i].set_title(labels[feature_count * i + i])
        axs[0, i].plot(
            query_index, expectation_data[:, feature_count * i + i], "-r"
        )
        # 2b.) Plot the true feature:
        axs[0, i].axhline(
            expectation_data[0, feature_count * i + i + 1],
            c="blue",
            label="Truth",
        )
        # 2c.) Plot the expected weight:
        axs[1, i].set_title(labels[feature_count * i + i + 2])
        axs[1, i].plot(
            query_index, expectation_data[:, (feature_count * i + i + 2)], "-r"
        )
        # 2d.) Plot the true weight:
        axs[1, i].axhline(
            expectation_data[0, (feature_count * i + i + 3)],
            c="blue",
            label="Truth",
        )
    axs[0, 0].legend()
    axs[1, 0].legend()
    plt.tight_layout()
    plt.show()


def plot_choice_comparisons(data: np.ndarray) -> None:
    humans_choices = data[:, 0]
    agents_choices = data[:, 1]
    fig, axs = plt.subplots()
    check = np.argwhere(humans_choices == agents_choices)
    # consecutive_matches = np.array([])
    # for i in range(check.shape[0]):
    #    if check[i] == 1:

    # axs.plot(np.arange(data.shape[0]), check, "-g")
    axs.vlines(check, 1)
    axs.title.set_text("Comparing oracle's choice against agent's choice.")
    axs.set_xlabel("Time")
    axs.set_ylabel("Chose the same option (boolean)")
    plt.show()


def plot_data(data_frame, data_context: str, save_fig=True) -> None:
    """Plot data which resulted from query active learning."""
    data_new = data_frame.to_numpy()
    if data_context == "expected_values":
        headers = data_frame.columns[1:]
        data_new = np.delete(data_new, 0, axis=1)
        plot_expected_values(headers, data_new)
    elif data_context == "choice_comparisons":
        data_new = np.delete(data_new, 0, axis=1)
        plot_choice_comparisons(data_new)
    elif data_context == "resulting_reward":
        pass


user_inputs = parse_input()

path = Path(user_inputs.file)

d = get_data(path)

plot_data(d, user_inputs.context)
