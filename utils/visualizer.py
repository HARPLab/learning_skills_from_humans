import matplotlib.pyplot as plt
import numpy as np
from particle_filters.particle import Particle


class Visualizer:
    """Visualize data."""

    def __init__(self, figure_forms: dict):
        """Instantiate a visualizer.

        ::inputs:
            ::figure_forms: A dictionary with keys:
                            - titles: titles of figures being plotted
                            - subplot_titles: titles of figure subplots
                            - x_axes_labels: labels of figure subplots
                            - y_axes_labels: labels of figure subplots

                e.g.:
                    {
                    "titles":         ["Particles", "Queries"],
                    "subplot_titles": [["Importance weights","Reward"],
                                       ["Option 1", "Option 2"]
                                      ],
                    "x_axes_labels":  [[]],
                    "y_axes_labels":  [[]]
                    }
        """
        self.figs = {}
        self.axs = {}
        self.count = 0
        for i, t in enumerate(figure_forms["titles"]):
            t = t.lower()
            self.figs[t], self.axs[t] = plt.subplots(
                1, len(figure_forms["subplot_titles"][i]), tight_layout=True
            )
            for j, st in enumerate(figure_forms["subplot_titles"][i]):
                st = st.lower()
                if len(figure_forms["subplot_titles"][i]) > 1:
                    self.axs[t][j].title.set_text(st)
                elif len(figure_forms["subplot_titles"][i]) <= 1:
                    self.axs[t].title.set_text(st)

    def visualize_expected_configuration(
        self, configuration: object, title: str
    ):
        """Visualize the filter's current expected result."""
        # 0.) Reset the plots:
        plt.ion()
        diam = configuration.diameter
        t_size = float(configuration.topping_size)
        crust = configuration.crust_thickness
        coords = configuration.toppings + diam / 2.0
        self.axs["belief"].clear()
        self.axs["belief"].set_xlabel(title)
        self.axs["belief"].set_xlim(0, diam)
        self.axs["belief"].set_ylim(0, diam)
        self.axs["belief"].set_xticks(coords[:, 0])
        self.axs["belief"].set_yticks(coords[:, 1])
        # self.axs["belief"].set_xlabel("After query {self.count}")

        # 1.) Create a base pizza and add to axes:
        dough = plt.Circle(
            (diam / 2.0, diam / 2.0),
            radius=diam / 2.0,
            color="peru",
            fill=True,
        )

        cheese_radius = diam / 2.0 - crust
        cheese = plt.Circle(
            (diam / 2.0, diam / 2.0),
            radius=cheese_radius,
            color="lemonchiffon",
            fill=True,
        )
        latest_plot = self.axs["belief"].add_patch(dough)
        latest_plot = self.axs["belief"].add_patch(cheese)

        # 2.) Plot the toppings. Currently using circle
        #     patches to easily moderate the topping size:
        for i in range(coords.shape[1]):
            topping = plt.Circle(
                (coords[:, i]),
                radius=t_size / 2.0,
                color="firebrick",
                fill=True,
            )
            latest_plot = self.axs["belief"].add_patch(topping)

        plt.show()
        input("Press enter to continue.")
        return latest_plot

    def visualize_choice_comparisons(self):
        pass

    def visualize_query_space(self, queries: np.ndarray) -> np.ndarray:
        """Visualize how "fully" our query generation fills the query-space."""
        pass

    def visualize_query(
        self, query_form: dict, query_options: np.ndarray, chosen_option: int
    ) -> np.ndarray:
        """Visualize the options in a query.

        ::inputs:
            ::query_options: A |query_options| X |option_features| X |2|
                             array corresponding to a option that has
                             M features, each of which has (x,y) coordiates.
        """
        # 0.) Reset the plots:
        plt.ion()
        diam = query_form["diameter"]
        q_options = query_options.copy()
        for i in range(len(query_options)):
            self.axs["queries"][i].clear()
        self.axs["queries"][chosen_option].set_xlabel("Chosen by human")
        # 1.) Squash the coordinate-space to [0,1]:
        min_coordinate = -(
            (query_diam / 2.0)
            + query_form["crust_thickness"]
            + query_form["topping_size"] / 2.0
        )
        # coordinate_range = query_diam
        coordinate_range = np.abs(min_coordinate * 2)
        for i in range(len(q_options)):
            q_options[i][:, :] = (
                q_options[i][:, :] - min_coordinate
            ) / coordinate_range
        # 2.) For each option, create a base pizza and add to axes:
        cheese_radius = 0.5 - query_form["crust_thickness"] / coordinate_range
        for i in range(len(q_options)):
            crust = plt.Circle((0.5, 0.5), radius=0.5, color="peru", fill=True)
            cheese = plt.Circle(
                (0.5, 0.5),
                radius=cheese_radius,
                color="lemonchiffon",
                fill=True,
            )
            latest_plot = self.axs["queries"][i].add_patch(crust)
            latest_plot = self.axs["queries"][i].add_patch(cheese)

        # 3.) Plot the toppings in each query. Currently using circle
        #     patches to easily moderate the topping size:
        t_size = float(query_form["topping_size"]) / (2 * coordinate_range)
        for i in range(len(q_options)):
            for j in range(q_options[i].shape[1]):
                topping = plt.Circle(
                    (q_options[i][:, j]),
                    radius=t_size,
                    color="firebrick",
                    fill=True,
                )
                latest_plot = self.axs["queries"][i].add_patch(topping)

        # plt.draw()
        plt.show()
        return latest_plot

    def visualize_weights(
        self,
        query_options: np.ndarray,
        particle_count: int,
        posterior_belief: np.ndarray,
        latest_reward: float,
    ) -> np.ndarray:
        """Plot the posterior."""
        self.count += 1
        self.figs["particles"].suptitle(f"Iteration {self.count}")
        self.axs["particles"][0].cla()
        self.axs["particles"][0].set_xlabel("Particles")
        self.axs["particles"][0].set_ylabel("Weight")
        self.axs["particles"][0].title.set_text("Posterior weights")
        self.axs["particles"][1].title.set_text("Expected accumulated reward")
        self.axs["particles"][1].set_xlim(0, 500)
        self.axs["particles"][1].set_xlabel("Iteration")
        self.axs["particles"][1].set_ylabel("Reward")

        particle_positions = np.arange(particle_count)
        # Plot particles' importance according to their weights:
        latest_belief_plot = self.axs["particles"][0].vlines(
            particle_positions, ymin=0, ymax=posterior_belief, colors="y"
        )
        # Plot the reward accumulated by all particles:
        latest_belief_plot = self.axs["particles"][1].plot(
            self.count, latest_reward, ".b"
        )
        return latest_belief_plot


class InteractiveVisualizer(Visualizer):
    """A class which handles mouse-click events within plots."""

    def __init__(self):
        pass
