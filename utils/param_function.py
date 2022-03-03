from itertools import combinations, product
from typing import Union

import numpy as np
import time


class ParamFunction:
    """A class which contains param-function objects."""

    def __init__(self, basis_functions: list):
        """Compute the params of an option from a query."""
        basis_fn_memory_blocks = []
        # 1.) Make a composition of the basis functions:
        for b in basis_functions:
            if b == "object_count":
                basis_fn_memory_blocks.append(self.get_count)
            elif b == "centroid":
                basis_fn_memory_blocks.append(self.get_centroid)
            elif b == "variance":
                basis_fn_memory_blocks.append(self.get_variance)
            elif b == "density":
                basis_fn_memory_blocks.append(self.density)
            elif b == "max_dist_from_origin":
                basis_fn_memory_blocks.append(self.max_dist_from_origin)
            elif b == "average_vector_change":
                basis_fn_memory_blocks.append(self.average_vector_change)
            elif b == "avg_magnitude":
                basis_fn_memory_blocks.append(self.avg_magnitude)
            elif b == "furthest_neighbor":
                basis_fn_memory_blocks.append(self.furthest_neighbor)
            elif b == "closest_neighbor":
                basis_fn_memory_blocks.append(self.closest_neighbor)
            elif b == "coverage":
                basis_fn_memory_blocks.append(
                    self.approximate_surface_coverage
                )

        # 2.) Define the param function as a composition of
        #     the basis functions:
        def param_fn(option: np.ndarray) -> np.ndarray:
            """Compute the params of a query's option."""
            param_values = np.array([])
            # 3.)  For each basis function in this param function:
            for i, b in enumerate(basis_fn_memory_blocks):
                # 3.) Compute the param value:
                param_values = np.append(param_values, b(option))
            return param_values

        self.compute_params = param_fn

    def density(self, query_option) -> int:
        """Compute how many objects are within one std. dev. of mean.

        As of 1/18/22: WORKING.
        """
        o_tops = query_option.toppings
        if o_tops.shape[0] < 1:
            return 0
        # 1.) Get the objects' centroid:
        centroid = self.get_centroid(query_option)
        # 2.) Get the objects' (x,y) standard deviation:
        x_var, y_var = self.get_variance(query_option)
        x_s_d = np.sqrt(x_var)
        y_s_d = np.sqrt(y_var)
        count = 0
        # 3.) Count how many objects are within
        #     one s.t.d. of the objects' centroid:
        o_tops = query_option.toppings
        for i in range(o_tops.shape[1]):
            if np.abs(o_tops[0, i] - centroid[0]) <= x_s_d:
                if np.abs(o_tops[1, i] - centroid[1]) <= y_s_d:
                    count += 1
        return count

    def get_centroid(self, query_option) -> np.ndarray:
        """Compute the centroid of the option's objects.
        As of 1/18/22: WORKING.
        """
        o_tops = query_option
        if type(query_option) != np.ndarray:
            o_tops = query_option.toppings
        if o_tops.shape[1] <= 1:
            return o_tops
        else:
            # centroid = (np.nanmean(o_tops[0, :]), np.nanmean(o_tops[1, :]))
            centroid = (np.mean(o_tops[0, :]), np.mean(o_tops[1, :]))
            # print(f"centroid: {centroid}")
            return centroid

    def get_variance(self, query_option) -> np.ndarray:
        """Compute the variance of objects' (x,y) coordinates.

        Note: these variances are w.r.t. the centroid of
        the objects themselves.

        As of 1/18/22: WORKING.
        """
        t_count = 0
        toppings = query_option
        centroid = None
        # 0.) Account for pytest cases:
        if type(query_option) == np.ndarray:
            t_count = toppings.shape[1]
            centroid = self.get_centroid(toppings)
        else:
            t_count = query_option.get_topping_count()
            if t_count == 0 or t_count == 1:
                return 0, 0
            toppings = query_option.toppings
            centroid = self.get_centroid(query_option)
        # 1.) Compute the difference between the positional means
        #     and each object's individual position:
        x_diff = np.abs(toppings[0, :] - centroid[0])
        y_diff = np.abs(toppings[1, :] - centroid[1])
        # 2.) Find the squared-sum of the results:
        x_diff = np.sum(x_diff ** 2)
        y_diff = np.sum(y_diff ** 2)
        # 3.) Normalize:
        norm_term = 1
        if t_count - 1 > 0:
            norm_term = 1 / (t_count - 1)
        x_var = norm_term * x_diff
        y_var = norm_term * y_diff

        # print(f"xvar:{x_var}\nyvar:{y_var}\n")
        return x_var, y_var

    def get_count(self, query_option: Union[object, np.ndarray]) -> int:
        """Return a count of the option's objects."""
        if type(query_option) == np.ndarray:
            return query_option.shape[1]
        else:
            return query_option.get_topping_count()

    def max_dist_from_origin(
        self, query_option: Union[object, np.ndarray]
    ) -> float:
        """Compute the maximum distance between origin and option's objects."""
        # 1.) Get option's object-positions:
        if type(query_option) == np.ndarray:
            positions = query_option
        else:
            positions = query_option.toppings

        # 2.) Compute the maximum distance from the origin among
        #    those positions:
        max_dist = np.max(
            [
                np.linalg.norm(positions[:, i])
                for i in range(positions.shape[1])
            ]
        )

        return max_dist

    def avg_magnitude(self, query_option: Union[object, np.ndarray]) -> float:
        """Compute the average pairwise distance between objects."""
        if type(query_option) == np.ndarray:
            objects = np.array(query_option, copy=True)
        else:
            objects = query_option.toppings
        # 1.) Get all topping pairs:
        xy_tuples = np.array(list(zip(objects[0, :], objects[1, :])))
        xy_combs = list(combinations(xy_tuples, 2))
        # 2.) Compute the magnitude of the vector between them:
        mags = np.array(
            [
                np.linalg.norm(xy_combs[i][0] - xy_combs[i][1])
                for i in range(len(xy_combs))
            ]
        )
        if objects.shape[1] > 1:
            avg_mag = np.sum(mags) / len(xy_combs)
        else:
            avg_mag = np.sum(mags)
        return avg_mag

    def closest_neighbor(
        self, query_option: Union[object, np.ndarray]
    ) -> float:
        """Find the distance of the object closest to this one."""
        if type(query_option) == np.ndarray:
            objects = np.array(query_option, copy=True)
        else:
            objects = query_option.toppings
        # 1.) Stack the topping that was placed last:
        object_of_interest = np.repeat(
            objects[:, -1].reshape(-1, 1), objects.shape[1], axis=1
        )
        assert (
            objects.shape == object_of_interest.shape
        ), "Misaligned shapes in closest_neighbor."
        # 2.) Compute the magnitude between that last topping and
        #    the other toppings:
        mags = np.linalg.norm(objects - object_of_interest, axis=0)
        if objects.shape[1] > 1:
            closest_neighbor = mags[:-1].min()
        else:
            closest_neighbor = 0
        return closest_neighbor

    def furthest_neighbor(
        self, query_option: Union[object, np.ndarray]
    ) -> float:
        """Find the distance of the object furthest from this one."""

        if type(query_option) == np.ndarray:
            objects = np.array(query_option, copy=True)
        else:
            objects = query_option.toppings
        # 1.) Stack the topping that was placed last:
        object_of_interest = np.repeat(
            objects[:, -1].reshape(-1, 1), objects.shape[1], axis=1
        )
        assert (
            objects.shape == object_of_interest.shape
        ), "Misaligned shapes in furthest_neighbor."
        # 2.) Compute the magnitude between that last topping and
        #    the other toppings:
        mags = np.linalg.norm(objects - object_of_interest, axis=0)
        if objects.shape[1] > 1:
            closest_neighbor = mags.max()
        else:
            closest_neighbor = 0
        return closest_neighbor

    # def average_vector_change(
    #    self, query_option: Union[object, np.ndarray]
    # ) -> float:
    #    """The point-by-point vector change.

    #    Objective is to compute both the average magnitude
    #    and average angle between subsequent points.
    #    """
    #    # 1.) Get option's object-positions:
    #    if type(query_option) == np.ndarray:
    #        positions = query_option
    #    else:
    #        positions = query_option.toppings
    #    # 2.) Compute magnitudes between each subsequent point:
    #    mags = np.linalg.norm(positions, axis=0)

    #    # 3.) Compute angle between each subsequent point:
    #    angles = np.zeros((1, positions.shape[1]))
    #    for i in range(angles.shape[1] - 1):
    #        translated_pt = positions[:, i + 1] - positions[:, i]
    #        angles[0, i] = np.arctan2(translated_pt[1], translated_pt[0])

    #    # 4.) Now get the average magnitude and angle change:
    #    avg_magnitude = np.sum(mags) / (positions.shape[1] - 1)
    #    avg_angle = np.sum(angles) / (angles.shape[1] - 1)

    #    return avg_angle, avg_magnitude

    def approximate_surface_coverage(self, query_option: object) -> float:
        """Compute the approximate surface-area coverage."""
        # 1.) Get option's object-positions:
        positions = query_option.toppings
        # 2.) Get the pertinent query attributes:
        c = query_option.crust_thickness
        t_s = query_option.topping_size
        d = query_option.diameter
        # 3.) Compute pertinent geometric params:
        area_per_topping = np.pi * (t_s / 2.0) ** 2
        viable_surface_radius = (d / 2.0) - (c + (t_s / 2.0))
        # 4.) Get the magnitudes of each object from the surface origin:
        sq_sum = np.sqrt(positions[0, :] ** 2 + positions[1, :] ** 2)
        # 5.) Get the indices of magnitudes <= viable surface radius:
        inds = np.argwhere(sq_sum <= viable_surface_radius)

        # 6.) Find toppings which overlap one another:
        xy_tuples = np.array(list(zip(positions[0, inds], positions[1, inds])))

        xy_combs = list(combinations(xy_tuples, 2))
        topping_dists = np.array(
            [
                np.linalg.norm(xy_combs[i][0] - xy_combs[i][1])
                for i in range(len(xy_combs))
            ]
        )
        # Avoid division-by-zero errors:
        topping_dists = np.where(topping_dists == 0, 0.0001, topping_dists)
        # 7.) Heuristically compute total overlap area:
        overlapping_area = np.where(
            topping_dists < t_s, area_per_topping * np.exp(-topping_dists), 0
        )
        overlapping_area = np.sum(overlapping_area)

        # 5.) Compute the approximation:
        approx_absolute_coverage = (
            area_per_topping * inds.shape[0] - overlapping_area
        )
        coverage = (
            approx_absolute_coverage / (np.pi * viable_surface_radius) ** 2
        )
        return coverage

    def approximate_surface_coverage_ignore_overlap(
        self, query_option: object
    ) -> float:
        """Compute the approximate surface-area coverage."""
        # 1.) Get option's object-positions:
        positions = query_option.toppings
        # 2.) Get the pertinent query attributes:
        c = query_option.crust_thickness
        t_s = query_option.topping_size
        d = query_option.diameter
        viable_surface_radius = (d / 2.0) - (c + (t_s / 2.0))
        # 3.) Get the magnitudes:
        sq_sum = np.sqrt(positions[0, :] ** 2 + positions[1, :] ** 2)
        # 4.) Get the indices of magnitudes <= viable surface radius:
        inds = np.argwhere(sq_sum <= viable_surface_radius)
        # 5.) Compute the approximation:
        coverage = ((np.pi * (t_s / 2.0) ** 2) * inds.shape[0]) / (
            np.pi * viable_surface_radius
        ) ** 2
        return coverage
