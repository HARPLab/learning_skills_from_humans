from __future__ import annotations

import numpy as np


class Pizza:
    """A generic pizza class."""

    def __init__(
        self,
        diameter: float,
        slices: int,
        crust_thickness: float,
        t_size: float,
        topping_placements: np.ndarray,
    ) -> Pizza:
        """Instantiate a pizza."""
        # 1.) Set user-defined attributes:
        self._diameter = diameter
        self._slices = slices
        self._crust_thickness = crust_thickness
        self._topping_size = t_size
        self._topping_placements = topping_placements

    @classmethod
    def from_pizza(cls, pizza) -> Pizza:
        """Instantiate a pizza from a dictionary."""
        pie = Pizza(
            pizza.diameter, pizza.slices, pizza.crust_thickness, pizza.toppings
        )
        return pie

    @classmethod
    def from_dict(
        cls, pizza_attributes: dict, topping_placements: np.ndarray
    ) -> Pizza:
        """Instantiate a pizza from a dictionary."""
        pizza = Pizza(
            pizza_attributes["diameter"],
            pizza_attributes["slices"],
            pizza_attributes["crust_thickness"],
            pizza_attributes["topping_size"],
            topping_placements,
        )
        return pizza

    @classmethod
    def random_init(cls) -> Pizza:
        """Instantiate a pizza from a dictionary."""
        diameter = np.random.randint(low=4, high=20)
        slices = np.random.randint(low=1, high=20)
        crust = np.random.rand(1) * 3
        topping_ct = np.random.randint(low=1, high=50)
        t_range = np.linspace(-1, 1, 10000)
        toppings = np.random.choice(t_range, size=(2, topping_ct))

        pizza = Pizza(diameter, slices, crust, toppings)
        return pizza

    @classmethod
    def from_sample(
        cls, sample: Pizza, topping_placements: np.ndarray
    ) -> Pizza:
        """Instantiate from a sample and particular toppings."""
        p = Pizza(
            sample.diameter,
            sample.slices,
            sample.crust_thickness,
            sample.topping_size,
            topping_placements,
        )
        return p

    def get_form(self) -> dict:
        "Return all attributes of a pizza." ""
        all_relevant = self.__dict__
        return all_relevant

    @property
    def slices(self) -> int:
        """Return the number of pizza slices."""
        return self._slices

    @property
    def topping_size(self) -> float:
        """Get the topping size."""
        return self._topping_size

    @property
    def crust_thickness(self) -> float:
        """Get the pizza's crust thickness in cm."""
        return self._crust_thickness

    @property
    def diameter(self) -> float:
        """Get the pizza's diameter in cm."""
        return self._diameter

    @property
    def toppings(self) -> np.ndarray:
        """Return the topping objects."""
        return np.array(self._topping_placements, copy=True)

    @toppings.setter
    def toppings(self, coordinates: np.ndarray) -> None:
        """Change topping coordinates."""
        assert (
            coordinates.shape[0] == 1
        ), "Coordinates need to be in (1xN) format."
        self._topping_placements = coordinates

    def get_topping_count(self) -> int:
        """Return the number of toppings."""
        return self._topping_placements.shape[1]
