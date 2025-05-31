import json

import numpy as np
import pandas as pd

from amznoptim.utils.preprocess import (
    compute_waiting_times,
    fetch_route_matrix,
    fetch_vehicle_info,
)


class DepotVRPBase:
    """Base class for Vehicle Routing Problem (VRP) solvers.
    This class provides the basic structure and methods for processing
    depot, order, and address data, as well as setting hyperparameters
    and solver configurations.
    Subclasses should implement the `solve` and `generate_plan` methods
    to provide specific VRP solving logic and plan generation.

    Attributes:
        depots (list[dict]): List of depot data, each containing depot information.
        orders (list[dict]): List of order data, each containing order information.
        stops (dict[str, list[str]]): Dictionary containing stop id and address.
        addresses (list[str]): List of addresses derived from depots and stops.
        weights (list[float]): List of package weights in grams for each order.
        volumes (list[float]): List of package volumes in cubic milimeter for each order.
        order_waiting_times (list[int]): List of waiting times in seconds for each order.
        route_durations (np.ndarray): Matrix of travel durations in meters between addresses.
        route_distances (np.ndarray): Matrix of travel distances in seconds between addresses.
        vehicles (list[tuple]): List of vehicle information, each containing vehicle model,
            weight capacity, volume capacity, and cruising distance.
        vehicles_per_depot (list[list[tuple]]): List of vehicles available at each depot.
        depot_vehicle_mapping (dict[int, int]): Mapping of vehicle index to depot index.
        dept_time (pd.Timestamp): Departure time for the delivery.
        model (cp_model.CpModel): OR-Tools CP-SAT model for the VRP.
        solver (cp_model.CpSolver): OR-Tools CP-SAT solver for the VRP.
        alpha (int): Hyperparameter for the penalty of order waiting time.
        beta (int): Hyperparameter for the penalty for using an extra vehicle.
        stopping_time (list[int]): List of stopping times at each stop in seconds.
        max_duty_time (int): Maximum delivery time for each vehicle in seconds.
    """

    def __init__(
        self,
        depot_data: list[dict],
        order_data: list[dict],
        address_data: dict[str, list[str]],
    ):
        """Initialize the DepotVRP with depot, order, and address data.

        For input data format, refer to the documentation at
        https://github.com/KagurazakaHinagi/AMZNOptim_MAT381/blob/main/docs/input.md

        Args:
            depot_data (list[dict]): List of depot data, each containing depot information.
            order_data (list[dict]): List of order data, each containing order information.
            address_data (dict[str, list[str]]): DIctionary containing stop id and address.
        """
        self.depots = depot_data
        self.orders = order_data
        self.stops = address_data
        self.addresses = []
        self.weights = []
        self.volumes = []
        self.order_waiting_times = []
        self.route_durations = np.array([[]])
        self.route_distances = np.array([[]])
        self.vehicles = []
        self.vehicles_per_depot = []
        self.depot_vehicle_mapping = {}
        self.dept_time = pd.Timestamp.now()
        self.model = None
        self.alpha = 0  # Hyperparameter: penalty of order waiting time
        self.beta = 0  # Hyperparameter: penalty for using an extra vehicle
        self.stopping_time = []  # Stopping time at each order
        self.max_duty_time = 0  # Maximum delivery time for each vehicle

    def set_hyperparams(self, *, alpha, beta):
        """
        Set hyperparameters for the objective function.
        """
        self.alpha = alpha or self.alpha
        self.beta = beta or self.beta

    def set_solver_max_time(self, max_time: int):
        """
        Set the maximum time in seconds for the solver to run.
        If the solver exceeds this time, it will stop and return the
        best solution found so far.
        """
        raise NotImplementedError(
            "This method should be implemented in subclasses to set the maximum solver time."
        )

    def set_stopping_time(self, stopping_time: list[int] | int):
        """
        Set the stopping time in seconds at each stop.
        If a single integer is provided, it will be used for all stops.
        """
        if isinstance(stopping_time, int):
            self.stopping_time = [0] * len(self.depots) + [stopping_time] * (
                len(self.addresses) - len(self.depots)
            )
        else:
            self.stopping_time = [0] * len(self.depots) + stopping_time

    def set_max_duty_time(self, max_duty_time: int):
        """
        Set the maximum duty time in seconds for all vehicles.
        """
        self.max_duty_time = max_duty_time

    def process_data(
        self,
        vehicle_data_path: str,
        dept_time: pd.Timestamp | None = None,
        matrix_json: str | None = None,
        traffic_aware: bool = False,
        matrix_save_path: str | None = None,
        api_key=None,
    ):
        """
        Process the input data for the VRP in batch.

        For input data format, refer to the documentation at
        https://github.com/KagurazakaHinagi/AMZNOptim_MAT381/blob/main/docs/input.md
        """
        self.process_order_data(dept_time)
        self.process_vehicle_data(vehicle_data_path)
        self.process_route_data(
            matrix_json=matrix_json,
            traffic_aware=traffic_aware,
            dept_time=dept_time,
            api_key=api_key,
            save_path=matrix_save_path,
        )

    def process_order_data(self, dept_time: pd.Timestamp | None = None):
        """
        Process the order data to compute waiting times and prepare
        assresses, weights, and volumes.

        Args:
            dept_time (pd.Timestamp | None): Departure time for the delivery.
                If None, the current time will be used.
        """
        self.dept_time = dept_time or pd.Timestamp.now()
        order_data = compute_waiting_times(self.orders, dept_time=self.dept_time)
        self.addresses = [depot["address"] for depot in self.depots] + self.stops[
            "addresses"
        ]
        self.weights = [order["package_weight"] for order in order_data]
        self.volumes = [order["package_volume"] for order in order_data]
        self.order_waiting_times = [order["waiting_time"] for order in order_data]
        self.stopping_time = [0] * len(self.addresses)

    def process_vehicle_data(self, vehicle_data_path: str):
        """
        Process the vehicle data to fetch the metric information of the available vehicles.

        For input data format, refer to the documentation at
        https://github.com/KagurazakaHinagi/AMZNOptim_MAT381/blob/main/docs/input.md

        Args:
            vehicle_data_path (str): Path to the vehicle data file.
        """
        self.vehicles_per_depot = []
        vehicle_idx = 0
        for depot_idx, depot in enumerate(self.depots):
            depot_vehicles = fetch_vehicle_info(depot, vehicle_data_path)
            self.vehicles_per_depot.append(depot_vehicles)

            for _ in depot_vehicles:
                self.depot_vehicle_mapping[vehicle_idx] = depot_idx
                vehicle_idx += 1

        self.vehicles = [
            vehicle
            for depot_vehicles in self.vehicles_per_depot
            for vehicle in depot_vehicles
        ]

    def process_route_data(
        self,
        matrix_json: str | None = None,
        traffic_aware: bool = False,
        dept_time: pd.Timestamp | None = None,
        save_path: str | None = None,
        api_key=None,
    ):
        """
        Process the addresses to fetch the route matrix.

        For input data format, refer to the documentation at
        https://github.com/KagurazakaHinagi/AMZNOptim_MAT381/blob/main/docs/input.md

        Args:
            matrix_json (str | None): Path to the route matrix JSON file generated
                by Google Maps RouteMatrix API. If None, the route matrix will be
                directly fetched from the API using the provided API key.
            traffic_aware (bool): Whether to use traffic-aware routing. This option
                requires a valid departure time and would cost at a higher rate
                when using Google Maps API.
            dept_time (pd.Timestamp | None): Departure time for the delivery.
                If None, the current time will be used.
            save_path (str | None): Path to save the computed route matrix JSON file.
            api_key (str | None): Google Maps API key for fetching route data.

        Raises:
            ValueError: If the addresses are not processed, or if the number of depots
                in the route matrix does not match the number of depots provided.
        """
        if not self.addresses:
            raise ValueError("Addresses are not set. Please process order data first.")
        route_matrix, depot_cnt = fetch_route_matrix(
            self.depots,
            self.addresses[len(self.depots) :],
            traffic_aware,
            dept_time,
            matrix_json,
            save_path,
            api_key=api_key,
        )
        if depot_cnt != len(self.depots):
            raise ValueError(
                "The number of depots in the route matrix does not match the number of depots provided."
            )
        self.route_durations = route_matrix["duration_seconds"]
        self.route_distances = route_matrix["distance_meters"]

    def validate(self):
        """
        Validate the input data for the VRP.
        """
        if not self.addresses:
            raise ValueError("Addresses are not set. Please process order data first.")
        if not self.vehicles:
            raise ValueError(
                "Vehicle information is not set. Please process vehicle data first."
            )
        if len(self.route_distances) == 0 or len(self.route_durations) == 0:
            raise ValueError(
                "Route distances are not set. Please process route data first."
            )
        if len(self.addresses) != len(self.route_durations):
            raise ValueError(
                "The number of addresses does not match the number of route durations."
            )

    def solve(self):
        """
        Run the solver to find the optimal routes for the VRP.
        This method should be implemented in subclasses.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")

    def generate_plan(self, solution, save_path: str | None = None):
        """
        Generate a human-readable plan from the solution.

        Args:
            solution (dict): The solution returned by the solver.
            save_path (str | None): Path to save the generated plan as a
                JSON-formatted file.

        Returns:
            list[dict]: A list of depot plans, each containing vehicle assignments
                and stop details.
        """
        plans = []
        for depot_idx, depot in enumerate(self.depots):
            depot_plan = {
                "depot": depot["id"],
                "depot_address": depot["address"],
                "departure_time": self.dept_time.isoformat(),
            }
            depot_plan["assignments"] = {}
            for k, route in enumerate(solution["routes"]):
                if depot_idx != solution["vehicle_depot_mapping"][k]:
                    continue
                vehicle_info = {}
                vehicle_info["vehicle_model"] = self.vehicles[k][0]
                if solution["vehicle_usage"][k] == 0:
                    vehicle_info["used"] = False
                    depot_plan["assignments"][f"vehicle_{k}"] = vehicle_info
                    continue  # Skip vehicles that is not used

                vehicle_info["used"] = True
                vehicle_info["total_time"] = 0
                vehicle_info["total_distance_mi"] = (
                    str(
                        round(
                            float(
                                sum(
                                    self.route_distances[route[i]][route[i + 1]]
                                    for i in range(len(route) - 1)
                                )
                            )
                            / 1609.34,
                            2,
                        )
                    )
                    + " mi"
                )
                vehicle_info["stops"] = {}
                for j, stop_index in enumerate(route[1:-1]):
                    stop_address = self.addresses[stop_index]
                    packages = solution["package_assignments"][k]
                    package_info = [
                        self.orders[o]["id"]
                        for o in packages
                        if self.orders[o]["address_index"] + 1 == stop_index
                    ]
                    stop_info = {
                        "address": stop_address,
                        "packages": package_info,
                        "travel_time": str(
                            round(
                                float(self.route_durations[route[j]][route[j + 1]])
                                / 60.0,
                                2,
                            )
                        )
                        + " min",
                        "travel_distance": str(
                            round(
                                float(self.route_distances[route[j]][route[j + 1]])
                                / 1609.34,
                                2,
                            )
                        )
                        + " mi",
                        "stopover_time": str(
                            round(self.stopping_time[stop_index] / 60.0, 2)
                        )
                        + " min",
                    }
                    vehicle_info["stops"][f"stop_{j}"] = stop_info
                    vehicle_info["total_time"] += (
                        self.route_durations[route[j]][route[j + 1]]
                        + self.stopping_time[stop_index]
                    )

                vehicle_info["total_time"] = (
                    str(round(vehicle_info["total_time"] / 3600.0, 2)) + " hrs"
                )
                depot_plan["assignments"][f"vehicle_{k}"] = vehicle_info

            plans.append(depot_plan)

        if save_path:
            with open(save_path, "w") as f:
                json.dump(plans, f, indent=4)

        return plans

