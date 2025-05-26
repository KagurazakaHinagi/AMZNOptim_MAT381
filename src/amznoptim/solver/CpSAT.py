import numpy as np
import pandas as pd
from ortools.sat.python import cp_model

from amznoptim.utils.preprocess import (
    compute_waiting_times,
    fetch_route_matrix,
    fetch_route_matrix_from_json,
    fetch_vehicle_info,
)


class SingleDepotVRPRegular:
    """Class to solve the Single Depot Vehicle Routing Problem (SDVRP) using Google OR-Tools.

    This class uses the Constraint Programming (CP) solver from Google OR-Tools to find an optimal
    solution for the SDVRP. The problem is defined by a depot and a set of stops, each with a specific
    weight and volume. The goal is to minimize the total travel time while satisfying the constraints
    of vehicle capacity and order delivery.
    The class allows for setting hyperparameters for the optimization, such as penalties for order
    waiting time and vehicle usage. It also allows for setting a stopping time at each order and a
    maximum delivery time for each schedule.

    Args:
        depot_data (dict): Contains information about the depot,
        including address and the available vehicle types.

        stop_data (dict): Contains information about the stops,
        including address, weight, and volume. Every numerical elements need to be
        preprocessed to use the metric system instead of imperial system.
        The first order in the list is considered as the depot. (not a real order)
    """

    def __init__(self, depot_data: list[dict], stop_data: dict):
        self.depot = depot_data[0]
        self.stops = stop_data
        self.addresses = []
        self.weights = []
        self.volumes = []
        self.order_waiting_times = []
        self.route_durations = []
        self.route_distances = []
        self.vehicles = []
        self.model = cp_model.CpModel()
        self.solver = cp_model.CpSolver()
        self.alpha = 0  # Hyperparameter: penalty of order waiting time
        self.beta = 0  # Hyperparameter: penalty for using a vehicle
        self.stopping_time = [0] * len(self.addresses)  # Stopping time at each order
        self.max_duty_time = np.inf  # Maximum delivery time for each schedule

    def set_hyperparams(self, *, alpha: float | None = None, beta: float | None = None):
        self.alpha = alpha or self.alpha
        self.beta = beta or self.beta

    def set_stopping_time(self, stopping_time: float):
        self.stopping_time = stopping_time

    def set_max_duty_time(self, max_duty_time: float):
        self.max_duty_time = max_duty_time

    def process_data(
        self,
        vehicle_data_path: str,
        dept_time: pd.Timestamp | None = None,
        matrix_json: str | None = None,
        traffic_aware: bool = False,
        api_key=None,
    ):
        self.process_stop_data(dept_time)
        self.process_vehicle_data(vehicle_data_path)
        self.process_route_data(
            matrix_json=matrix_json,
            traffic_aware=traffic_aware,
            dept_time=dept_time,
            api_key=api_key,
        )

    def process_stop_data(self, dept_time: pd.Timestamp | None = None):
        self.addresses = [self.depot["address"]] + [
            stop["address"] for _, stop in self.stops.items()
        ]
        self.weights = [0] + [stop["total_weight"] for _, stop in self.stops.items()]
        self.volumes = [0] + [stop["total_volume"] for _, stop in self.stops.items()]
        self.order_waiting_times = [0]
        dept_time = dept_time or pd.Timestamp.now()
        stops_with_waiting_time = compute_waiting_times(self.stops, dept_time=dept_time)
        for _, stop in stops_with_waiting_time.items():
            self.order_waiting_times.append(stop["max_waiting_time"])

    def process_vehicle_data(self, vehicle_data_path: str):
        self.vehicles = fetch_vehicle_info(self.depot, vehicle_data_path)

    def process_route_data(
        self,
        matrix_json: str | None = None,
        traffic_aware: bool = False,
        dept_time: pd.Timestamp | None = None,
        api_key=None,
    ):
        if matrix_json:
            route_matrix = fetch_route_matrix_from_json(matrix_json, 1)[0]
            self.route_durations = route_matrix["duration_seconds"]
            self.route_distances = route_matrix["distance_meters"]
            return
        route_matrix = fetch_route_matrix(
            self.depot, self.stops, traffic_aware, dept_time, api_key=api_key
        )[0]
        self.route_durations = route_matrix["duration_seconds"]
        self.route_distances = route_matrix["distance_meters"]

    def validate(self):
        pass

    def solve(self):
        num_nodes = len(self.addresses)
        # Decision variables
        x = {}  # x[i][j][k] = 1 if vehicle k travels from order i to order j directly
        y = {}  # y[j][k] = 1 if vehicle k serves order j
        z = {}  # z[k] = 1 if vehicle k is used
        u = {}  # u[i][k] = arrival time of order i at vehicle k
        if not self.vehicles or not self.stops:
            raise ValueError("Vehicle or order information is not set.")
        for k, (weight_cap, volume_cap, cruising_dist) in enumerate(self.vehicles):
            z[k] = self.model.NewBoolVar(f"z_{k}")
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if i != j:
                        x[i, j, k] = self.model.NewBoolVar(f"x_{i}_{j}_{k}")
            for j in range(1, num_nodes):
                y[j, k] = self.model.NewBoolVar(f"y_{j}_{k}")
                # Constraint 1.1: Routing
                # If j is served by vehicle k, then there's one path to j from some i
                self.model.Add(
                    sum(x[i, j, k] for i in range(num_nodes) if i != j) == y[j, k]
                )
            # Constraint 1.2: Routing (Inflow and Outflow)
            # Each order must be entered and exited exactly once
            for h in range(1, num_nodes):
                self.model.Add(
                    sum(x[i, h, k] for i in range(num_nodes) if i != h)
                    == sum(x[h, j, k] for j in range(num_nodes) if j != h)
                )
            # Constraint 2: Loading Capacity
            # Vehicle k must not exceed its capacity (payload and volume limit)
            self.model.Add(
                sum(self.weights[j] * y[j, k] for j in range(1, num_nodes))
                <= weight_cap
            )
            self.model.Add(
                sum(self.volumes[j] * y[j, k] for j in range(1, num_nodes))
                <= volume_cap
            )
            # Constraint 3: Depot Flow
            # If vehicle k is used, then it must leave and return to the depot once
            self.model.Add(
                sum(x[0, j, k] for j in range(num_nodes) if j != 0) <= num_nodes * z[k]
            )
            self.model.Add(
                sum(x[i, 0, k] for i in range(num_nodes) if i != 0) <= num_nodes * z[k]
            )
            # Constraint 4: Miller-Tucker-Zemlin (MTZ) Subtour Elimination
            # See https://en.wikipedia.org/wiki/Travelling_salesman_problem
            for i in range(num_nodes):
                u[i, k] = self.model.NewIntVar(0, num_nodes, f"u_{i}_{k}")
            for i in range(1, num_nodes):
                for j in range(1, num_nodes):
                    if i != j:
                        self.model.Add(
                            u[i, k] + 1 <= u[j, k] + (num_nodes) * (1 - x[i, j, k])
                        )
            # Constraint 5: Maximum duty time
            # The total travel time and stopover time must not exceed
            # the maximum duty time of the driver
            travel_time = sum(
                self.route_durations[i][j] * x[i, j, k]
                for i in range(num_nodes)
                for j in range(num_nodes)
                if i != j
            )
            stopover_time = sum(
                self.stopping_time * y[j, k] for j in range(1, num_nodes)
            )
            self.model.Add(travel_time + stopover_time <= self.max_duty_time)
            # Constraint 6: Maximum distance
            # The total travel distance must not exceed the maximum cruising
            # distance of the vehicle
            self.model.Add(
                sum(
                    self.route_distances[i][j] * x[i, j, k]
                    for i in range(num_nodes)
                    for j in range(num_nodes)
                    if i != j
                )
                <= cruising_dist
            )

        # Constraint 7: Each customer must be served exactly once
        for j in range(1, num_nodes):
            self.model.Add(sum(y[j, k] for k in range(len(self.vehicles))) == 1)

        # Objective: Minimize
        # total travel time * factor + beta * truck usage - alpha * order waiting time
        time_cost = sum(
            self.route_durations[i][j] * x[i, j, k]
            for k in range(len(self.vehicles))
            for i in range(num_nodes)
            for j in range(num_nodes)
            if i != j
        )
        priority_cost = sum(
            self.order_waiting_times[j] * y[j, k]
            for k in range(len(self.vehicles))
            for j in range(1, num_nodes)
        )
        vehicle_cost = sum(z[k] for k in range(len(self.vehicles)))
        self.model.Minimize(
            time_cost + self.beta * vehicle_cost - self.alpha * priority_cost
        )

        # Solve the model
        self.solver.parameters.max_time_in_seconds = 300
        status = self.solver.Solve(self.model)

        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            return self.solver.Value
        raise ValueError("No feasible solution found.")

    def interpret_solution(self, solution, save_path=None):
        pass
