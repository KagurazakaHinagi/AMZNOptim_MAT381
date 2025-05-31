import numpy as np
import pandas as pd
from ortools.sat.python import cp_model

from amznoptim.utils.preprocess import (
    compute_waiting_times,
    fetch_route_matrix,
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

    def __init__(
        self,
        depot_data: list[dict],
        order_data: list[dict],
        address_data: dict[str, list[str]],
    ):
        self.depot = depot_data[0]
        self.orders = order_data
        self.stops = address_data
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
        self.stopping_time = []  # Stopping time at each order
        self.max_duty_time = 28800  # Maximum delivery time for each schedule

    def set_hyperparams(self, *, alpha: int | None = None, beta: int | None = None):
        self.alpha = alpha or self.alpha
        self.beta = beta or self.beta

    def set_solver_max_time(self, max_time: int):
        self.solver.parameters.max_time_in_seconds = max_time

    def set_stopping_time(self, stopping_time: list[int] | int):
        if isinstance(stopping_time, int):
            self.stopping_time = [0] + [stopping_time] * (len(self.addresses) - 1)
        else:
            self.stopping_time = stopping_time

    def set_max_duty_time(self, max_duty_time: int):
        self.max_duty_time = max_duty_time

    def process_data(
        self,
        vehicle_data_path: str,
        dept_time: pd.Timestamp | None = None,
        matrix_json: str | None = None,
        traffic_aware: bool = False,
        save_path: str | None = None,
        api_key=None,
    ):
        self.process_order_data(dept_time)
        self.process_vehicle_data(vehicle_data_path)
        self.process_route_data(
            matrix_json=matrix_json,
            traffic_aware=traffic_aware,
            dept_time=dept_time,
            api_key=api_key,
            save_path=save_path,
        )

    def process_order_data(self, dept_time: pd.Timestamp | None = None):
        dept_time = dept_time or pd.Timestamp.now()
        order_data = compute_waiting_times(
            self.orders, dept_time=dept_time
        )
        self.addresses = [self.depot["address"]] + self.stops["addresses"]
        self.weights = [order["package_weight"] for order in order_data]
        self.volumes = [order["package_volume"] for order in order_data]
        self.order_waiting_times = [order["waiting_time"] for order in order_data]
        self.stopping_time = [0] * len(self.addresses)

    def process_vehicle_data(self, vehicle_data_path: str):
        self.vehicles = fetch_vehicle_info(self.depot, vehicle_data_path)

    def process_route_data(
        self,
        matrix_json: str | None = None,
        traffic_aware: bool = False,
        dept_time: pd.Timestamp | None = None,
        save_path: str | None = None,
        api_key=None,
    ):
        route_matrix = fetch_route_matrix(
            [self.depot],
            self.addresses,
            traffic_aware,
            dept_time,
            matrix_json,
            save_path,
            api_key=api_key,
        )[0]
        self.route_durations = route_matrix["duration_seconds"]
        self.route_distances = route_matrix["distance_meters"]

    def validate(self):
        pass

    def solve(self, save_path: str | None = None):
        num_nodes = len(self.addresses)
        num_packages = len(self.orders)
        # Decision variables
        x = {}  # x[i][j][k] = 1 if vehicle k travels from stop i to stop j directly
        y = {}  # y[o][k] = 1 if vehicle k serves order o
        z = {}  # z[k] = 1 if vehicle k is used
        u = {}  # u[i][k] = arrival time of stop i at vehicle k
        if not self.vehicles or not self.stops:
            raise ValueError("Vehicle or order information is not set.")

        # Create decision variables
        for k, (weight_cap, volume_cap, cruising_dist) in enumerate(self.vehicles):
            z[k] = self.model.NewBoolVar(f"z_{k}")
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if i != j:
                        x[i, j, k] = self.model.NewBoolVar(f"x_{i}_{j}_{k}")
                u[i, k] = self.model.NewIntVar(0, num_nodes, f"u_{i}_{k}")
            for o in range(num_packages):
                y[o, k] = self.model.NewBoolVar(f"y_{o}_{k}")

        # Constraints
        # Constraint 1: Routing
        for k in range(len(self.vehicles)):
            for o in range(num_packages):
                stop_index = self.orders[o]["address_index"] + 1  # +1 for depot
                # 1a. If package o is served by vehicle k, then there must be an incoming edge to the stop
                self.model.Add(
                    sum(x[i, stop_index, k] for i in range(num_nodes) if i != stop_index) >= y[o, k]
                )

            # Flow conservation (incoming flow = outgoing flow)
            for h in range(num_nodes):
                if h == 0: # 1b. Depot flow
                    self.model.Add(
                        sum(x[0, j, k] for j in range(1, num_nodes) if j != 0)  ==
                        sum(x[i, 0, k] for i in range(1, num_nodes) if i != 0)
                    )
                else: # 1c. Non-depot flow
                    self.model.Add(
                        sum(x[i, h, k] for i in range(num_nodes) if i != h) ==
                        sum(x[h, j, k] for j in range(num_nodes) if j != h)
                    )
        # Constraint 2 & 3: Vehicle Capacity and Usage
        for k, (weight_cap, volume_cap, cruising_dist) in enumerate(self.vehicles):
            # Constraint 2: Vehicle Capacity
            self.model.Add(
            sum(self.weights[o] * y[o, k] for o in range(num_packages)) <= weight_cap
            )
            self.model.Add(
                sum(self.volumes[o] * y[o, k] for o in range(num_packages)) <= volume_cap
            )

            # Constraint 3: Vehicle Usage
            # z[k] = 1 iff any package is served
            package_assignments = [y[o, k] for o in range(num_packages)]
            self.model.AddMaxEquality(z[k], package_assignments)

            # Depot constraint: vehicle leaves/returns iff it's used
            self.model.Add(sum(x[0, j, k] for j in range(1, num_nodes)) == z[k])
            self.model.Add(sum(x[i, 0, k] for i in range(1, num_nodes)) == z[k])


        # Constraint 4: Miller-Tucker-Zemlin (MTZ) Subtour Elimination
        # See https://en.wikipedia.org/wiki/Travelling_salesman_problem
        for k in range(len(self.vehicles)):
            for i in range(1, num_nodes):
                for j in range(1, num_nodes):
                    if i != j:
                        self.model.Add(
                            u[i, k] + 1 <= u[j, k] + (num_nodes) * (1 - x[i, j, k])
                        )
        # Constraint 5: Time Constraints (travel time + stopover time)
        for k in range(len(self.vehicles)):
            travel_time = sum(
                int(self.route_durations[i][j]) * x[i, j, k]
                for i in range(num_nodes) for j in range(num_nodes) if i != j
            )

            stopover_time = 0
            unique_stops = set(self.orders[o]["address_index"] + 1 for o in range(num_packages))
            for stop_index in unique_stops:
                stop_visited = self.model.NewBoolVar(f"stop_visited_{stop_index}_{k}")
                packages_at_stop = [o for o in range(num_packages)
                                    if self.orders[o]["address_index"] + 1 == stop_index]
                for o in packages_at_stop:
                    # 5b. If package o is assigned to vehicle k, then the stop must be visited
                    self.model.Add(stop_visited >= y[o, k])
                # 5c. If the stop is visited, then at least one package must be assigned to vehicle k
                self.model.Add(stop_visited <= sum(y[o, k] for o in packages_at_stop))
                stopover_time += int(self.stopping_time[stop_index]) * stop_visited
            # 5a. Total travel time + stopover time must not exceed max duty time
            self.model.Add(
                travel_time + stopover_time <= self.max_duty_time
            )


        # Constraint 6: Distance Constraint
        # The total travel distance must not exceed the maximum cruising
        # distance of the vehicle
        for k, (_, _, cruising_dist) in enumerate(self.vehicles):
            self.model.Add(
                sum(int(self.route_distances[i][j]) * x[i, j, k]
                    for i in range(num_nodes) for j in range(num_nodes) if i != j)
                <= cruising_dist)

        # Constraint 7 & 8: Assignment Constraints
        # Constraint 7: Each package is assigned to exactly one vehicle
        for o in range(num_packages):
            self.model.Add(sum(y[o, k] for k in range(len(self.vehicles))) == 1)

        # Constraint 8: Each stop is visited exactly once
        for j in range(1, num_nodes):
            self.model.Add(
                sum(x[i, j, k] for i in range(num_nodes) if i != j
                    for k in range(len(self.vehicles))) == 1
            )

        # Objective: Minimize
        # total travel time + beta * truck usage - alpha * order waiting time
        time_cost = sum(
            self.route_durations[i][j] * x[i, j, k]
            for k in range(len(self.vehicles))
            for i in range(num_nodes) for j in range(num_nodes) if i != j
        )
        priority_cost = sum(
            self.order_waiting_times[o] * y[o, k]
            for k in range(len(self.vehicles)) for o in range(num_packages)
        )
        vehicle_cost = sum(z[k] for k in range(len(self.vehicles)))

        self.model.Minimize(
            time_cost + self.beta * vehicle_cost - self.alpha * priority_cost
        )

        # Solve the model
        status = self.solver.Solve(self.model)
        # Format the returned solution
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            routes = []
            for k in range(len(self.vehicles)):
                route = [0]
                curr = 0
                while True:
                    nxt = next(
                        (
                            j
                            for j in range(num_nodes)
                            if j != curr and self.solver.Value(x[curr, j, k])
                        ),
                        None,
                    )
                    if nxt is None or nxt == 0:
                        route.append(0)
                        break
                    route.append(nxt)
                    curr = nxt
                routes.append(route)

            # Calculate travel time and stopover time for the solution
            total_travel_time = sum(
                self.route_durations[i][j] * self.solver.Value(x[i, j, k])
                for k in range(len(self.vehicles))
                for i in range(num_nodes)
                for j in range(num_nodes)
                if i != j
            )
            total_stopover_time = 0
            for k in range(len(self.vehicles)):
                for o in range(num_packages):
                    stop_index = self.orders[o]["address_index"] + 1
                    if self.solver.Value(y[o, k]) and stop_index < len(self.stopping_time):
                        total_stopover_time += self.stopping_time[stop_index] * self.solver.Value(y[o, k])

            return {
                "status": "optimal" if status == cp_model.OPTIMAL else "feasible",
                "routes": routes,
                "travel_time": total_travel_time,
                "stopover_time": total_stopover_time,
                "vehicle_usage": [
                    self.solver.Value(z[k]) for k in range(len(self.vehicles))
                ],
                "priority_cost": sum(
                    self.order_waiting_times[o] * self.solver.Value(y[o, k])
                    for k in range(len(self.vehicles))
                    for o in range(num_packages)
                ),
                "total_cost": self.solver.ObjectiveValue(),
                "solver": "CpSAT",
            }
        raise ValueError("No feasible solution found.")
