import json

from ortools.sat.python import cp_model

from amznoptim.solver.base import DepotVRPBase


class DepotVRPCpBase(DepotVRPBase):
    """
    Base class for Depot Vehicle Routing Problem (VRP) solver using
    Google OR-Tools CP-SAT solver.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the DepotVRPCpModel with default parameters.
        This class should be extended by specific VRP implementations.
        """
        super().__init__(*args, **kwargs)
        self.model = cp_model.CpModel()
        self.solver = cp_model.CpSolver()

    def set_solver_max_time(self, max_time: int):
        """
        Set the maximum time in seconds for the solver to run.
        If the solver exceeds this time, it will stop and return the
        best solution found so far.
        """
        self.solver.parameters.max_time_in_seconds = max_time


class DepotVRPCpRegular(DepotVRPCpBase):
    """
    Vehicle Routing Problem (VRP) solver for Amazon regular delivery services
    using Google OR-Tools CP-SAT solver.
    """

    def __init__(
        self,
        depot_data: list[dict],
        order_data: list[dict],
        address_data: dict[str, list[str]],
    ):
        """
        Initialize the DepotVRPCpRegular with depot, order, and address data.
        """
        super().__init__(depot_data, order_data, address_data)
        self.max_duty_time = (
            28800  # Maximum delivery time for each vehicle (default: 8 hours)
        )

    def solve(self):
        """
        Run the solver to find the optimal routes for the VRP.
        """
        self.validate()

        num_depots = len(self.depots)
        num_nodes = len(self.addresses)
        num_packages = len(self.orders)

        # Decision variables
        x = {}  # x[i][j][k] = 1 if vehicle k travels from stop i to stop j directly
        y = {}  # y[o][k] = 1 if vehicle k serves order o
        z = {}  # z[k] = 1 if vehicle k is used
        u = {}  # u[i][k] = arrival time of stop i at vehicle k

        # Create decision variables
        for k, (_, weight_cap, volume_cap, cruising_dist) in enumerate(self.vehicles):
            z[k] = self.model.NewBoolVar(f"z_{k}")
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if i != j:
                        x[i, j, k] = self.model.NewBoolVar(f"x_{i}_{j}_{k}")
                u[i, k] = self.model.NewIntVar(0, num_nodes, f"u_{i}_{k}")
            for o in range(num_packages):
                y[o, k] = self.model.NewBoolVar(f"y_{o}_{k}")

        # Constraints
        # Constraint 1: Depot Assignment
        for k in range(len(self.vehicles)):
            depot_idx = self.depot_vehicle_mapping[k]

            # 1a. Vehicle k can only leave from and return to its assigned depot
            self.model.Add(
                sum(x[depot_idx, j, k] for j in range(num_depots, num_nodes)) == z[k]
            )
            self.model.Add(
                sum(x[i, depot_idx, k] for i in range(num_depots, num_nodes)) == z[k]
            )

            # 1b. Vehicle cannot travel between depots
            for other_depot in range(num_depots):
                if other_depot != depot_idx:
                    for j in range(num_nodes):
                        if j != other_depot:
                            self.model.Add(x[other_depot, j, k] == 0)
                            self.model.Add(x[j, other_depot, k] == 0)

        # Constraint 2: Package Assignment and Routing
        for k in range(len(self.vehicles)):
            for o in range(num_packages):
                stop_index = self.orders[o]["address_index"] + num_depots
                # 2a. If the package o is served by vehicle k,
                # then we need to ensure an incoming flow to the stop.
                self.model.Add(
                    sum(
                        x[i, stop_index, k] for i in range(num_nodes) if i != stop_index
                    )
                    >= y[o, k]
                )

            depot_idx = self.depot_vehicle_mapping
            for h in range(num_depots, num_nodes):
                # 2b. For each stop, the incoming flow must equal the outgoing flow
                self.model.Add(
                    sum(x[i, h, k] for i in range(num_nodes) if i != h)
                    == sum(x[h, j, k] for j in range(num_nodes) if j != h)
                )

        # Constraint 3: Vehicle Capacity
        for k, (_, weight_cap, volume_cap, cruising_dist) in enumerate(self.vehicles):
            self.model.Add(
                sum(self.weights[o] * y[o, k] for o in range(num_packages))
                <= weight_cap
            )
            self.model.Add(
                sum(self.volumes[o] * y[o, k] for o in range(num_packages))
                <= volume_cap
            )

        # Constraint 4: Vehicle Usage
        for k in range(len(self.vehicles)):
            package_assignments = [y[o, k] for o in range(num_packages)]
            self.model.AddMaxEquality(z[k], package_assignments)

        # Constraint 5: Miller-Tucker-Zemlin (MTZ) Subtour Elimination
        # See https://en.wikipedia.org/wiki/Travelling_salesman_problem
        for k in range(len(self.vehicles)):
            for i in range(num_depots, num_nodes):
                for j in range(num_depots, num_nodes):
                    if i != j:
                        self.model.Add(
                            u[i, k] + 1 <= u[j, k] + (num_nodes) * (1 - x[i, j, k])
                        )

        # Constraint 6: Time Constraints (travel time + stopover time)
        for k in range(len(self.vehicles)):
            travel_time = sum(
                int(self.route_durations[i][j]) * x[i, j, k]
                for i in range(num_nodes)
                for j in range(num_nodes)
                if i != j
            )

            stopover_time = 0
            unique_stops = set(
                self.orders[o]["address_index"] + num_depots
                for o in range(num_packages)
            )
            for stop_index in unique_stops:
                stop_visited = self.model.NewBoolVar(f"stop_visited_{stop_index}_{k}")
                packages_at_stop = [
                    o
                    for o in range(num_packages)
                    if self.orders[o]["address_index"] + num_depots == stop_index
                ]
                for o in packages_at_stop:
                    # 6b. If package o is assigned to vehicle k, then the stop must be visited
                    self.model.Add(stop_visited >= y[o, k])
                # 6c. If the stop is visited,
                # then at least one package must be assigned to vehicle k
                self.model.Add(stop_visited <= sum(y[o, k] for o in packages_at_stop))
                stopover_time += int(self.stopping_time[stop_index]) * stop_visited
            # 6a. Total travel time + stopover time must not exceed max duty time
            self.model.Add(travel_time + stopover_time <= self.max_duty_time)

        # Constraint 7: Distance Constraint
        # The total travel distance must not exceed the maximum cruising
        # distance of the vehicle
        for k, (_, _, _, cruising_dist) in enumerate(self.vehicles):
            self.model.Add(
                sum(
                    int(self.route_distances[i][j]) * x[i, j, k]
                    for i in range(num_nodes)
                    for j in range(num_nodes)
                    if i != j
                )
                <= cruising_dist
            )

        # Constraint 8: Assignment Constraints
        # Each package is assigned to exactly one vehicle
        for o in range(num_packages):
            self.model.Add(sum(y[o, k] for k in range(len(self.vehicles))) == 1)

        # Objective: Minimize
        # total travel time + beta * truck usage - alpha * order waiting time
        time_cost = sum(
            self.route_durations[i][j] * x[i, j, k]
            for k in range(len(self.vehicles))
            for i in range(num_nodes)
            for j in range(num_nodes)
            if i != j
        )
        priority_cost = sum(
            self.order_waiting_times[o] * y[o, k]
            for k in range(len(self.vehicles))
            for o in range(num_packages)
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
            package_assignments = []

            for k in range(len(self.vehicles)):
                depot_idx = self.depot_vehicle_mapping[k]
                route = [depot_idx]
                curr = depot_idx
                while True:
                    nxt = next(
                        (
                            j
                            for j in range(num_nodes)
                            if j != curr and self.solver.Value(x[curr, j, k])
                        ),
                        None,
                    )
                    if nxt is None or nxt == depot_idx:
                        route.append(depot_idx)
                        break
                    route.append(nxt)
                    curr = nxt
                routes.append(route)

                vehicle_packages = [
                    o for o in range(num_packages) if self.solver.Value(y[o, k]) == 1
                ]
                package_assignments.append(vehicle_packages)

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
                    stop_index = self.orders[o]["address_index"] + num_depots
                    if self.solver.Value(y[o, k]) and stop_index < len(
                        self.stopping_time
                    ):
                        total_stopover_time += self.stopping_time[
                            stop_index
                        ] * self.solver.Value(y[o, k])

            return {
                "status": "optimal" if status == cp_model.OPTIMAL else "feasible",
                "routes": routes,
                "package_assignments": package_assignments,
                "travel_time": total_travel_time,
                "stopover_time": total_stopover_time,
                "vehicle_usage": [
                    self.solver.Value(z[k]) for k in range(len(self.vehicles))
                ],
                "vehicle_depot_mapping": self.depot_vehicle_mapping,
                "priority_cost": sum(
                    self.order_waiting_times[o] * self.solver.Value(y[o, k])
                    for k in range(len(self.vehicles))
                    for o in range(num_packages)
                ),
                "total_cost": self.solver.ObjectiveValue(),
                "solver": "CpSAT",
            }
        raise ValueError("No feasible solution found.")