import json

import gurobipy as gp
from gurobipy import GRB

from amznoptim.solver.base import DepotVRPBase

class DepotVRPGurobiBase(DepotVRPBase):
    """
    Base class for Depot Vehicle Routing Problem (VRP) solver using
    Gurobi MIP (Mixed Integer Programming) solver.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = gp.Model("DepotVRP")

    def use_licensed_environment(self, wls_file: str):
        """
        Use a licensed Gurobi environment with the provided WLS license.
        See https://support.gurobi.com/hc/en-us/articles/4409582394769-Google-Colab-Installation-and-Licensing
        """
        license_info = {}
        for line in open(wls_file, "r"):
            if line.strip() and not line.startswith("#"):
                key, value = line.split("=", 1)
                license_info[key.strip()] = value.strip()
        env = gp.Env(params=license_info)
        self.model = gp.Model("DepotVRP", env=env)

    def set_solver_max_time(self, max_time: int):
        self.model.setParam("TimeLimit", max_time)


class DepotVRPGurobiRegular(DepotVRPGurobiBase):
    """
    Vehicle Routing Problem (VRP) solver for Amazon regular delivery services
    using Gurobi MIP solver.
    """
    def __init__(self, *args, **kwargs):
        """
        Initialize the DepotVRPGurobiRegular with depot, order, and address data.
        """
        super().__init__(*args, **kwargs)
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
            z[k] = self.model.addVar(vtype=GRB.BINARY, name=f"z_{k}")
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if i != j:
                        x[i, j, k] = self.model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}_{k}")
                u[i, k] = self.model.addVar(vtype=GRB.INTEGER, lb=0, ub=num_nodes, name=f"u_{i}_{k}")
            for o in range(num_packages):
                y[o, k] = self.model.addVar(vtype=GRB.BINARY, name=f"y_{o}_{k}")
        # Update model to integrate new variables
        self.model.update()

        # Constraints
        # Constraint 1: Depot Assignment
        for k in range(len(self.vehicles)):
            depot_idx = self.depot_vehicle_mapping[k]

            # 1a. Vehicle k can only leave from and return to its assigned depot
            self.model.addConstr(
                gp.quicksum(x[depot_idx, j, k] for j in range(num_depots, num_nodes)) == z[k],
                name=f"depot_outflow_{k}"
            )
            self.model.addConstr(
                gp.quicksum(x[i, depot_idx, k] for i in range(num_depots, num_nodes)) == z[k],
                name=f"depot_inflow_{k}"
            )

            # 1b. Vehicle cannot travel between depots
            for other_depot in range(num_depots):
                if other_depot != depot_idx:
                    for j in range(num_nodes):
                        self.model.addConstr(x[other_depot, j, k] == 0, name=f"no_depot_travel_{other_depot}_{j}_{k}")
                        self.model.addConstr(x[j, other_depot, k] == 0, name=f"no_depot_travel_{j}_{other_depot}_{k}")

        # Constraint 2: Package Assignment and Routing
        for k in range(len(self.vehicles)):
            for o in range(num_packages):
                stop_index = self.orders[o]["address_index"] + num_depots
                # 2a. If the package o is served by vehicle k,
                # then we need to ensure an incoming flow to the stop.
                self.model.addConstr(
                    gp.quicksum(
                        x[i, stop_index, k] for i in range(num_nodes) if i != stop_index
                    ) >= y[o, k],
                    name=f"package_routing_{o}_{k}"
                )

            for h in range(num_depots, num_nodes):
                # 2b. For each stop, the incoming flow must equal the outgoing flow
                self.model.addConstr(
                    gp.quicksum(x[i, h, k] for i in range(num_nodes) if i != h)
                    == gp.quicksum(x[h, j, k] for j in range(num_nodes) if j != h),
                    name=f"flow_conservation_{h}_{k}"
                )

        # Constraint 3: Vehicle Capacity
        for k, (_, weight_cap, volume_cap, cruising_dist) in enumerate(self.vehicles):
            self.model.addConstr(
                gp.quicksum(self.weights[o] * y[o, k] for o in range(num_packages)) <= weight_cap,
                name=f"weight_capacity_{k}"
            )
            self.model.addConstr(
                gp.quicksum(self.volumes[o] * y[o, k] for o in range(num_packages)) <= volume_cap,
                name=f"volume_capacity_{k}"
            )

        # Constraint 4: Vehicle Usage
        for k in range(len(self.vehicles)):
            for o in range(num_packages):
                self.model.addConstr(z[k] >= y[o, k], name=f"vehicle_usage_{k}_{o}")

        # Constraint 5: Miller-Tucker-Zemlin (MTZ) Subtour Elimination
        # See https://en.wikipedia.org/wiki/Travelling_salesman_problem
        for k in range(len(self.vehicles)):
            for i in range(num_depots, num_nodes):
                for j in range(num_depots, num_nodes):
                    if i != j:
                        self.model.addConstr(
                            u[i, k] + 1 <= u[j, k] + num_nodes * (1 - x[i, j, k]),
                            name=f"mtz_{i}_{j}_{k}"
                        )

        # Constraint 6: Time Constraints (travel time + stopover time)
        for k in range(len(self.vehicles)):
            travel_time = gp.quicksum(
                int(self.route_durations[i][j]) * x[i, j, k]
                for i in range(num_nodes)
                for j in range(num_nodes)
                if i != j
            )

            # Create auxiliary variables for stops visited
            stop_visited = {}
            unique_stops = set(
                self.orders[o]["address_index"] + num_depots
                for o in range(num_packages)
            )
            stopover_time = 0
            for stop_index in unique_stops:
                stop_visited[stop_index, k] = self.model.addVar(vtype=GRB.BINARY, name=f"stop_visited_{stop_index}_{k}")
                packages_at_stop = [
                    o
                    for o in range(num_packages)
                    if self.orders[o]["address_index"] + num_depots == stop_index
                ]
                for o in packages_at_stop:
                    # 6b. If package o is assigned to vehicle k, then the stop must be visited
                    self.model.addConstr(stop_visited[stop_index, k] >= y[o, k], name=f"stop_visit_{stop_index}_{k}_{o}")
                # 6c. If the stop is visited, then at least one package must be assigned to vehicle k
                self.model.addConstr(
                    stop_visited[stop_index, k] <= gp.quicksum(y[o, k] for o in packages_at_stop),
                    name=f"stop_package_{stop_index}_{k}"
                )
                stopover_time += int(self.stopping_time[stop_index]) * stop_visited[stop_index, k]

            # 6a. Total travel time + stopover time must not exceed max duty time
            self.model.addConstr(
                travel_time + stopover_time <= self.max_duty_time,
                name=f"duty_time_{k}"
            )

        # Constraint 7: Distance Constraint
        # The total travel distance must not exceed the maximum cruising
        # distance of the vehicle
        for k, (_, _, _, cruising_dist) in enumerate(self.vehicles):
            self.model.addConstr(
                gp.quicksum(
                    int(self.route_distances[i][j]) * x[i, j, k]
                    for i in range(num_nodes)
                    for j in range(num_nodes)
                    if i != j
                ) <= cruising_dist,
                name=f"distance_constraint_{k}"
            )

        # Constraint 8: Assignment Constraints
        # Each package is assigned to exactly one vehicle
        for o in range(num_packages):
            self.model.addConstr(
                gp.quicksum(y[o, k] for k in range(len(self.vehicles))) == 1,
                name=f"package_assignment_{o}"
            )

        # Objective: Minimize
        # total travel time + beta * truck usage - alpha * order waiting time
        time_cost = gp.quicksum(
            self.route_durations[i][j] * x[i, j, k]
            for k in range(len(self.vehicles))
            for i in range(num_nodes)
            for j in range(num_nodes)
            if i != j
        )
        priority_cost = gp.quicksum(
            self.order_waiting_times[o] * y[o, k]
            for k in range(len(self.vehicles))
            for o in range(num_packages)
        )
        vehicle_cost = gp.quicksum(z[k] for k in range(len(self.vehicles)))

        self.model.setObjective(
            time_cost + self.beta * vehicle_cost - self.alpha * priority_cost,
            GRB.MINIMIZE
        )

        # Solve the model
        self.model.optimize()

        # Format the returned solution
        if self.model.status == GRB.OPTIMAL or self.model.status == GRB.SUBOPTIMAL:
            routes = []
            package_assignments = []

            for k in range(len(self.vehicles)):
                depot_idx = self.depot_vehicle_mapping[k]
                route = [depot_idx]
                curr = depot_idx
                while True:
                    nxt = None
                    for j in range(num_nodes):
                        if j != curr and (curr, j, k) in x and x[curr, j, k].X > 0.5:
                            nxt = j
                            break
                    if nxt is None or nxt == depot_idx:
                        if nxt == depot_idx and len(route) > 1:
                            route.append(depot_idx)
                        break
                    route.append(nxt)
                    curr = nxt
                routes.append(route)

                vehicle_packages = [
                    o for o in range(num_packages) if y[o, k].X > 0.5
                ]
                package_assignments.append(vehicle_packages)

            # Calculate travel time and stopover time for the solution
            total_travel_time = sum(
                self.route_durations[i][j] * x[i, j, k].X
                for k in range(len(self.vehicles))
                for i in range(num_nodes)
                for j in range(num_nodes)
                if i != j and (i, j, k) in x
            )
            total_stopover_time = 0
            for k in range(len(self.vehicles)):
                for o in range(num_packages):
                    stop_index = self.orders[o]["address_index"] + num_depots
                    if y[o, k].X > 0.5 and stop_index < len(self.stopping_time):
                        total_stopover_time += self.stopping_time[stop_index]

            return {
                "status": "optimal" if self.model.status == GRB.OPTIMAL else "feasible",
                "routes": routes,
                "package_assignments": package_assignments,
                "travel_time": total_travel_time,
                "stopover_time": total_stopover_time,
                "vehicle_usage": [
                    z[k].X for k in range(len(self.vehicles))
                ],
                "vehicle_depot_mapping": self.depot_vehicle_mapping,
                "priority_cost": sum(
                    self.order_waiting_times[o] * y[o, k].X
                    for k in range(len(self.vehicles))
                    for o in range(num_packages)
                    if (o, k) in y
                ),
                "total_cost": self.model.objVal,
                "solver": "Gurobi",
            }

        if self.model.status == GRB.INFEASIBLE:
            raise ValueError("No feasible solution found - problem is infeasible.")
        elif self.model.status == GRB.TIME_LIMIT:
            raise ValueError("Solver reached time limit without finding a feasible solution.")
        else:
            raise ValueError(f"Solver terminated with status: {self.model.status}")