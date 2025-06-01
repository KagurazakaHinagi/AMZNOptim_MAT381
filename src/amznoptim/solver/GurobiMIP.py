import gurobipy as gp
import pandas as pd
from gurobipy import GRB

from amznoptim.solver.base import DepotVRPBase
from amznoptim.utils.preprocess import (
    compute_delivery_windows,
    compute_waiting_times,
)


class DepotVRPGurobiBase(DepotVRPBase):
    """
    Base class for Depot Vehicle Routing Problem (VRP) solver using
    Gurobi MIP (Mixed Integer Programming) solver.

    This solver may require a Gurobi license to run.
    See https://www.gurobi.com/features/web-license-service/ for WLS licensing details.
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
        license_info["LICENSEID"] = int(license_info["LICENSEID"])
        env = gp.Env(params=license_info)
        self.model = gp.Model("DepotVRP", env=env)

    def set_solver_max_time(self, max_time: int):
        """
        Set the maximum time in seconds for the solver to run.
        If the solver exceeds this time, it will stop and return the
        best solution found so far.
        """
        self.model.setParam("TimeLimit", max_time)

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
        s = {}  # s[i][k] = visit order of stop i by vehicle k

        # Create decision variables
        for k, (_, weight_cap, volume_cap, cruising_dist) in enumerate(self.vehicles):
            z[k] = self.model.addVar(vtype=GRB.BINARY, name=f"z_{k}")
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if i != j:
                        x[i, j, k] = self.model.addVar(
                            vtype=GRB.BINARY, name=f"x_{i}_{j}_{k}"
                        )
                u[i, k] = self.model.addVar(
                    vtype=GRB.CONTINUOUS, lb=0, ub=self.max_duty_time, name=f"u_{i}_{k}"
                )
                s[i, k] = self.model.addVar(
                    vtype=GRB.INTEGER, lb=0, ub=num_nodes - 1, name=f"s_{i}_{k}"
                )
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
                gp.quicksum(x[depot_idx, j, k] for j in range(num_depots, num_nodes))
                == z[k],
                name=f"depot_outflow_{k}",
            )
            self.model.addConstr(
                gp.quicksum(x[i, depot_idx, k] for i in range(num_depots, num_nodes))
                == z[k],
                name=f"depot_inflow_{k}",
            )

            # 1b. Vehicle cannot travel between depots
            for other_depot in range(num_depots):
                if other_depot != depot_idx:
                    for j in range(num_nodes):
                        if j != other_depot:
                            self.model.addConstr(
                                x[other_depot, j, k] == 0,
                                name=f"no_depot_travel_{other_depot}_{j}_{k}",
                            )
                            self.model.addConstr(
                                x[j, other_depot, k] == 0,
                                name=f"no_depot_travel_{j}_{other_depot}_{k}",
                            )

        # Constraint 2: Package Assignment and Routing
        for k in range(len(self.vehicles)):
            for o in range(num_packages):
                stop_index = self.orders[o]["address_index"] + num_depots
                # 2a. If the package o is served by vehicle k,
                # then we need to ensure an incoming flow to the stop.
                self.model.addConstr(
                    gp.quicksum(
                        x[i, stop_index, k] for i in range(num_nodes) if i != stop_index
                    )
                    >= y[o, k],
                    name=f"package_routing_{o}_{k}",
                )

            for h in range(num_depots, num_nodes):
                # 2b. For each stop, the incoming flow must equal the outgoing flow
                self.model.addConstr(
                    gp.quicksum(x[i, h, k] for i in range(num_nodes) if i != h)
                    == gp.quicksum(x[h, j, k] for j in range(num_nodes) if j != h),
                    name=f"flow_conservation_{h}_{k}",
                )

        # Constraint 3: Vehicle Capacity
        for k, (_, weight_cap, volume_cap, cruising_dist) in enumerate(self.vehicles):
            # Weight capacity
            self.model.addConstr(
                gp.quicksum(self.weights[o] * y[o, k] for o in range(num_packages))
                <= weight_cap,
                name=f"weight_capacity_{k}",
            )
            # Volume capacity
            self.model.addConstr(
                gp.quicksum(self.volumes[o] * y[o, k] for o in range(num_packages))
                <= volume_cap,
                name=f"volume_capacity_{k}",
            )

        # Constraint 4: Vehicle Usage
        for k in range(len(self.vehicles)):
            for o in range(num_packages):
                self.model.addConstr(z[k] >= y[o, k], name=f"vehicle_usage_{k}_{o}")

        # Constraint 5: Miller-Tucker-Zemlin (MTZ) Subtour Elimination
        # See https://en.wikipedia.org/wiki/Travelling_salesman_problem
        for k in range(len(self.vehicles)):
            depot_idx = self.depot_vehicle_mapping[k]
            # 5a. Set depot visit order to 0
            self.model.addConstr(s[depot_idx, k] == 0, name=f"mtz_depot_{k}")

            # 5b. For each stop, the visit order must be greater than or equal to 1
            for i in range(num_depots, num_nodes):
                for j in range(num_depots, num_nodes):
                    if i != j:
                        self.model.addConstr(
                            s[i, k] - s[j, k] + num_nodes * x[i, j, k] <= num_nodes - 1,
                            name=f"mtz_{i}_{j}_{k}",
                        )

        # Constraint 6: Time Progression
        for k in range(len(self.vehicles)):
            depot_idx = self.depot_vehicle_mapping[k]

            # 6a. Depot departure time is 0
            self.model.addConstr(u[depot_idx, k] == 0, name=f"depot_departure_{k}")

            # 6b. Arrival time at j must be greater than or equal to
            # arrival time at i + travel time + stopover time
            for i in range(num_depots, num_nodes):
                for j in range(num_depots, num_nodes):
                    if i != j:
                        service_time_i = (
                            self.stopping_time[i] if i < len(self.stopping_time) else 0
                        )
                        travel_time_ij = self.route_durations[i][j]
                        self.model.addConstr(
                            u[j, k]
                            >= u[i, k]
                            + service_time_i
                            + travel_time_ij
                            - self.max_duty_time * (1 - x[i, j, k]),
                            name=f"time_progression_{i}_{j}_{k}",
                        )

        # Constraint 7: Vehicle Cruising Distance
        for k, (_, _, _, cruising_dist) in enumerate(self.vehicles):
            # The total distance traveled by vehicle k cannot exceed its cruising distance.
            self.model.addConstr(
                gp.quicksum(
                    int(self.route_distances[i][j]) * x[i, j, k]
                    for i in range(num_nodes)
                    for j in range(num_nodes)
                    if i != j
                )
                <= cruising_dist,
                name=f"distance_constraint_{k}",
            )

        # Constraint 8: Package Assignment
        # Each package is assigned to exactly one vehicle
        for o in range(num_packages):
            self.model.addConstr(
                gp.quicksum(y[o, k] for k in range(len(self.vehicles))) == 1,
                name=f"package_assignment_{o}",
            )

        # Same-day Service Specific Constraints
        if self.service != "Regular":
            # Additional constraints for same-day services including
            # Amazon Fresh and Prime Now delivery

            # Constraint 9: Delivery Window
            for k in range(len(self.vehicles)):
                for o in range(num_packages):
                    stop_index = self.orders[o]["address_index"] + num_depots
                    earliest_time = self.orders[o]["delivery_window_start_rel"]
                    latest_time = self.orders[o]["delivery_window_end_rel"]

                    # Ensure that the vehicle arrives within the delivery window
                    self.model.addConstr(
                        u[stop_index, k]
                        >= earliest_time - self.max_duty_time * (1 - y[o, k]),
                        name=f"delivery_window_start_{o}_{k}",
                    )
                    self.model.addConstr(
                        u[stop_index, k]
                        <= latest_time + self.max_duty_time * (1 - y[o, k]),
                        name=f"delivery_window_end_{o}_{k}",
                    )

        # OBJECTIVE FUNCTION: Minimize
        # total travel time + beta * truck usage - alpha * order waiting time
        time_cost = gp.quicksum(
            self.route_durations[i][j] * x[i, j, k]
            for k in range(len(self.vehicles))
            for i in range(num_nodes)
            for j in range(num_nodes)
            if i != j
        )

        if self.service == "Regular":
            priority_cost = gp.quicksum(
                self.order_waiting_times[o] * y[o, k]
                for k in range(len(self.vehicles))
                for o in range(num_packages)
            )
        elif self.service == "Amazon Fresh":
            priority_cost = gp.quicksum(
                self.perishable_costs[o] * y[o, k]
                for k in range(len(self.vehicles))
                for o in range(num_packages)
            )
        else:  # Prime Now
            priority_cost = 0

        vehicle_cost = gp.quicksum(z[k] for k in range(len(self.vehicles)))

        self.model.setObjective(
            time_cost + self.beta * vehicle_cost - self.alpha * priority_cost,
            GRB.MINIMIZE,
        )

        # Solve the model
        self.model.optimize()

        # Format the returned solution
        try:
            test_var = next(iter(z.values()))
            test_var.X

            routes = []
            package_assignments = []
            minutes_before_deadline = []

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

                vehicle_packages = [o for o in range(num_packages) if y[o, k].X > 0.5]
                package_assignments.append(vehicle_packages)

            # For Fresh and Prime Now services, calculate the minutes before deadline
            if self.service != "Regular":
                for o in range(num_packages):
                    serving_vehicle = None
                    for k in range(len(self.vehicles)):
                        if y[o, k].X > 0.5:
                            serving_vehicle = k
                            break
                    if serving_vehicle is not None:
                        stop_index = self.orders[o]["address_index"] + num_depots
                        arrival_time = u[stop_index, serving_vehicle].X
                        latest_time = self.orders[o]["delivery_window_end_rel"]
                        minutes_before = (latest_time - arrival_time) / 60.0
                        minutes_before_deadline.append(minutes_before)
                    else:
                        minutes_before_deadline.append(None)
            else:
                minutes_before_deadline = [None] * num_packages

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

            if self.model.status == GRB.OPTIMAL:
                status_str = "optimal"
            elif self.model.status == GRB.SUBOPTIMAL:
                status_str = "feasible"
            elif self.model.status == GRB.TIME_LIMIT:
                status_str = "time_limit_feasible"
            else:
                status_str = "other_feasible"

            return_dict = {
                "status": status_str,
                "service": self.service,
                "routes": routes,
                "package_assignments": package_assignments,
                "travel_time": total_travel_time,
                "stopover_time": total_stopover_time,
                "vehicle_usage": [z[k].X for k in range(len(self.vehicles))],
                "vehicle_depot_mapping": self.depot_vehicle_mapping,
                "total_cost": self.model.objVal,
                "solver": "Gurobi",
            }

            if self.service == "Regular":
                return_dict["priority_cost"] = sum(
                    self.order_waiting_times[o] * y[o, k].X
                    for k in range(len(self.vehicles))
                    for o in range(num_packages)
                    if (o, k) in y
                )
            elif self.service == "Amazon Fresh":
                return_dict["priority_cost"] = sum(
                    self.perishable_costs[o] * y[o, k].X
                    for k in range(len(self.vehicles))
                    for o in range(num_packages)
                    if (o, k) in y
                )
                return_dict["minutes_before_deadline"] = minutes_before_deadline

            else:  # Prime Now
                return_dict["priority_cost"] = 0
                return_dict["minutes_before_deadline"] = minutes_before_deadline

            return return_dict

        except (AttributeError, gp.GurobiError):
            if self.model.status == GRB.INFEASIBLE:
                raise ValueError("No feasible solution found - problem is infeasible.")
            elif self.model.status == GRB.TIME_LIMIT:
                raise ValueError(
                    "Solver reached time limit without finding a feasible solution."
                )
            else:
                raise ValueError(f"Solver terminated with status: {self.model.status}")


class DepotVRPGurobiRegular(DepotVRPGurobiBase):
    """
    Vehicle Routing Problem (VRP) solver for Amazon regular delivery services
    using Gurobi MIP solver for Amazon Regular delivery services.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the DepotVRPGurobiRegular with depot, order, and address data.
        """
        super().__init__(*args, **kwargs)
        self.service = "Regular"
        self.max_duty_time = (
            28800  # Maximum delivery time for each vehicle (default: 8 hours)
        )

    def process_order_data(self, dept_time: pd.Timestamp | None = None):
        """
        Process order data for Regular delivery services.
        Add waiting times to each order based on the departure time.
        """
        super().process_order_data(dept_time)
        order_data = compute_waiting_times(self.orders, dept_time=self.dept_time)
        self.order_waiting_times = [order["waiting_time"] for order in order_data]


class DepotVRPGurobiFresh(DepotVRPGurobiBase):
    """
    Vehicle Routing Problem (VRP) solver for Amazon Fresh delivery services
    using Gurobi MIP solver for Amazon Fresh delivery services.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the DepotVRPGurobiFresh with depot, order, and address data.
        """
        super().__init__(*args, **kwargs)
        self.service = "Amazon Fresh"
        self.max_duty_time = (
            14400  # Maximum delivery time for each vehicle (default: 4 hours)
        )

    def process_order_data(self, dept_time: pd.Timestamp | None = None):
        """
        Process order data for Amazon Fresh delivery services.
        Add delivery windows to each order based on the departure time.
        Add perishable marker to each order.
        """
        super().process_order_data(dept_time)
        self.orders = compute_delivery_windows(self.orders, dept_time=self.dept_time)
        self.perishable_costs = [order["perishable"] for order in self.orders]


class DepotVRPGurobiPrimeNow(DepotVRPGurobiBase):
    """
    Vehicle Routing Problem (VRP) solver for Amazon Prime Now delivery services
    using Gurobi MIP solver for Amazon Prime Now delivery services.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the DepotVRPGurobiPrimeNow with depot, order, and address data.
        """
        super().__init__(*args, **kwargs)
        self.service = "Prime Now"
        self.max_duty_time = (
            14400  # Maximum delivery time for each vehicle (default: 4 hours)
        )

    def process_order_data(self, dept_time: pd.Timestamp | None = None):
        """
        Process order data for Prime Now delivery services.
        Add delivery windows to each order based on the departure time.
        """
        super().process_order_data(dept_time)
        self.orders = compute_delivery_windows(self.orders, dept_time=self.dept_time)
