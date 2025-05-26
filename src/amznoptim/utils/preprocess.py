import json

import pandas as pd


def stop_info_from_orders(order_csv: str, packaging_info_tsv: str) -> dict:
    """Extracts stop information from orders CSV file."""
    df = pd.read_csv(order_csv, index_col=0)
    stop_data = {}
    for i, row in df.iterrows():
        package_volume, package_dimension = fetch_packaging_info(
            row["packaging_type"], packaging_info_tsv=packaging_info_tsv
        )
        if row["customer_id"] not in stop_data:
            stop_data[row["customer_id"]] = {
                "address": row["address"],
                "order": [
                    {
                        "id": i,
                        "timestamp": row["order_timestamp"],
                        "package_weight": row["packaging_weight"],
                        "package_volume": package_volume,
                        "package_dimension": package_dimension,
                    }
                ],
            }
        else:
            stop_data[row["customer_id"]]["order"].append(
                {
                    "id": i,
                    "timestamp": row["order_timestamp"],
                    "package_weight": row["packaging_weight"],
                    "package_volume": package_volume,
                    "package_dimension": package_dimension,
                }
            )
    for stop in stop_data.values():
        total_weight = sum(order["package_weight"] for order in stop["order"])
        total_volume = sum(order["package_volume"] for order in stop["order"])
        stop["total_weight"] = total_weight
        stop["total_volume"] = total_volume
    return stop_data


def compute_waiting_times(stop_data: dict, dept_time: pd.Timestamp) -> dict:
    """Computes maximum order waiting times for each stop."""
    for _, stop in stop_data.items():
        stop["max_waiting_time"] = 0
        for order in stop["order"]:
            order_time = pd.to_datetime(order["timestamp"]) - dept_time
            stop["max_waiting_time"] = max(
                stop["max_waiting_time"], order_time.total_seconds()
            )
    return stop_data


def fetch_packaging_info(packaging_type: str, packaging_info_tsv: str):
    """Fetches packaging information from the TSV file."""
    df = pd.read_csv(packaging_info_tsv, sep="\t")
    length = df.loc[df["Code"] == packaging_type, "Length(in)"].values[0]
    length = int(length * 254)  # Convert inches to mm
    width = df.loc[df["Code"] == packaging_type, "Width(in)"].values[0]
    width = int(width * 254)  # Convert inches to mm
    height = df.loc[df["Code"] == packaging_type, "Height(in)"].values[0]
    height = int(height * 254)  # Convert inches to mm
    volume = length * width * height
    return (volume, (length, width, height))


def fetch_route_matrix(
    depot_data: list[dict],
    stop_data: dict,
    traffic_aware: bool = False,
    dept_time: pd.Timestamp | None = None,
    matrix_json: str | None = None,
    save_path: str | None = None,
    api_key=None,
) -> tuple[dict, int]:
    """Fetches the route matrix for the stops using Google Maps API."""
    from amznoptim.utils.gmaps_service import RouteMatrix

    matrix_service = RouteMatrix(api_key=api_key)
    stop_addresses = [stop["address"] for stop in stop_data.values()]
    depot_addresses = [depot["address"] for depot in depot_data]
    addresses = depot_addresses + stop_addresses
    matrix_service.set_origins(addresses)
    matrix_service.set_destinations(addresses)
    if traffic_aware:
        if dept_time is None:
            raise ValueError(
                "Departure time must be provided for traffic-aware routing."
            )
        matrix_service.set_departure_time(
            dept_time=dept_time, routing_pref="TRAFFIC_AWARE_OPTIMAL"
        )
    if not matrix_json:
        service_result = matrix_service.get_route()
    else:
        with open(matrix_json, "r") as f:
            service_result = json.load(f)
    if save_path:
        with open(save_path, "w") as f:
            json.dump(service_result, f, indent=4)
    return matrix_service.process_route(service_result), len(depot_data)


def fetch_vehicle_info(
    depot_data: dict, vehicle_data_path: str
) -> list[tuple[float, float, float]]:
    """Fetches vehicle payload, capacity, and cruising range info."""
    with open(vehicle_data_path, "r") as f:
        all_vehicles = json.load(f)
    available_vehicles = depot_data.get("vehicles", {})
    if not available_vehicles:
        raise ValueError("No vehicles available in depot data.")
    vehicles = []
    for k, v in available_vehicles.items():
        if k in all_vehicles["Regular"]:
            weight_cap = all_vehicles["Regular"][k]["Weight_capacity"]
            volume_cap = all_vehicles["Regular"][k]["Volume_capacity"]
            cruising_dist = all_vehicles["Regular"][k]["Max_distance"]
            vehicles.extend([(weight_cap, volume_cap, cruising_dist)] * v)
    return vehicles
