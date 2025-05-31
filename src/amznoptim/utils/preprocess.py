import json

import pandas as pd


def compute_waiting_times(
    order_data: list[dict], dept_time: pd.Timestamp
) -> list[dict]:
    """Computes maximum order waiting times for each stop."""
    for order in order_data:
        order["waiting_time"] = int((dept_time - pd.to_datetime(order["timestamp"])).total_seconds())
    return order_data


def order_info_from_csv(
    order_csv: str, packaging_info_tsv: str
) -> tuple[list[dict], dict[str, list[str]]]:
    """Extracts order information from orders CSV file."""
    df = pd.read_csv(order_csv, index_col=0)
    order_data = []
    address_data = {
        "ids": [],
        "addresses": [],
    }
    for i, row in df.iterrows():
        package_volume, package_dimension = fetch_packaging_info(
            row["packaging_type"], packaging_info_tsv=packaging_info_tsv
        )
        try:
            address_index = address_data["ids"].index(row["customer_id"])
        except ValueError:
            address_data["ids"].append(row["customer_id"])
            address_index = len(address_data["ids"]) - 1
            address_data["addresses"].append(row["address"])
        order_data.append(
            {
                "id": i,
                "address_id": row["customer_id"],
                "address": row["address"],
                "timestamp": row["order_timestamp"],
                "package_weight": row["packaging_weight"],
                "package_volume": package_volume,
                "package_dimension": package_dimension,
                "address_index": address_index,
            }
        )
    return order_data, address_data


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
    stop_addresses: list[str],
    traffic_aware: bool = False,
    dept_time: pd.Timestamp | None = None,
    matrix_json: str | None = None,
    save_path: str | None = None,
    api_key=None,
) -> tuple[dict, int]:
    """Fetches the route matrix for the stops using Google Maps API."""
    from amznoptim.utils.gmaps_service import RouteMatrix

    matrix_service = RouteMatrix(api_key=api_key)
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
) -> list[tuple[str, float, float, float]]:
    """Fetches vehicle model, payload, capacity, and cruising range info."""
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
            vehicles.extend([(k, weight_cap, volume_cap, cruising_dist)] * v)
    return vehicles
