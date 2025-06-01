import json
import os

import pandas as pd


def compute_waiting_times(
    order_data: list[dict], dept_time: pd.Timestamp
) -> list[dict]:
    """Computes the waiting time for each order in Regular delivery services."""
    for order in order_data:
        order["waiting_time"] = int(
            (dept_time - pd.to_datetime(order["timestamp"])).total_seconds()
        )
    return order_data


def compute_delivery_windows(
    order_data: list[dict], dept_time: pd.Timestamp
) -> list[dict]:
    """Computes the delivery windows for each order in Fresh delivery services."""
    for order in order_data:
        order["delivery_window_start_rel"] = int(
            (pd.to_datetime(order["delivery_window_start"]) - dept_time).total_seconds()
        )
        order["delivery_window_end_rel"] = int(
            (pd.to_datetime(order["delivery_window_end"]) - dept_time).total_seconds()
        )
    return order_data


def regular_order_info_from_csv(
    order_csv: str | os.PathLike, packaging_info_tsv: str | os.PathLike
) -> tuple[list[dict], dict[str, list]]:
    """Extracts Regular order information from orders CSV file."""
    df = pd.read_csv(order_csv, index_col=0)
    order_data = []
    address_data = {
        "ids": [],
        "addresses": [],
        "max_num_packages": [],
    }
    for i, row in df.iterrows():
        package_volume, package_dimension = fetch_packaging_info(
            row["packaging_type"], packaging_info_tsv=packaging_info_tsv
        )
        try:
            address_index = address_data["ids"].index(row["customer_id"])
            address_data["max_num_packages"][address_index] += 1
        except ValueError:
            address_data["ids"].append(row["customer_id"])
            address_index = len(address_data["ids"]) - 1
            address_data["addresses"].append(row["address"])
            address_data["max_num_packages"].append(1)
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


def sameday_order_info_from_csv(
    order_csv: str | os.PathLike, packaging_info_tsv: str | os.PathLike | None = None
) -> tuple[list[dict], dict[str, list]]:
    """Extracts Fresh / Prime Now order information from orders CSV file."""
    df = pd.read_csv(order_csv, index_col=0)
    order_data = []
    address_data = {
        "ids": [],
        "addresses": [],
        "max_num_packages": [],
    }
    for i, row in df.iterrows():
        if "packaging_type" in row:  # For Prime Now delivery services
            if packaging_info_tsv:
                package_volume, package_dimension = fetch_packaging_info(
                    row["packaging_type"], packaging_info_tsv=packaging_info_tsv
                )
                row["packaging_volume"] = package_volume
                row["packaging_dimension"] = package_dimension
            else:
                raise ValueError(
                    "Packaging type is provided but packaging info TSV is not specified."
                )
        try:
            address_index = address_data["ids"].index(row["customer_id"])
            address_data["max_num_packages"][address_index] += 1
        except ValueError:
            address_data["ids"].append(row["customer_id"])
            address_index = len(address_data["ids"]) - 1
            address_data["addresses"].append(row["address"])
            address_data["max_num_packages"].append(1)
        order_data.append(
            {
                "id": i,
                "address_id": row["customer_id"],
                "address": row["address"],
                "timestamp": row["order_timestamp"],
                "delivery_window_start": row["delivery_window_start"],
                "delivery_window_end": row["delivery_window_end"],
                "package_weight": row["packaging_weight"],
                "package_volume": row["packaging_volume"],
                "address_index": address_index,
            }
        )
        if "perishable" in row:  # For Fresh delivery services
            order_data[-1]["perishable"] = row["perishable"]

    return order_data, address_data


def fetch_packaging_info(packaging_type: str | os.PathLike, packaging_info_tsv: str | os.PathLike) -> tuple[int, tuple[int, int, int]]:
    """Fetches packaging information from the TSV file."""
    df = pd.read_csv(packaging_info_tsv, sep="\t")
    length = df.loc[df["Code"] == packaging_type, "Length(in)"].values[0]
    length = int(length * 25.4)  # Convert inches to mm
    width = df.loc[df["Code"] == packaging_type, "Width(in)"].values[0]
    width = int(width * 25.4)  # Convert inches to mm
    height = df.loc[df["Code"] == packaging_type, "Height(in)"].values[0]
    height = int(height * 25.4)  # Convert inches to mm
    volume = length * width * height
    return (volume, (length, width, height))


def fetch_route_matrix(
    depot_data: list[dict],
    stop_addresses: list[str],
    traffic_aware: bool = False,
    dept_time: pd.Timestamp | None = None,
    matrix_json: str | os.PathLike | None = None,
    save_path: str | os.PathLike | None = None,
    api_key: str | None = None,
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
    service: str, depot_data: dict, vehicle_data_path: str
) -> list[tuple[str, float, float, float]]:
    """Fetches vehicle model, payload, capacity, and cruising range info."""
    with open(vehicle_data_path, "r") as f:
        all_vehicles = json.load(f)
    available_vehicles = depot_data.get("vehicles", {})
    if not available_vehicles:
        raise ValueError("No vehicles available in depot data.")
    vehicles = []
    for k, v in available_vehicles.items():
        if k in all_vehicles[service]:
            weight_cap = all_vehicles[service][k]["Weight_capacity"]
            volume_cap = all_vehicles[service][k]["Volume_capacity"]
            cruising_dist = all_vehicles[service][k]["Max_distance"]
            vehicles.extend([(k, weight_cap, volume_cap, cruising_dist)] * v)
    return vehicles


def calculate_stopover_times(
    stop_info: dict,
    validation_json: str | os.PathLike | None = None,
    save_path: str | os.PathLike | None = None,
    api_key: str | None = None,
):
    """
    Calculates stopover times for each stop using Google Maps Address Validation API.
    """
    BASE_TIMES = {
        "RESIDENTIAL": {
            "C": 240,  # Residential + City: 4 minutes
            "R": 360,  # Residential + Rural: 6 minutes
            "H": 480,  # Residential + High-rise: 8 minutes
            "U": 480,  # Residential + Unknown: 8 minutes
        },
        "BUSINESS": {
            "C": 180,  # Business + City: 3 minutes
            "R": 300,  # Business + Rural: 5 minutes
            "H": 600,  # Business + High-rise: 10 minutes
            "U": 600,  # Business + Unknown: 10 minutes
        },
        "UNKNOWN": {
            "C": 300,  # Unknown + City: 5 minutes
            "R": 420,  # Unknown + Rural: 7 minutes
            "H": 600,  # Unknown + High-rise: 10 minutes
            "U": 600,  # Unknown + Unknown: 10 minutes
        },
    }

    PER_PACKAGE_TIME = {
        "UNKNOWN": 70,  # 70 seconds for unknown
        "RESIDENTIAL": 60,  # 60 seconds for residential
        "BUSINESS": 50,  # 50 seconds for business
    }

    from amznoptim.utils.gmaps_service import AddressValidation

    validation_service = AddressValidation(api_key=api_key)
    validation_responses = []
    processed_info = []
    if validation_json:
        with open(validation_json, "r") as f:
            validation_responses = json.load(f)

    for idx, address in enumerate(stop_info["addresses"]):
        validation_service.set_address(address)
        if not validation_json:
            response = validation_service.get_address_validation(UspsCass=True)
            validation_responses.append(response)
        else:
            response = validation_responses[idx]
        processed_info.append(validation_service.process_address_validation(response))

    if save_path:
        with open(save_path, "w") as f:
            json.dump(validation_responses, f, indent=4)

    stopover_times = []
    for idx, info in enumerate(processed_info):
        address_type = info["address_type"]
        if address_type not in BASE_TIMES:
            raise ValueError(f"Unknown address type: {address_type}")
        carrier_route = info["usps_carrier_route"]
        if carrier_route is None or carrier_route[0] not in BASE_TIMES[address_type]:
            carrier_route = ["U"]  # Default to 'Unknown' if not available
        base_time = BASE_TIMES[address_type][carrier_route[0]]
        package_time = (
            PER_PACKAGE_TIME[address_type] * stop_info["max_num_packages"][idx]
        )
        stopover_times.append(base_time + package_time)

    print(stopover_times)
    return stopover_times
