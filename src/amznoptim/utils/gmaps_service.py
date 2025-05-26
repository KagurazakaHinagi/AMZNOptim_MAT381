import os
import warnings

import numpy as np
import pandas as pd
import requests

VALID_ROUTING_PREFS = ["TRAFFIC_UNAWARE", "TRAFFIC_AWARE", "TRAFFIC_AWARE_OPTIMAL"]


def save_to_csv(data: dict, file_path):
    """
    Helpers function to save data to a CSV file.
    """
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)


class Route:
    def __init__(self, api_key=None):
        """
        Fetches single route information from Google Maps Directions API.
        https://developers.google.com/maps/documentation/routes/reference/rest/v2/TopLevel/computeRoutes
        """
        self.base_url = "https://routes.googleapis.com/directions/v2:computeRoutes"
        self.api_key = api_key or os.getenv("GMAPS_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key is required. Set it as an environment variable or pass it directly."
            )
        self.headers = {
            "Content-Type": "application/json",
            "X-Goog-FieldMask": "routes.distanceMeters,routes.duration",
            "X-Goog-Api-Key": self.api_key,
        }
        self.params = {
            "origin": {},
            "destination": {},
            "departureTime": None,
            "routingPreference": "TRAFFIC_UNAWARE",
            "intermediates": [],
            "travelMode": "DRIVE",
            "languageCode": "en-US",
            "regionCode": "US",
            "units": "METRIC",
        }

    def set_origin(self, origin_addr: str):
        """
        Sets the origin address for the route.
        """
        self.params["origin"] = {"address": origin_addr}

    def set_destination(self, dest_addr: str):
        """
        Sets the destination address for the route.
        """
        self.params["destination"] = {"address": dest_addr}

    def set_departure_time(self, dept_time: pd.Timestamp, routing_pref: str):
        """
        Sets the departure time and routing preference for the route.
        If the routing preference is not "TRAFFIC_UNAWARE", the departure time
        must be in RFC 3339 format.
        NOTE: The time zone is not considered in this implementation (UTC is assumed).
        NOTE: "TRAFFIC_AWARE" and "TRAFFIC_AWARE_OPTIMAL" are billed at a higher rate.
        See https://developers.google.com/maps/documentation/routes/usage-and-billing
        """
        if routing_pref not in VALID_ROUTING_PREFS:
            raise ValueError(
                f"Invalid routing preference. Choose from {VALID_ROUTING_PREFS}."
            )
        self.params["routingPreference"] = routing_pref
        if self.params["routingPreference"] != "TRAFFIC_UNAWARE":
            # RFC 3339 format with 0, 3, 6, or 9 digits of fractional seconds
            # https://datatracker.ietf.org/doc/html/rfc3339#section-5.6
            # Example: 2023-10-01T12:00:00Z or 2023-10-01T12:00:00.123456789Z
            self.params["departureTime"] = dept_time.isoformat()[:-6] + "Z"

    def set_intermediates(self, intermediates: list):
        """
        Sets the intermediate addresses for the route.
        Each address should be a string.
        Example: ["123 Main St, City, State", "456 Elm St, City, State"]
        """
        self.params["intermediates"] = []
        for addr in intermediates:
            self.params["intermediates"].append({"address": addr})

    def validate_params(self):
        """
        Validates the parameters before making the API request.
        Raises ValueError if any required parameter is missing.
        """
        if not self.headers["X-Goog-Api-Key"]:
            raise ValueError("API key is required.")
        if not self.params["origin"]:
            raise ValueError("Origin address is required.")
        if not self.params["destination"]:
            raise ValueError("Destination address is required.")
        if self.params["intermediates"]:
            warnings.warn(
                "Intermediates have been set. Ensure you have the correct API key and billing enabled."
            )
        if self.params["routingPreference"] != "TRAFFIC_UNAWARE":
            warnings.warn(
                "Routing preference is set to traffic-aware. Ensure you have the correct API key and billing enabled."
            )

    def dry_run(self):
        """
        Performs a dry run to print the parameters and headers without making the API call.
        """
        print("Headers:", self.headers)
        print("Parameters:", self.params)
        print("URL:", self.base_url)

    def get_route(self, full_ver: bool = False):
        """
        Fetches the route information from the Google Maps Directions API.
        If full_ver is True, fetches all available fields.
        Otherwise, fetches only the distance and duration.
        """
        if full_ver:
            self.headers["X-Goog-FieldMask"] = "*"
        self.validate_params()
        response = requests.post(self.base_url, headers=self.headers, json=self.params)

        if response.status_code != 200:
            raise Exception(f"Error: {response.status_code} - {response.text}")

        data = response.json()

        return data

    def process_route(self, data):
        """
        Processes the route data and extracts distance and duration.
        Returns a dictionary with distance in meters and duration in seconds.
        Raises ValueError if no routes are found.
        """
        if "routes" not in data or not data["routes"]:
            raise ValueError("No routes found.")

        route = data["routes"][0]
        distance = route["distanceMeters"]
        duration = route["duration"]

        return {"distance_meters": distance, "duration_seconds": duration}


class RouteMatrix(Route):
    def __init__(self, api_key=None):
        """
        Fetches route matrix information from Google Maps Directions API.
        https://developers.google.com/maps/documentation/routes/reference/rest/v2/TopLevel/computeRouteMatrix
        """
        super().__init__(api_key)
        self.base_url = "https://routes.googleapis.com/directions/v2:computeRouteMatrix"
        self.params = {
            "origins": [],
            "destinations": [],
            "departureTime": None,
            "routingPreference": "TRAFFIC_UNAWARE",
            "travelMode": "DRIVE",
            "languageCode": "en-US",
            "regionCode": "US",
            "units": "METRIC",
        }
        self.headers["X-Goog-FieldMask"] = (
            "originIndex,destinationIndex,duration,distanceMeters,status,condition"
        )

    def set_origins(self, origins: list):
        """
        Sets the origin addresses for the route matrix.
        Each address should be a string.
        Example: ["123 Main St, City, State", "456 Elm St, City, State"]
        """
        self.params["origins"] = [{"waypoint": {"address": addr}} for addr in origins]

    def set_destinations(self, destinations: list):
        """
        Sets the destination addresses for the route matrix.
        Each address should be a string.
        Example: ["789 Oak St, City, State", "101 Pine St, City, State"]
        """
        self.params["destinations"] = [
            {"waypoint": {"address": addr}} for addr in destinations
        ]

    def validate_params(self):
        super().validate_params()
        origins = self.params.get("origins", []) or []
        destinations = self.params.get("destinations", []) or []
        if len(origins) + len(destinations) > 650:
            raise IndexError(
                "The total number of origins and destinations must not exceed 650."
            )

    def process_route(self, data):
        """
        Processes the route matrix data and extracts distances and durations.
        Returns a dictionary with distance in meters and duration in seconds.
        The matrices are in the form of numpy arrays of size
        (number of origins, number of destinations).
        The distance and duration matrices are filled with np.inf
        for routes that do not exist or with errors.
        Raises ValueError if no routes are found.
        """
        origins = self.params.get("origins", []) or []
        destinations = self.params.get("destinations", []) or []
        time_matrix = np.full((len(origins), len(destinations)), np.inf)
        dist_matrix = np.full((len(origins), len(destinations)), np.inf)
        for entry in data:
            origin_index = entry["originIndex"]
            destination_index = entry["destinationIndex"]
            if entry["condition"] == "ROUTE_EXISTS":
                time_matrix[origin_index][destination_index] = entry["duration"]
                dist_matrix[origin_index][destination_index] = entry["distanceMeters"]
            else:
                time_matrix[origin_index][destination_index] = np.inf
                dist_matrix[origin_index][destination_index] = np.inf

        return {"distance_meters": dist_matrix, "duration_seconds": time_matrix}


class AddressValidation:
    def __init__(self, api_key=None):
        """
        Fetches address validation information from Google Maps Address Validation API.
        https://developers.google.com/maps/documentation/address-validation/reference/rest/v1/TopLevel/validateAddress
        """
        self.base_url = "https://addressvalidation.googleapis.com/v1:validateAddress"
        self.api_key = api_key or os.getenv("GMAPS_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key is required. Set it as an environment variable or pass it directly."
            )
        self.headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": self.api_key,
        }
        self.params = {
            "address": {},
            "enableUspsCass": False,
        }

    def set_address(self, address: str):
        """
        Sets the address for validation.
        """
        self.params["address"] = {"addressLines": [address]}

    def validate_params(self):
        """
        Validates the parameters before making the API request.
        Raises ValueError if any required parameter is missing.
        """
        if not self.headers["X-Goog-Api-Key"]:
            raise ValueError("API key is required.")
        if not self.params["address"]:
            raise ValueError("Address is required.")

    def dry_run(self):
        """
        Performs a dry run to print the parameters and headers without making the API call.
        """
        print("Headers:", self.headers)
        print("Parameters:", self.params)
        print("URL:", self.base_url)

    def get_address_validation(self, UspsCass: bool = False):
        """
        Fetches the address validation information from the Google Maps Address Validation API.
        """
        if UspsCass:
            self.params["enableUspsCass"] = True
        else:
            self.params["enableUspsCass"] = False
        self.validate_params()
        response = requests.post(self.base_url, headers=self.headers, json=self.params)

        if response.status_code != 200:
            raise Exception(f"Error: {response.status_code} - {response.text}")

        data = response.json()

        return data

    def process_address_validation(self, data):
        pass  # TODO: Implement this method to process the address validation data
        # for stopover windows estimation.
