import os
import warnings

import pandas as pd
import requests

VALID_ROUTING_PREFS = ["TRAFFIC_UNAWARE", "TRAFFIC_AWARE", "TRAFFIC_AWARE_OPTIMAL"]


class Route:
    def __init__(self, api_key=None):
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
        self.params["origin"] = {"address": origin_addr}

    def set_destination(self, dest_addr: str):
        self.params["destination"] = {"address": dest_addr}

    def set_departure_time(self, dept_time: pd.Timestamp, routing_pref: str):
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

    def set_routing_preference(self, routing_pref: str):
        if routing_pref not in VALID_ROUTING_PREFS:
            raise ValueError(
                f"Invalid routing preference. Choose from {VALID_ROUTING_PREFS}."
            )
        self.params["routingPreference"] = routing_pref

    def set_intermediates(self, intermediates: list):
        self.params["intermediates"] = []
        for addr in intermediates:
            self.params["intermediates"].append({"address": addr})

    def validate_params(self):
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

    def get_route(self, full_ver: bool = False):
        if full_ver:
            self.headers["X-Goog-FieldMask"] = "*"
        self.validate_params()
        response = requests.post(self.base_url, headers=self.headers, json=self.params)

        if response.status_code != 200:
            raise Exception(f"Error: {response.status_code} - {response.text}")

        data = response.json()
        if full_ver:
            return data

        if "routes" not in data or not data["routes"]:
            raise ValueError("No routes found.")

        route = data["routes"][0]
        distance = route["distanceMeters"]
        duration = route["duration"]

        return {"distance_meters": distance, "duration_seconds": duration}

