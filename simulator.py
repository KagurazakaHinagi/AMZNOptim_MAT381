import json
import os
import uuid
from functools import partial

import numpy as np
import pandas as pd


class OrderGenerator:
    """
    Abstract class to generate random orders.
    """

    def __init__(self, n_address: int, n_package: int):
        self.n_address = n_address
        self.n_package = n_package
        self.rng = np.random.default_rng()
        self.address_sampler = RandomAddressSampler(n_address)
        self.package_generator = RandomPackageGenerator(n_package)

    def random_datetime(self, hours_from_now: int | float):
        """
        Generate a random datetime within the past range_from_today days.
        """
        now = pd.Timestamp.now()
        end_time = now - pd.Timedelta(hours=hours_from_now)
        timestamps = []
        timestamps_formatted = []
        for _ in range(self.n_package):
            timestamps.append(self.rng.integers(end_time.value, now.value))
            timestamps_formatted.append(pd.Timestamp(timestamps[-1]))
        return timestamps, timestamps_formatted

    def random_delivery_window(
        self, max_hours_from_now: int | float, min_hour_from_now: int | float
    ):
        """
        Generate a random delivery window within the future range
        """
        now = pd.Timestamp.now()
        end_time_min = now + pd.Timedelta(hours=min_hour_from_now)
        end_time_max = (
            now + pd.Timedelta(hours=max_hours_from_now) - pd.Timedelta(hours=2)
        )
        start_timestamps = []
        end_timestamps = []
        for _ in range(self.n_package):
            start_time = self.rng.integers(end_time_min.value, end_time_max.value)
            start_timestamp = pd.Timestamp(start_time)
            end_timestamp = start_timestamp + pd.Timedelta(
                hours=self.rng.integers(1, 2)
            )  # Random window of 1 or 2 hours
            start_timestamps.append(start_timestamp)
            end_timestamps.append(end_timestamp)
        return start_timestamps, end_timestamps

    def generate(self, *args, **kwargs):
        """
        Generate random orders.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def save(self, orders: pd.DataFrame, output_file: str | os.PathLike):
        """
        Save the sampled orders to a file.
        """
        orders.to_csv(output_file, index=False)
        print(f"Orders saved to {output_file}")


class RegularOrderGenerator(OrderGenerator):
    """
    Class to generate random regular Amazon orders.
    """

    def __init__(self, n_address: int, n_package: int):
        super().__init__(n_address, n_package)
        self.max_hours_from_now = 120

    def generate(
        self,
        dimension_tsv: str | os.PathLike,
        address_list_file: str | os.PathLike,
        max_hours_from_now: int | float | None = None,
        save_path: str | os.PathLike | None = None,
    ) -> pd.DataFrame:
        """
        Generate random orders.
        """
        packages = self.package_generator.generate_boxed(dimension_tsv=dimension_tsv)
        addresses = self.address_sampler.sample(address_list_file=address_list_file)
        if len(packages) > len(addresses):
            addresses = addresses.sample(
                n=len(packages), replace=True, random_state=self.rng
            ).reset_index(drop=True)
        elif len(packages) < len(addresses):
            raise ValueError("Number of packages is less than number of addresses.")

        max_hours_from_now = max_hours_from_now or self.max_hours_from_now
        orders = pd.DataFrame()
        orders["order_id"] = [uuid.uuid4() for _ in range(len(packages))]
        orders["order_timestamp"], _ = self.random_datetime(max_hours_from_now)
        orders = pd.concat([orders, addresses, packages], axis=1)
        orders.set_index("order_id", inplace=True)

        if save_path:
            self.save(orders, save_path)

        return orders


class FreshOrderGenerator(OrderGenerator):
    def __init__(self, n_address: int, n_package: int):
        super().__init__(n_address, n_package)
        self.max_hours_from_now = 48
        self.max_window_hours_from_now = 6
        self.min_window_hours_from_now = 1

    def generate(
        self,
        address_list_file: str | os.PathLike,
        max_hours_from_now: int | float | None = None,
        max_window_hours_from_now: int | float | None = None,
        min_window_hours_from_now: int | float | None = None,
        save_path: str | os.PathLike | None = None,
    ) -> pd.DataFrame:
        """
        Generate random fresh orders.
        """
        packages = self.package_generator.generate_unboxed()
        addresses = self.address_sampler.sample(address_list_file=address_list_file)
        if len(packages) > len(addresses):
            addresses = addresses.sample(
                n=len(packages), replace=True, random_state=self.rng
            ).reset_index(drop=True)
        elif len(packages) < len(addresses):
            raise ValueError("Number of packages is less than number of addresses.")

        max_hours_from_now = max_hours_from_now or self.max_hours_from_now
        max_window_hours_from_now = (
            max_window_hours_from_now or self.max_window_hours_from_now
        )
        min_window_hours_from_now = (
            min_window_hours_from_now or self.min_window_hours_from_now
        )

        orders = pd.DataFrame()
        orders["order_id"] = [uuid.uuid4() for _ in range(len(packages))]
        orders["order_timestamp"], _ = self.random_datetime(max_hours_from_now)
        orders["delivery_window_start"], orders["delivery_window_end"] = (
            self.random_delivery_window(
                max_window_hours_from_now, min_window_hours_from_now
            )
        )

        orders = pd.concat([orders, addresses, packages], axis=1)
        orders.set_index("order_id", inplace=True)

        if save_path:
            self.save(orders, save_path)

        return orders


class PrimeNowOrderGenerator(OrderGenerator):
    def __init__(self, n_address: int, n_package: int):
        super().__init__(n_address, n_package)
        self.max_hours_from_now = 48
        self.max_window_hours_from_now = 6
        self.min_window_hours_from_now = 1

    def generate(
        self,
        dimension_tsv: str | os.PathLike,
        address_list_file: str | os.PathLike,
        max_hours_from_now: int | float | None = None,
        max_window_hours_from_now: int | float | None = None,
        min_window_hours_from_now: int | float | None = None,
        save_path: str | os.PathLike | None = None,
    ) -> pd.DataFrame:
        """
        Generate random fresh orders.
        """
        packages = self.package_generator.generate_boxed(dimension_tsv=dimension_tsv)
        addresses = self.address_sampler.sample(address_list_file=address_list_file)
        if len(packages) > len(addresses):
            addresses = addresses.sample(
                n=len(packages), replace=True, random_state=self.rng
            ).reset_index(drop=True)
        elif len(packages) < len(addresses):
            raise ValueError("Number of packages is less than number of addresses.")

        max_hours_from_now = max_hours_from_now or self.max_hours_from_now
        max_window_hours_from_now = (
            max_window_hours_from_now or self.max_window_hours_from_now
        )
        min_window_hours_from_now = (
            min_window_hours_from_now or self.min_window_hours_from_now
        )

        orders = pd.DataFrame()
        orders["order_id"] = [uuid.uuid4() for _ in range(len(packages))]
        orders["order_timestamp"], _ = self.random_datetime(max_hours_from_now)
        orders["delivery_window_start"], orders["delivery_window_end"] = (
            self.random_delivery_window(
                max_window_hours_from_now, min_window_hours_from_now
            )
        )

        orders = pd.concat([orders, addresses, packages], axis=1)
        orders.set_index("order_id", inplace=True)

        if save_path:
            self.save(orders, save_path)

        return orders


class RandomAddressSampler:
    """
    Class to sample addresses from a list of Washington addresses in a file.
    """

    def __init__(self, n: int):
        self.n = n

    def sample(self, address_list_file: str | os.PathLike) -> pd.DataFrame:
        """
        Sample n addresses from a list of addresses in a file.
        """
        reservoir = []
        with open(address_list_file, "r") as infile:
            for i, line in enumerate(infile):
                if i < self.n:
                    reservoir.append(line)
                else:
                    j = np.random.randint(0, i + 1)
                    if j < self.n:
                        reservoir[j] = line
        addresses = self.format_addresses(reservoir)

        return addresses

    def format_addresses(self, addresses: list) -> pd.DataFrame:
        """
        Format addresses into a DataFrame.
        """
        formatted_addresses = pd.DataFrame(columns=["customer_id", "address"])
        for line in addresses:
            address = json.loads(line)
            address_id = address["properties"]["hash"]
            formatted_address = (
                address["properties"]["number"]
                + " "
                + address["properties"]["street"]
                + ", "
                + address["properties"]["city"]
                + ", "
                + "WA"
                + ", "
                + address["properties"]["postcode"]
            )
            formatted_addresses = pd.concat(
                [
                    formatted_addresses,
                    pd.DataFrame(
                        {"customer_id": [address_id], "address": [formatted_address]}
                    ),
                ],
                ignore_index=True,
            )
        return formatted_addresses


class RandomPackageGenerator:
    """
    Class to generate random package information.
    """

    def __init__(self, n: int):
        self.n = n
        self.rng = np.random.default_rng()
        self.sampling_dist_boxed = partial(self.rng.lognormal, mean=np.log(2), sigma=1)
        self.sampling_dist_unboxed_weight = partial(
            self.rng.lognormal, mean=3.44, sigma=0.65
        )
        self.sampling_dist_unboxed_volume = partial(
            self.rng.lognormal, mean=1.14, sigma=0.65
        )

    def generate_boxed(
        self, dimension_tsv: str | os.PathLike, sampling_dist: partial | None = None
    ) -> pd.DataFrame:
        """
        Generate random package information, simulate given distribution.
        """
        sampling_dist = sampling_dist or self.sampling_dist_boxed
        weights = sampling_dist(size=self.n) * 454  # Convert lbs to g
        avail_package_types = pd.read_csv(dimension_tsv, sep="\t", usecols=[0])
        packages = avail_package_types.sample(self.n, replace=True).reset_index(
            drop=True
        )
        packages["packaging_weight"] = [int(weight) for weight in weights]
        packages.columns.values[0] = "packaging_type"

        return packages

    def generate_unboxed(
        self,
        weight_sampling_dist: partial | None = None,
        volume_sampling_dist: partial | None = None,
    ) -> pd.DataFrame:
        """
        Generate random package information for grocery orders, simulate given distribution.
        """
        weight_sampling_dist = weight_sampling_dist or self.sampling_dist_unboxed_weight
        volume_sampling_dist = volume_sampling_dist or self.sampling_dist_unboxed_volume
        weights = weight_sampling_dist(size=self.n) * 454  # Convert lbs to g
        volumes = (
            volume_sampling_dist(size=self.n) * 2.832e7
        )  # Convert cubic inches to cubic mm
        perishable = self.rng.integers(0, 2, size=self.n)
        packages = pd.DataFrame(
            {
                "packaging_weight": [int(weight) for weight in weights],
                "packaging_volume": [int(volume) for volume in volumes],
                "perishable": [int(p) for p in perishable],
            }
        )

        return packages
