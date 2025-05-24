import json
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
        self.max_hours_from_now = 24

    def random_datetime(self, hours_from_now):
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

    def generate(self):
        """
        Generate random orders.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def _generate(self, max_hours_from_now=None, save_path=None, **kwargs):
        """
        Generate random orders. (Helper function)
        """
        packages = self.package_generator.generate(
            save_path=save_path, dimension_tsv=kwargs.get("dimension_tsv")
        )
        addresses = self.address_sampler.sample(
            save_path=save_path, address_list_file=kwargs.get("address_list_file")
        )
        if len(packages) > len(addresses):
            addresses = addresses.sample(
                n=len(packages), replace=True, random_state=self.rng
            ).reset_index(drop=True)
        elif len(packages) < len(addresses):
            raise ValueError(
                "Number of packages is less than number of addresses."
            )

        max_hours_from_now = max_hours_from_now or self.max_hours_from_now
        orders = pd.DataFrame()
        orders["order_id"] = [uuid.uuid4() for _ in range(len(packages))]
        orders["order_timestamp"], _ = self.random_datetime(
            max_hours_from_now
        )
        orders = pd.concat([orders, addresses, packages], axis=1)
        orders.set_index("order_id", inplace=True)
        return orders

    def _save(self, orders, output_file):
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

    def generate(self, max_hours_from_now=None, save_path=None, **kwargs):
        """
        Generate random orders.
        """
        orders = self._generate(max_hours_from_now, save_path, **kwargs)
        if save_path:
            self._save(orders, save_path)
        return orders


class SameDayOrderGenerator(OrderGenerator):
    def __init__(self, n_address: int, n_package: int):
        super().__init__(n_address, n_package)

    # TODO: Implement same day order generation


class FreshOrderGenerator(OrderGenerator):
    def __init__(self, n_address: int, n_package: int):
        super().__init__(n_address, n_package)

    # TODO: Implement fresh order generation


class RandomAddressSampler:
    """
    Class to sample addresses from a list of addresses in a file.
    """

    def __init__(self, n):
        self.n = n

    def sample(self, address_list_file, save_path=None) -> pd.DataFrame:
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
        if save_path:
            self._save(addresses, save_path)
        return addresses

    def format_addresses(self, addresses):
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

    def _save(self, addresses, output_file):
        """
        Save the sampled addresses to a file.
        """
        addresses.to_csv(output_file, index=False)


class RandomPackageGenerator:
    """
    Class to generate random package information.
    """

    def __init__(self, n: int):
        self.n = n
        self.rng = np.random.default_rng()
        self.sampling_dist = partial(self.rng.lognormal, mean=np.log(2), sigma=1)

    def generate(
        self, dimension_tsv, sampling_dist=None, save_path=None
    ) -> pd.DataFrame:
        """
        Generate random package information, simulate given distribution.
        """
        sampling_dist = sampling_dist or self.sampling_dist
        weights = sampling_dist(size=self.n)
        avail_package_types = pd.read_csv(dimension_tsv, sep="\t", usecols=[0])
        packages = avail_package_types.sample(self.n, replace=True).reset_index(drop=True)
        packages["packaging_weight"] = weights.round(2)
        packages.columns.values[0] = "packaging_type"
        if save_path:
            self._save(packages, save_path)
        return packages

    def _save(self, packages, output_file):
        """
        Save the sampled addresses to a file.
        """
        packages.to_csv(output_file, index=False)
