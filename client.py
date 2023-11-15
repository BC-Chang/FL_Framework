import numpy as np
import matplotlib.pyplot as plt
import flwr as fl
import torch
import argparse


def main() -> None:
    # Parse command line argument `partition`
    parser = argparse.ArgumentParser(description="Flower Client")
    parser.add_argument(
        "--input_data",
        type=int,
        default=0,
        choices=range(0, 10),
        required=False,
        help="Specifies the artificial data partition of CIFAR10 to be used. \
        Picks partition 0 by default",
    )
    parser.add_argument(
        "--use_cuda",
        type=bool,
        default=False,
        required=False,
        help="Set to true to use GPU. Default: False",
    )

    args = parser.parse_args()

    # TODO: Load data from a specific datafile
    # Load local data partition
    trainset, testset = load_data.load_partition(args.partition)

    # TODO: Instantiate Flower client
    client = MSNet_Client(trainset, testset, device)

    # Start Flower client
    # TODO: Check server address
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)

if __name__ == "__main__":
    main()

