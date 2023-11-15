import numpy as np
import matplotlib.pyplot as plt
import flwr as fl
import torch
import argparse

def MSNet_Client():



def main() -> None:
    # Parse command line argument `partition`
    parser = argparse.ArgumentParser(description="Flower Client")
    parser.add_argument(
        "--input_data",
        type=str,
        required=True,
        help="Specify the location of the data input yaml with listed data files.",
    )

    args = parser.parse_args()

    # TODO: Load data from a specific datafile
    # Load local data partition
    trainloader, valloader = load_data.load_partition(args.partition)

    # TODO: Instantiate Flower client
    client = MSNet_Client(trainloader, valloader, device)

    # Start Flower client
    # TODO: Check server address
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)

if __name__ == "__main__":
    main()


