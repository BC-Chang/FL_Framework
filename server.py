from typing import Dict, Optional, Tuple
from collections import OrderedDict
from torch.utils.data import DataLoader
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

# FL & ML parameters
import flwr as fl
import torch
from agg_strats import SaveModelStrategy
from flwr.server.strategy import FedAvg
from network import MS_Net
import utils
from tasks import weighted_average, get_evaluate_fn, get_on_fit_config
import load_data
import pandas as pd

# Set GRPC Poll strategy. epollex is standard implementation, but it causes a bug. Use epoll1 instead
import os
os.environ["GRPC_POLL_STRATEGY"] = "epoll1"

@hydra.main(config_path="conf/", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Load model for
       1. server-side parameter initialization
       2. server-side parameter evaluation
    """

    # Parse config * get experiment output directory
    save_path = HydraConfig.get().runtime.output_dir

    # TODO: Allow option to load from checkpoint
    model = hydra.utils.instantiate(cfg.model).to(cfg.device)

    # Get model parameters
    model_parameters = utils.get_model_parameters(model)

    # Centralized test data
    testloader = load_data.load_data(cfg.test_input_file, path_to_data=cfg.data_loc, phases=['test'])[0]

    # Instantiate a strategy
    strategy = hydra.utils.instantiate(cfg.strategy, evaluate_metrics_aggregation_fn=weighted_average,
                                       evaluate_fn=get_evaluate_fn(cfg.model, testloader, cfg.device))

    # Start Flower server for four rounds of federated learning
    history = fl.server.start_server(
        server_address="127.0.0.1:8080",
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds,
                                      round_timeout=cfg.round_timeout),
        strategy=strategy,
        # TODO: Add security certificates here if needed
        # https://github.com/adap/flower/blob/821d843278e60c55acdfb3574de8958c26f7a644/src/py/flwr/server/app.py#L117
    )

    # Get global parameters with:
    print(history)
    print("All done :)")
    df = pd.DataFrame(history.losses_centralized, columns=["Round", "Loss_Centralized"])
    df.to_csv(f"./{save_path}/losses_centralized.csv", index=False)

if __name__ == "__main__":
    main()
