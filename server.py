from typing import Dict, Optional, Tuple
from collections import OrderedDict
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig

# FL & ML parameters
import flwr as fl
import torch
from agg_strats import SaveModelStrategy

import utils


@hydra.main(config_path="docs/config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """Load model for
       1. server-side parameter initialization
       2. server-side parameter evaluation
       """

    # TODO: Either instantiate a new model or load from a checkpoint


    model_parameters = [val.cpu().numpy() for _, val in model.state_dict().items()]

    # TODO: Import strategy

    # Start Flower server for four rounds of federated learning
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        strategy=SaveModelStrategy,
    )