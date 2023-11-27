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

import utils


@hydra.main(config_path="conf/", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Load model for
       1. server-side parameter initialization
       2. server-side parameter evaluation
    """

    # Parse config * get experiment output directory
    save_path = HydraConfig.get().runtime.output_dir

    # TODO: Either instantiate a new model or load from a checkpoint
    model = hydra.utils.instantiate(cfg.model)

    model_parameters = utils.get_model_parameters(model)

    # TODO: Instantiate a strategy
    strategy = SaveModelStrategy(save_path=save_path)


    # Start Flower server for four rounds of federated learning
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        strategy=SaveModelStrategy,
        # TODO: Add security certificates here if needed
    )

    # Get global parameters with:
    # strategy.global_parameters
    print("All done")


if __name__ == "__main__":
    main()