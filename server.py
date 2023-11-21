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



@hydra.main(config_path="conf/", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Load model for
       1. server-side parameter initialization
       2. server-side parameter evaluation
    """

    # Parse config * get experiment output directory
    save_path = HydraConfig.get().runtime.output_dir

    # TODO: Either instantiate a new model or load from a checkpoint
    # model = hydra.utils.instantiate(cfg.model)
    model = MS_Net(
        num_scales=4,
        num_features=1,
        num_filters=2,
        device='cuda',
        f_mult=2,
        summary=False
    ).to("cuda")
    #
    model_parameters = utils.get_model_parameters(model)

    # TODO: Instantiate a strategy
    model_dict = {'num_scales': 4,
                  'num_features': 1,
                  'num_filters': 2,
                   'device': 'cuda',
                   'f_mult': 2}
    # strategy = SaveModelStrategy(save_path=save_path, model_dict=model_dict)
    strategy = FedAvg()

    # Start Flower server for four rounds of federated learning
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        strategy=strategy,
        # TODO: Add security certificates here if needed
    )

    # Get global parameters with:
    # strategy.global_parameters
    print("All done")


if __name__ == "__main__":
    main()
