# IO Packages
import hydra
from omegaconf import DictConfig

# FL Libraries
# import flwr as fl
import torch
# from agg_strats import SaveModelStrategy

# Utility Packages
from utils import get_device


@hydra.main(config_path="docs/config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """
    Main function to run MS-Net federated learning
    :param cfg: An omegaconf object that stores the hydra config.
    :return:
    """


if __name__ == "__main__":
    main()