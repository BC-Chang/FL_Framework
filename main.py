# IO Packages
import hydra
from omegaconf import DictConfig

# FL Libraries
import torch

# Hydra Packages
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

# Utility Packages
from utils import get_device
import load_data
from tasks import train

@hydra.main(config_path="conf/", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:

    # Instantiate an ms-net model
    net = instantiate(cfg.model).to(cfg.device)

    # Instantiate an optimizer
    optimizer = instantiate(cfg.optimizer, params=net.parameters())

    # TODO: Load data from a specific datafile
    # Load local data partition
    trainset, valset = load_data.load_data(cfg.train_input_file, path_to_data=cfg.data_loc, phases=["train", "val"])

    # Train the model
    train(net, trainset, valset, optimizer, epochs=100, batch_size=1, val_interval=10, device=cfg.device)



if __name__ == "__main__":
    main()
