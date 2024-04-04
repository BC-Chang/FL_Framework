import warnings
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import flwr as fl
import torch
import utils
from collections import OrderedDict
from hydra.utils import instantiate
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from tasks import train, test
import load_data
from network import MS_Net

warnings.filterwarnings("ignore", category=UserWarning)
class MSNet_Client(fl.client.NumPyClient):
    """
    Initialize a flower client with specified network
    ...
    Attributes:
        net: torch.nn.Module
            Network to be used for each client
        trainloader:
            Training set dataloader
        valloader:
            Validation set dataloader
        model_dict:
            Dictionary of model parameters for model setup
    Methods:
        get_parameters:
        set_parameters:
        fit:
        evaluate:
    """
    def __init__(self, trainloader, valloader, cfg):
        """
        Constructs attributes for the Flower client
        Parameters:
            net:
            trainloader:
            valloader:
            model_dict:
        """
        self.net = instantiate(cfg.model).to(cfg.device)
        self.trainloader = trainloader
        self.valloader = valloader
        self.cfg = cfg

    def get_parameters(self, config):
        """
        Get parameters for the client
        """
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        """
        Set parameters for the client
        Args:
            parameters: Model parameters to be set

        Returns:

        """
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        """
        Train the client
        Args:
            parameters: Parameters from central model
            config: Configuration of the model

        Returns:
            Locally updated model parameters and number of training samples
        """

        self.set_parameters(parameters)
        # TODO: Optimizer from config file
        optimizer = instantiate(self.cfg.optimizer, params=self.net.parameters())
        if self.cfg.strategy == "FedProx":
            results = train_fedprox(self.net, self.trainloader, self.valloader, optimizer, epochs=config["epochs"],
                            device=self.cfg.device, proximal_mu=config["proximal_mu"])
        else:
            results = train(self.net, self.trainloader, self.valloader, optimizer, epochs=config["epochs"],
                            batch_size=config["batch_size"], val_interval=config["val_interval"], device=self.cfg.device)

        return self.get_parameters(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)

        loss = test(self.net, self.valloader, device=self.cfg.device)

        # Append to end of results excel
        save_path = HydraConfig.get().runtime.output_dir
        df = pd.DataFrame([[float(loss)]], columns=["Loss_Distributed"])

        utils.append_csv(df, file=f"{save_path}/round_loss_distributed.xlsx")
        return float(loss), len(self.valloader), {"loss": float(loss)}


@hydra.main(config_path="conf/", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:

    # TODO: Load data from a specific datafile
    # Load local data partition
    trainset, valset = load_data.load_data(cfg.train_input_file, path_to_data=cfg.data_loc, phases=["train", "val"])

    # Instantiate Flower client
    client = MSNet_Client(trainset, valset, cfg)


    # Start Flower client
    fl.client.start_numpy_client(server_address="127.0.0.1:8080",
                                 client=client,
                                 # TODO: Add security certificates if needed
                                 # https://github.com/adap/flower/blob/821d843278e60c55acdfb3574de8958c26f7a644/src/py/flwr/client/app.py#L242
                                 )

if __name__ == "__main__":
    main()


