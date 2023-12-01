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
from opacus import PrivacyEngine
from time import perf_counter_ns

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
        super().__init__()
        net = instantiate(cfg.model).to(cfg.device)
        optimizer = instantiate(cfg.optimizer, params=net.parameters())

        self.valloader = valloader
        self.cfg = cfg

        if self.cfg.dp.use:
            self.privacy_engine = PrivacyEngine()
            self.net, self.optimizer, self.trainloader = self.privacy_engine.make_private(
                module=net,
                optimizer=optimizer,
                data_loader=trainloader,
                max_grad_norm=self.cfg.dp.max_grad_norm,
                noise_multiplier=self.cfg.dp.noise_multiplier,
            )
        else:
            self.privacy_engine = None
            self.net = net
            self.optimizer = optimizer
            self.trainloader = trainloader

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
        start_time = perf_counter_ns()
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)
        end_time = perf_counter_ns()
        print(f"Time taken to set parameters: {end_time - start_time} ns")

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

        start_time = perf_counter_ns()
        proximal_mu = config["proximal_mu"] if self.cfg.strategy == "FedProx" else None

        results = train(self.net, self.trainloader, self.valloader, self.optimizer, epochs=config["epochs"],
                        privacy_engine=self.privacy_engine, device=self.cfg.device, proximal_mu=proximal_mu,
                        fedprox=self.cfg.strategy == "fedprox")
        end_time = perf_counter_ns()
        print(f"Time taken to fit: {end_time - start_time} ns")
        save_path = HydraConfig.get().runtime.output_dir
        df = pd.DataFrame([results["epsilon"]], columns=["Epsilon"])

        utils.append_csv(df, file=f"{save_path}/epsilon.csv")
        return self.get_parameters(config={}), len(self.trainloader), {"epsilon": results["epsilon"]}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        start_time = perf_counter_ns()
        loss = test(self.net, self.valloader, device=self.cfg.device)
        end_time = perf_counter_ns()
        print(f"Time taken to evaluate: {end_time - start_time} ns")
        # Append to end of results excel
        save_path = HydraConfig.get().runtime.output_dir
        df = pd.DataFrame([[float(loss)]], columns=["Loss_Distributed"])

        utils.append_csv(df, file=f"{save_path}/round_loss_distributed.csv")
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


