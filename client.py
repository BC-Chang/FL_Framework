import numpy as np
import matplotlib.pyplot as plt
import flwr as fl
import torch
#import argparse
import utils
from collections import OrderedDict
from hydra.utils import instantiate
import hydra
from omegaconf import DictConfig, OmegaConf
from tasks import train, test
import load_data
from network import MS_Net

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
    def __init__(self, trainloader, valloader):
        """
        Constructs attributes for the Flower client
        Parameters:
            net:
            trainloader:
            valloader:
            model_dict:
        """
        self.net = MS_Net(
            num_scales=4,
            num_features=1,
            num_filters=2,
            device='cuda',
            f_mult=2,
            summary=False,).to("cuda")#instantiate(model_config)
        self.trainloader = trainloader
        self.valloader = valloader
        #self.model_config = model_config

    def get_parameters(self, **kwargs):
        """
        Get parameters for the client
        """
        return utils.get_model_parameters(self.net)

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

    def fit(self, parameters, conf):
        """
        Train the client
        Args:
            parameters: Parameters from central model
            conf: Configuration of the model

        Returns:
            Locally updated model parameters and number of training samples
        """
        # optimizer = torch.optim.Adam(self.net.parameters, lr=1e-5)#instantiate(conf.optimizer)
        # TODO: Get model parameters from config
        self.set_parameters(parameters)
        results = train(self.net, self.trainloader, epochs=10,
                 learning_rate=1e-5, DEVICE="cuda")

        return self.get_parameters(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)

        loss = test(self.net, self.valloader, self.device)


#@hydra.main(config_path="conf/model", config_name="msnet", version_base=None)
def main():#cfg: DictConfig) -> None:
    # Parse command line argument `partition`
    #parser = argparse.ArgumentParser(description="Flower Client")
    #parser.add_argument(
    #    "--input_data",
    #    type=str,
    #    required=True,
    #    help="Specify the path to the input data file to be used for client-side training",
    #)

    #args = parser.parse_args()

    # TODO: Load data from a specific datafile
    # Load local data partition
    trainset, testset = load_data.load_data("test_net.yml", split="random")

    # TODO: Instantiate Flower client
    client = MSNet_Client(trainset, testset)


    # Start Flower client
    # TODO: Check server address
    fl.client.start_numpy_client(server_address="127.0.0.1:8080",
                                 client=client,
                                 # TODO: Add security certificates if needed
                                 )

if __name__ == "__main__":
    main()


