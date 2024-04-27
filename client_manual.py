import warnings

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
import os
from pathlib import Path
os.environ['CURL_CA_BUNDLE'] = ''

warnings.filterwarnings("ignore", category=UserWarning)
class MSNet_Client(fl.client.NumPyClient):
    """
    Initialize a flower client with specified network
    ...
    Attributes:
        net: torch.nn.Module
            Network to be used for each client
        model_dict:
            Dictionary of model parameters for model setup
    Methods:
        get_parameters:
        set_parameters:
        fit:
        evaluate:
    """
    def __init__(self, cfg):
        """
        Constructs attributes for the Flower client
        Parameters:
            net:
            model_dict:
        """
        self.net = instantiate(cfg.model).to(cfg.device)
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
        Update the client with a saved model
        Args:
            parameters: Parameters from central model
            config: Configuration of the model

        Returns:
            Locally updated model parameters and number of training samples
        """
        model_dir = Path(f"{self.cfg['model_dir']}/round_{self.cfg['round']}")
        model_loc = list(model_dir.glob('*.ckpt'))[-1]
        print(f"Attempting model load from {str(model_loc)}")

        yaml_dict = load_data.load_hparams(list(model_dir.glob("lightning_logs/version_*/hparams.yaml"))[-1])
        self.net = MS_Net.load_from_checkpoint(model_loc,
                                          net_name=yaml_dict['net_name'],
                                          num_scales=yaml_dict['num_scales'],
                                          num_features=yaml_dict['num_features'],
                                          num_filters=yaml_dict['num_filters'],
                                          f_mult=yaml_dict['f_mult'],
                                          summary=False)

        with open(model_dir / "training_size.txt") as f:
            training_size = f.read()


        return self.get_parameters(self.net), int(training_size), {}

    # def evaluate(self, parameters, config):
    #     self.set_parameters(parameters)
    #
    #     loss = test(self.net, self.valloader, device=self.cfg.device)
    #
    #     # Append to end of results excel
    #     save_path = HydraConfig.get().runtime.output_dir
    #     df = pd.DataFrame([[float(loss)]], columns=["Loss_Distributed"])
    #
    #     utils.append_csv(df, file=f"{save_path}/round_loss_distributed.xlsx")
    #     return float(loss), len(self.valloader), {"loss": float(loss)}


@hydra.main(config_path="conf/", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:

    # Instantiate Flower client
    client = MSNet_Client(cfg)

    # Start Flower client
    fl.client.start_client(server_address="129.114.35.162:443",
                           client=client.to_client(),
                           #root_certificates=Path("/etc/ssl/certs/ISRG_Root_X1.pem").read_bytes(),
                                 # TODO: Add security certificates if needed
                                 # https://github.com/adap/flower/blob/821d843278e60c55acdfb3574de8958c26f7a644/src/py/flwr/client/app.py#L242
                                 )

if __name__ == "__main__":
    main()


