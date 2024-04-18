import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

# FL & ML parameters
import flwr as fl
import torch
# from agg_strats import SaveFedAvg
# from flwr.server.strategy import FedAvg
from network import MS_Net
import utils
from tasks import weighted_average, get_evaluate_fn
import load_data
import pandas as pd
from pathlib import Path

# Set GRPC Poll strategy. epollex is standard implementation, but it causes a bug. Use epoll1 instead
import os
os.environ["GRPC_POLL_STRATEGY"] = "epoll1"

# Get IP
import socket
ip_address = socket.gethostbyname((socket.gethostname()))
server_address = ip_address + ":443"

@hydra.main(config_path="conf/", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Load model for
       1. server-side parameter initialization
       2. server-side parameter evaluation
    """

    # Parse config * get experiment output directory
    save_path = HydraConfig.get().runtime.output_dir

    model = hydra.utils.instantiate(cfg.model)
    
    # Try to load from checkpoint
    try:
        model_dir = Path(f"./server_models/")
        model_loc = list(model_dir.glob(f"server_round_{cfg.round-1}.ckpt"))[-1]
        print(model_loc)
        print(f"Attempting model load from {str(model_loc)}")
        ckpt = torch.load(model_loc)
        model.load_state_dict(ckpt)
        print(f"Successfully loaded server model from round {cfg.round-1}!")

    except (FileNotFoundError, IndexError):
        print("Could not load model... instantiating a new model")
    

    # Get model parameters
    model_parameters = utils.get_model_parameters(model)

    # Centralized test data
    testloader = load_data.load_data(cfg.test_input_file, path_to_data=cfg.data_loc, phases=['test'])[0]
    model_dict = dict((key, cfg.model[key]) for key in ["num_scales", "num_features", "num_filters", "f_mult"])
    print(cfg["device"])
    model_dict["device"] = cfg["device"]
    model_dict["round"] = cfg["round"]
    # Instantiate a strategy
    strategy = hydra.utils.instantiate(cfg.strategy, model_dict, evaluate_metrics_aggregation_fn=weighted_average,
                                       evaluate_fn=get_evaluate_fn(cfg.model, testloader, cfg.device))

    # Start Flower server for four rounds of federated learning
    history = fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds,
                                      round_timeout=cfg.round_timeout),
        strategy=strategy,
        # TODO: Add security certificates here if needed
        #certificates=(
        #    Path("/home/bchang/flower/examples/advanced-tensorflow/.cache/certificates/ca.crt").read_bytes(),
        #    Path("/home/bchang/flower/examples/advanced-tensorflow/.cache/certificates/server.pem").read_bytes(),
         #   Path("/home/bchang/flower/examples/advanced-tensorflow/.cache/certificates/server.key").read_bytes(),)
        # https://github.com/adap/flower/blob/821d843278e60c55acdfb3574de8958c26f7a644/src/py/flwr/server/app.py#L117
    )

    # Get global parameters with:
    print(history)
    print("All done :)")
    df = pd.DataFrame(history.losses_centralized, columns=["Round", "Loss_Centralized"])
    df.to_csv(f"{save_path}/losses_centralized.csv", index=False)

if __name__ == "__main__":
    main()
