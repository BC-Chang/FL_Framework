from copy import deepcopy
import numpy as np
from collections import OrderedDict

from typing import Dict, Optional, Tuple, List
import torch
from flwr.server.strategy import FedAvg
from flwr.common.typing import Parameters, FitIns, FitRes
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager
from flwr.common import Parameters, Scalar, parameters_to_ndarrays
from lightning import Trainer

from network import MS_Net


class SaveFedAvg(FedAvg):
    def __init__(self, model_dict, *args, **kwargs):
        # self.save_path = save_path

        self.net = MS_Net(num_scales=model_dict['num_scales'],
                          num_features=model_dict['num_features'],
                          num_filters=model_dict['num_filters'],
                          device=model_dict['device'],
                          f_mult=model_dict['f_mult'],
                          summary=False).to(model_dict['device'])
        super().__init__(*args, **kwargs)

    def aggregate_fit(self, rnd: int, results, failures,):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(rnd, results, failures)

        if aggregated_parameters is not None:
            # Save weights
            aggregated_ndarrays: List[np.ndarray] = parameters_to_ndarrays(aggregated_parameters)
            # Convert List[np.ndarray] to Pytorch 'state_dict'
            params_dict = zip(self.net.state_dict().keys(), aggregated_ndarrays)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.net.load_state_dict(state_dict, strict=True)

            # Save the model
            print(f"Saving round {rnd} weights...")
            trainer = Trainer(model=self.net, max_epochs=0, max_steps=0, default_root_dir="server_models")
            trainer.save_checkpoint(f"./server_round_{rnd}.ckpt")
            # torch.save(self.net.state_dict(), f"./server_models/server_round_{rnd}.ckpt")

        return aggregated_parameters, aggregated_metrics

