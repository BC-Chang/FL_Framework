import torch.nn as nn
import torch
from network_utils import calc_loss
import pandas as pd

from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from collections import OrderedDict

from utils import get_device, append_csv
from tqdm import tqdm


def train(net, trainloader, valloader, optimizer, epochs: int, loss_f=nn.MSELoss(), device: str="cpu"):
    """
    Function for training loop
    Parameters:
        net: torch.nn.Module
            Network to be used for each client
        trainloader:
            Training set dataloader
        valloader:
            Validation set dataloader
        optimizer: torch.optim.Optimizer
            Optimizer, to be instantiated in client
        epochs: int
            Number of epochs to train for
        loss_f: torch.nn.Module
            Loss function
        device: str
            Device on which to run the model
    Returns:
        results: Dict
            Dictionary with training and validation losses
    """

    # Set up training parameters
    # optimizer = optimizer_f(net.parameters(), lr=learning_rate)  # Initialize optimizer
    train_loss = 1e9  # Initialize value of training loss
    val_loss = 1e9  # Initialize value of validation loss
    net.train()

    # Training loop
    for epoch in tqdm(range(epochs)):
        optimizer.zero_grad()

        # Loop through the training batches - Gradient accumulation
        for _ in range(len(trainloader)):
            # Get the next batch
            sample, masks, xy = next(iter(trainloader))
            x_n, y_n = xy[0], xy[1]
            # Send data to cpu or gpu
            masks, x_n, y_n = [[elem.to(device) for elem in data] for data in [masks, x_n, y_n]]

            # compute the model output
            y_hat = net(x_n, masks)
            # TODO: Log loss values
            loss_prev = train_loss
            train_loss = calc_loss(y_hat, y_n, loss_f)
            # credit assignment
            train_loss.backward()

        # update model weights
        optimizer.step()

        # Compute validation loss
        # TODO: Get config file for val step instead of hard-coding and log the validation loss
        if epoch % 1 == 0:
            val_loss = test(net, valloader, loss_f, device)

        results = {"train_loss": train_loss, "val_loss": val_loss}
        print(results)

    return results

def train_fedprox(net, trainloader, valloader, optimizer, epochs: int, proximal_mu: float, loss_f=nn.MSELoss(), device: str="cpu"):
    # Set up training parameters
    # optimizer = optimizer_f(net.parameters(), lr=learning_rate)  # Initialize optimizer
    train_loss = 1e9  # Initialize value of training loss
    val_loss = 1e9  # Initialize value of validation loss
    global_params = [param.detach().clone() for param in net.parameters()]
    net.train()

    # Training loop
    for epoch in tqdm(range(epochs)):
        optimizer.zero_grad()

        # Loop through the training batches - Gradient accumulation
        for _ in range(len(trainloader)):
            # Get the next batch
            sample, masks, xy = next(iter(trainloader))
            x_n, y_n = xy[0], xy[1]
            # Send data to cpu or gpu
            masks, x_n, y_n = [[elem.to(device) for elem in data] for data in [masks, x_n, y_n]]

            # compute the model output
            y_hat = net(x_n, masks)
            # TODO: Log loss values
            loss_prev = train_loss
            train_loss = calc_loss(y_hat, y_n, loss_f)
            # FedProx term
            proximal_term = 0.0
            for param, global_param in zip(net.parameters(), global_params):
                proximal_term += torch.norm(param - global_param) ** 2
            train_loss += 0.5*proximal_mu * proximal_term
            # credit assignment
            train_loss.backward()

        # update model weights
        optimizer.step()

        # Compute validation loss
        # TODO: Get config file for val step instead of hard-coding and log the validation loss
        if epoch % 1 == 0:
            val_loss = test(net, valloader, loss_f, device)

        results = {"train_loss": train_loss, "val_loss": val_loss}
        print(results)

    return results


def test(net, testloader, loss_f=nn.MSELoss(), device: str = "cpu"):
    """
    Validate the network on the entire test set.
    Parameters:
        net: torch.nn.Module
            Network to be used for each client
        testloader:
            Validation set dataloader
        loss_f: torch.nn.Module
            Loss function
        device: str
            Device on which to run the model
    Returns:
        loss: float
            Summed loss over the entire test set
    """

    net.to(device)  # move model to specified device
    loss = 0.0
    net.eval()
    with torch.no_grad():
        for _ in enumerate(testloader):
            sample, masks, xy = next(iter(testloader))
            x_n, y_n = xy[0], xy[1]
            # Send data to cpu or gpu
            masks, x_n, y_n = [[elem.to(device) for elem in data] for data in [masks, x_n, y_n]]

            y_hat = net(x_n, masks)
            # TODO: Log loss values
            loss += calc_loss(y_hat, y_n, loss_f)

    return loss


def weighted_average(metrics):
    # Multiply accuracy of each client by number of examples used
    loss = [m["loss"] for _, m in metrics]
    # accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"loss": sum(loss)}


def get_on_fit_config(config: DictConfig):
    """
    Get the configuration for the on_fit callback
    """
    def fit_config_fn(server_round: int):
        return {
            "lr": config.lr,
            "epochs": config.local_epochs,
        }
    return fit_config_fn


def get_evaluate_fn(model_cfg: int, testloader, device: str):
    """Return a function to evaluate the global model."""

    def evaluate_fn(server_round: int, parameters, config):

        save_path = HydraConfig.get().runtime.output_dir
        model = instantiate(model_cfg).to(device)

        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

        loss = test(model, testloader, device=device)

        # Append to end of results excel
        df = pd.DataFrame([[server_round, float(loss)]], columns=["Round", "Loss_Centralized"])

        append_csv(df, file=f"{save_path}/round_loss_centralized.xlsx")

        return float(loss), {}

    return evaluate_fn

