import torch.nn as nn
import torch
from network_utils import calc_loss

from hydra.utils import instantiate
from omegaconf import DictConfig

from collections import OrderedDict

from utils import get_device
from tqdm import tqdm

def train(net, trainloader, valloader, epochs: int, learning_rate: float = 1.E-5,
          loss_f=nn.MSELoss(), optimizer_f=torch.optim.Adam,
          device: str="cpu"):
    """
    Function for training loop
    Parameters:
        net: torch.nn.Module
            Network to be used for each client
        trainloader:
            Training set dataloader
        valloader:
            Validation set dataloader
        epochs: int
            Number of epochs to train for
        learning_rate: float
            Learning rate for optimizer
        loss_f: torch.nn.Module
            Loss function
        optimizer_f: torch.optim.Optimizer
            Optimizer function
        DEVICE: str
            Device on which to run the model
    Returns:
        results: Dict
            Dictionary with training and validation losses
    """

    # Set up training parameters
    optimizer = optimizer_f(net.parameters(), lr=learning_rate)  # Initialize optimizer
    print("Optimizer Initialized")
    train_loss = 1e9  # Initialize value of training loss
    val_loss = 1e9  # Initialize value of validation loss
    net.train()

    # Training loop
    print("Starting training")
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
            loss = calc_loss(y_hat, y_n, loss_f)
            # credit assignment
            loss.backward()

        # update model weights
        optimizer.step()

        # Compute validation loss
        # TODO: Get config file for val step instead of hard-coding and log the validation loss
        if epoch % 10:
            print(f"{epoch = }")
            val_loss = test(net, valloader, loss_f, device)

        results = {"train_loss": train_loss, "val_loss": val_loss}

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

def get_on_fit_config(config: DictConfig):
    """
    Get the configuration for the on_fit callback
    """
    def fit_config_fn(server_round: int):
        return {
            "lr": config.lr,
            "epochs": config.epochs,
        }
    return fit_config_fn

def get_evalulate_fn(model_cfg: int, testloader):
    """Return a function to evaluate the global model."""

    def evaluate_fn(server_round: int, parameters, config):
        model = instantiate(model_cfg)

        device = get_device()

        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

        loss, accuracy = test(model, testloader, device)

        return loss, {"accuracy": accuracy}

    return evaluate_fn


