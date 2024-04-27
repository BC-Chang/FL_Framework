import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


def load_client_round_losses(client_path: Path, train_round: int) -> pd.DataFrame:
    """
    Load the client loss log

    Parameters:
        client_path: Path to the client parent directory
        train_round: Federated training round number

    Returns:
        pd.DataFrame: Client loss log
    """
    # Load the metrics file
    log_path = list((client_path / f'round_{train_round}' / 'lightning_logs').glob("*/metrics.csv"))[-1]
    loss_log = pd.read_csv(log_path, delimiter=",")

    try:
        # Shift the val loss down one row to get it on the same row as train loss
        loss_log.loc[:, "val_loss_epoch":] = loss_log.loc[:, "val_loss_epoch":].shift(1)
        # Drop rows where train/val loss is nan - this logs individual batch steps.
        loss_log = loss_log.dropna(subset=["loss_epoch", "val_loss_epoch"], how='all', ignore_index=True)
    except KeyError:
        print("No validation loss for this round :(")
        loss_log = loss_log.dropna(subset=["loss_epoch"], how='all', ignore_index=True)

    return loss_log

def load_client_losses(client_path: Path, n_rounds: int) -> pd.DataFrame:
    """
    Load the client models from round [1:n]

    Parameter:
        client_path: Path to the client parent directory
        n_rounds: Total number of training rounds

    Returns:
        pd.DataFrame: DataFrame of client losses for each round.
    """
    loss_log = load_client_round_losses(client_path, train_round=1)
    for round_num in range(2, n_rounds+1):
        loss_log_tmp = load_client_round_losses(client_path, train_round=round_num)
        loss_log_tmp['epoch'] += loss_log['epoch'].values[-1] + 1
        loss_log = pd.concat([loss_log, loss_log_tmp], axis=0, ignore_index=True)

    return loss_log

def plot_losses(loss_df: pd.DataFrame, fig: plt.Figure=None, ax: plt.Axes=None, **kwargs) -> list[plt.Figure, plt.Axes]:
    """
    Plot the loss curves.
    """
    if fig is None and ax is None:
        fig, ax = plt.subplots(1, 1, dpi=400, figsize=(5, 4))

    sns.lineplot(data=loss_df, x='epoch', y='loss_epoch', ax=ax, linestyle='-',
                 linewidth=1, **kwargs)
    try:
        sns.lineplot(data=loss_df, x='epoch', y='val_loss_epoch', ax=ax, linestyle='--', linewidth=1, **kwargs)
    except (KeyError, ValueError):
        pass

    plt.yscale('log')
    plt.ylabel("Loss")
    plt.xlabel("Epoch")

    return fig, ax



if __name__ == "__main__":
    parent_path = Path(r"C:\Users\bcc2459\Box\Research\Conference_Stuff\SCA_2024\client_models")

    ut_df = load_client_losses(parent_path / "ut", n_rounds=10,)
    petrobras_df = load_client_losses(parent_path / "petrobras", n_rounds=10,)
    bp_df = load_client_losses(parent_path / "bp", n_rounds=10)
    # server_df = pd.read_excel(parent_path / "server_loss.xlsx")
    # server_df['epoch'] = server_df["Round"] * 10

    fig, ax = plot_losses(ut_df, color="#bf5700")
    fig, ax = plot_losses(petrobras_df, color="#FFC632", ax=ax, fig=fig)
    fig, ax = plot_losses(bp_df, color="#005f86", ax=ax, fig=fig)
    # sns.lineplot(data=server_df, x='epoch', y='Central_Loss', ax=ax, linestyle=':', linewidth=2.5, color='k')

    legend_elements = [Line2D([0], [0], color="k", lw=2, label="Client Training Loss"),
                       Line2D([0], [0], color="k", linestyle="--", lw=2, label="Client Validation Loss"),
                       Patch(color="#005f86", lw=4, label="bp"),
                       Patch(color="#FFC632", lw=4, label="Petrobras"),
                       Patch(color="#bf5700", lw=4, label="UT")]

    ax.legend(handles=legend_elements, fontsize=8, loc='upper right')
    ax.set_ylim([5e-1, 4e3])
    fig.show()
