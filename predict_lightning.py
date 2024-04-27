import torch
from network import MS_Net
from lightning import Trainer

# Utility Packages
from pathlib import Path
import load_data
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def compute_permeability(vz) -> float:
    """
    Compute the permeability using the Z component of velocity using the method outlined in LBPM:
    https://github.com/OPM/LBPM/blob/12b80d2a34d33b0cdf95f4e5a8a36ca75b42de5c/models/MRTModel.cpp#L394

    Parameters:
        vz: 3D z-component of the velocity field

    Returns:
        Permeability [L.U.^2]
    """
    flow_rate = torch.mean(vz)
    mu = (0.7 - 0.5) / 3
    porosity = np.count_nonzero(vz != 0) / np.prod(vz.shape)
    absperm = mu * porosity * porosity * flow_rate

    return absperm

@hydra.main(config_path="conf/", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:

    # Look for an updated checkpoint, instantiate a new model if it does not exist
    model_dir = Path(f"{cfg['model_dir']}")
    # Load a client
    if "client" in cfg['model_dir']:
        model_loc = list(model_dir.glob(f"round_{cfg['round']}/*.ckpt"))[-1]

        print(f"Attempting client model load from {str(model_loc)}")

        yaml_dict = load_data.load_hparams(list(model_loc.parent.glob(f"lightning_logs/version_*/hparams.yaml"))[-1])
        model = MS_Net.load_from_checkpoint(model_loc,
                                            net_name=yaml_dict['net_name'],
                                            num_scales=yaml_dict['num_scales'],
                                            num_features=yaml_dict['num_features'],
                                            num_filters=yaml_dict['num_filters'],
                                            f_mult=yaml_dict['f_mult'],
                                            summary=False)

    elif "server" in cfg['model_dir']:
        model = instantiate(cfg.model)
        model_loc = list(model_dir.glob(f'server_round_{cfg.round}.ckpt'))[-1]

        print(f"Attempting model load from {str(model_loc)}")
        ckpt = torch.load(model_loc)
        model.load_state_dict(ckpt)

    else:
        raise FileNotFoundError("Could not find a model to load")

    (model_loc.parent / "predictions" / "images").mkdir(parents=True, exist_ok=True)
    (model_loc.parent / "predictions" / "slices").mkdir(parents=True, exist_ok=True)
    model.eval()

    # try:
    #     model_dir = Path(f'server_models/')
    #     model_loc = list(model_dir.glob(f'server_round_{cfg.round}.ckpt'))[-1]
    #     print(f"Attempting model load from {str(model_loc)}")
    #     ckpt = torch.load(model_loc)
    #     model.load_state_dict(ckpt)
    #     model.eval()
    #
    #     print(f"Model loaded successfully!")
    #
    # except RuntimeError:
    #     model_dir = Path(f"{cfg['model_dir']}/round_{self.cfg['round']}")
    #     model_loc = list(model_dir.glob('*.ckpt'))[-1]
    #     print(f"Attempting model load from {str(model_loc)}")
    #
    #     yaml_dict = load_data.load_hparams(list(model_dir.glob("lightning_logs/version_*/hparams.yaml"))[-1])
    #     self.net = MS_Net.load_from_checkpoint(model_loc,
    #                                       net_name=yaml_dict['net_name'],
    #                                       num_scales=yaml_dict['num_scales'],
    #                                       num_features=yaml_dict['num_features'],
    #                                       num_filters=yaml_dict['num_filters'],
    #                                       f_mult=yaml_dict['f_mult'],
    #                                       summary=False)
    #
    # except (FileNotFoundError, IndexError):
    #     raise FileNotFoundError("Could not find a model for that round")


    # Load the datasets
    test_set = load_data.load_data(cfg.test_input_file, path_to_data=cfg.data_loc, phases=["test"])
    permeability = np.empty((len(test_set[0].dataset), 2), dtype=np.float64)
    predictor = Trainer()
    predictions = predictor.predict(model, test_set)

    for sample, prediction in enumerate(predictions, start=1):
        #######################
        # Model Predictions
        #######################
        y, y_pred = prediction
        y = y[-1].cpu().squeeze()
        y_pred = y_pred[-1].cpu().squeeze()

        y_rel_err = abs(y - y_pred) / abs(y)
        y_rel_err[torch.isnan(y_rel_err)] = 0

        #######################
        # Prediction statistics
        #######################
        y_var = y.var()

        # means
        y_m      = y.mean().item()
        y_pred_m = y_pred.mean().item()

        # errors
        L2   = ((y-y_pred)**2).mean().item()
        L1   = (y-y_pred).mean().item()
        Lend = L2/y_var
        #L1   = np.abs(y_m-y_pred_m)
        er   = np.abs(y_m-y_pred_m)/y_m*100

        # print metrics
        #append_list(file_name, metrics:=[phase, name, f'{y_var:.3}',
        #                            f'{y_m:.3}', f'{y_pred_m:.3}',
        #                            f'{L1:.3}',f'{L2:.3}', f'{Lend:.3}',
        #                            f'{er:.3}'])
        target_k = compute_permeability(y)
        predicted_k = compute_permeability(y_pred)
        # print(f"Target Absolute Permeability = {target_k}")
        # print(f"Predicted Absolute Permeability = {predicted_k}")

        permeability[sample-1] = target_k, predicted_k

        plt.figure(dpi=400, figsize=(8, 4))
        plt.subplot(1, 3, 1)
        y_true = plt.imshow(y[:, :, 128], cmap="inferno")
        plt.title("$y$")
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.axis('off')

        # Get the color range of subplot 1 to match subplot 2
        cmin, cmax = y_true.get_clim()

        plt.subplot(1, 3, 2)
        plt.imshow(y_pred[:, :, 128], vmin=cmin, vmax=cmax, cmap="inferno")
        plt.title("$\hat{y}$")
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(y_rel_err[:, :, 128], vmax=np.nanpercentile(y_rel_err[:, :, 128], 98), cmap="inferno")
        plt.title("Rel. Error")
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.axis('off')


        plt.suptitle(f"Sample: {sample}")

        plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.3, hspace=0.3)
        plt.savefig(model_loc.parent / f"predictions/slices/sample_{sample}.png")
        plt.close()
        np.save(model_loc.parent / f"./predictions/images/sample_{sample}.npy", y_pred)

    plt.figure(dpi=400)
    plt.plot(permeability[:, 0], permeability[:, 1], 'o')
    plt.plot([0, max(permeability[:, 0])], [0, max(permeability[:, 0])], 'k--')
    plt.xlabel('Predicted')
    plt.ylabel('Target')
    plt.savefig(model_loc.parent / f"predictions/permeability.png")
    plt.show()

    perm_df = pd.DataFrame(permeability, columns=["Target", "Predicted"])
    perm_df.to_parquet(model_loc.parent / f"predictions/permeability.parquet")


if __name__ == "__main__":
    main()