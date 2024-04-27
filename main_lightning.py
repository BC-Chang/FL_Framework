import torch
from network import MS_Net
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

# Hydra CLI Packages
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

# Utility Packages
from pathlib import Path
import load_data


@hydra.main(config_path="conf/", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
    # Look for an updated checkpoint, instantiate a new model if it does not exist

    model = instantiate(cfg.model)#.to(cfg.device)

    try:
        model_dir = Path(f'server_models/')
        model_loc = list(model_dir.glob(f'server_round_{cfg.round}.ckpt'))[-1]
        print(f"Attempting model load from {str(model_dir)}")
        ckpt = torch.load(model_loc)
        model.load_state_dict(ckpt)
        # yaml_dict = load_data.load_hparams(list(model_dir.glob("lightning_logs/version_*/hparams.yaml"))[-1])
        # model = MS_Net.load_from_checkpoint(model_loc,
        #                                     num_scales=cfg.model['num_scales'],
        #                                     num_features=cfg.model['num_features'],
        #                                     num_filters=cfg.model['num_filters'],
        #                                     f_mult=cfg.model['f_mult'],
        #                                     summary=False)

        print(f"Model loaded successfully!")

    except (FileNotFoundError, IndexError):
        print("No checkpoint found, instantiating a new model...")
        print(cfg.model)

        model_loc = None

    # Load the datasets
    trainset, valset = load_data.load_data(cfg.train_input_file, path_to_data=cfg.data_loc, phases=["train", "val"])

    # Callbacks
    callbacks = [ModelCheckpoint(dirpath=f"client_model/round_{cfg.round+1}",
                                 save_on_train_epoch_end=True,
                                 save_weights_only=True)]

    # Instantiate a Lightning Trainer
    trainer = Trainer(precision="32",
                      max_epochs=cfg.config_fit.local_epochs,
                      min_epochs=1,
                      accumulate_grad_batches=cfg.config_fit.batch_size,
                      check_val_every_n_epoch=cfg.config_fit.val_interval,
                      default_root_dir=f"client_model/round_{cfg.round+1}",
                      callbacks=callbacks,
                      strategy='auto')

    # Train the model, reset everything but the weights
    trainer.fit(model, trainset, valset) #, ckpt_path=model_loc)

    # Write the training set size to a text file
    with open(f"client_model/round_{cfg.round+1}/training_size.txt", "w") as f:
        f.write(f"{len(trainset)}")



if __name__ == "__main__":
    main()
