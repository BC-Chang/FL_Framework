import torch
from network import MS_Net
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
# Hydra CLI Packages
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

# Utility Packages
from pathlib import Path
import load_data


@hydra.main(config_path="conf/", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
    torch.set_float32_matmul_precision('medium')
    
    # Look for an updated checkpoint, instantiate a new model if it does not exist
    model = instantiate(cfg.model)#.to(cfg.device)

    wandb_logger = WandbLogger(project="FL_Framework", name=f"resolution_model{cfg.round}_client{cfg.client_num}", log_model="all")

    #try:
        #model_dir = Path(f'server_models/')
        #model_loc = list(model_dir.glob(f'server_round_{cfg.round}.ckpt'))[-1]
        #print(f"Attempting model load from {str(model_dir)}")
        #ckpt = torch.load(model_loc)
        #model.load_state_dict(ckpt)
        # yaml_dict = load_data.load_hparams(list(model_dir.glob("lightning_logs/version_*/hparams.yaml"))[-1])
        # model = MS_Net.load_from_checkpoint(model_loc,
        #                                     num_scales=cfg.model['num_scales'],
        #                                     num_features=cfg.model['num_features'],
        #                                     num_filters=cfg.model['num_filters'],
        #                                     f_mult=cfg.model['f_mult'],
        #                                     summary=False)

        #print(f"Model loaded successfully!")

    #except (FileNotFoundError, IndexError):
    #    print("No checkpoint found, instantiating a new model...")
    #    print(cfg.model)

    #    model_loc = None

    # Load the datasets
    trainset, valset = load_data.load_data(cfg.train_input_file, path_to_data=cfg.data_loc, phases=["train", "val"])

    # Log data parameters
    wandb_logger.experiment.config.update({"num_scales": cfg.model.num_scales,
                                           "x": "bin",
                                           "y": "vz",
                                           "n_train": len(trainset.dataset),
                                           "n_val": len(valset.dataset)})
    
    # Callbacks
    callbacks = [ModelCheckpoint(monitor="loss", dirpath=f"interpore/model_{cfg.round}_client_{cfg.client_num}",
                                 save_on_train_epoch_end=True,
                                 save_weights_only=True,
                                 every_n_epochs=1),
                 ModelCheckpoint(monitor="val_loss"),
                 EarlyStopping(monitor="val_loss", check_finite=False, patience=9999),
                 LearningRateMonitor(logging_interval='step')]

    wandb_logger.watch(model, log="all", log_freq=2)
                                          
    # Instantiate a Lightning Trainer
    trainer = Trainer(precision="32",
                      max_epochs=cfg.config_fit.local_epochs,
                      min_epochs=1,
                      log_every_n_steps=25,
                      logger=wandb_logger,
                      accumulate_grad_batches=cfg.config_fit.batch_size,
                      check_val_every_n_epoch=cfg.config_fit.val_interval,
                      default_root_dir=f"interpore/model_{cfg.round}_client_{cfg.client_num}",
                      callbacks=callbacks,
                      strategy='auto')

    # Train the model, reset everything but the weights
    trainer.fit(model, trainset, valset) #, ckpt_path=model_loc)

    # Write the training set size to a text file
    with open(f"interpore/model_{cfg.round}_client_{cfg.client_num}/training_size.txt", "w") as f:
        f.write(f"{len(trainset)}")



if __name__ == "__main__":
    main()
