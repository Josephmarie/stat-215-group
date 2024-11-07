import yaml

import torch
import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from model import UNetSegmenter
from data import CloudDataset


# Load hyperparameter search configuration
with open("hyperparameter_search_config.yaml") as config_file:
    hyperparameter_config = yaml.safe_load(config_file)

# Set up data paths
train_images = ["../../../data/image_data/image1.txt", "../../../data/image_data/image2.txt"]


# Define the training function
def hyperparameter_search(config=None):
    with wandb.init(config=config):
        config = wandb.config

        # Cross-validation setup
        per_fold_results = []
        for idx, (train_image, val_image) in enumerate(
                [(train_images[0], train_images[1]), (train_images[1], train_images[0])]
        ):
            print(f"Training fold {idx + 1} with config: {config}")
            # Load datasets
            train_dataset = CloudDataset([train_image], augment=True)
            val_dataset = CloudDataset([val_image], augment=False, train=False)

            train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE)

            # Initialize model
            model = UNetSegmenter(
                learning_rate=config.learning_rate,
                weight_decay=config.weight_decay
            )

            # Early stopping callback
            early_stopping = EarlyStopping(
                monitor="val_auroc",
                mode="max",
                patience=config.patience,
                verbose=True
            )

            # Set up trainer with early stopping and logging
            trainer = Trainer(
                max_epochs=config.max_epochs,
                gpus=1 if torch.cuda.is_available() else 0,
                logger=WandbLogger(),
                callbacks=[early_stopping]
            )

            # Train
            trainer.fit(model, train_loader, val_loader)
            # Validate
            results = trainer.validate(model, val_loader)
            auroc = results[0]["val_auroc"]
            per_fold_results.append(auroc)
            wandb.log({f"fold_{idx}_val_auroc": auroc})

        # Average AUROC over folds
        avg_auroc = sum(per_fold_results) / len(per_fold_results)
        wandb.log({"avg_val_auroc": avg_auroc})


if __name__ == "__main__":
    # Initialize sweep
    sweep_id = wandb.sweep(hyperparameter_config, project="cloud_detection")
    # Start hyperparameter search
    print("Starting hyperparameter search...")
    wandb.agent(sweep_id, hyperparameter_search, count=50)
