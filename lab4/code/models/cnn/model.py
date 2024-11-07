import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from torchmetrics.classification import (
    BinaryAccuracy, BinaryAUROC, BinaryPrecision, BinaryRecall, BinaryF1Score, BinaryAveragePrecision
)


class UNetSegmenter(pl.LightningModule):
    def __init__(
        self,
        in_channels: int = 8,
        out_channels: int = 1,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
    ):
        super().__init__()

        # Initialize model weights
        self._init_contracting_path(in_channels)
        self._init_expansive_path(out_channels)

        # Use the binary cross-entropy loss function
        self.loss_fn = nn.BCELoss()

        # Initialize optimizer parameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # Performance metrics
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_metrics = {
            "train_accuracy": BinaryAccuracy().to(device),
            "train_auroc": BinaryAUROC().to(device),
            "train_precision": BinaryPrecision().to(device),
            "train_recall": BinaryRecall().to(device),
            "train_f1": BinaryF1Score().to(device),
            "train_average_precision": BinaryAveragePrecision().to(device),
        }
        self.val_metrics = {
            "val_accuracy": BinaryAccuracy().to(device),
            "val_auroc": BinaryAUROC().to(device),
            "val_precision": BinaryPrecision().to(device),
            "val_recall": BinaryRecall().to(device),
            "val_f1": BinaryF1Score().to(device),
            "val_average_precision": BinaryAveragePrecision().to(device),
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Contracting Path
        # In each contracting double convolution, we increase the number of channels by a factor of 2
        # After each convolution layer, we apply ReLU activation
        # After each double convolution, we perform max pooling with kernel size 2 and stride 2
        # to reduce the spatial dimensions by half
        x1 = F.relu(self.contracting_conv1(x))
        x1 = F.relu(self.contracting_conv2(x1))
        x1_pooled = self.pool(x1)

        x2 = F.relu(self.contracting_conv3(x1_pooled))
        x2 = F.relu(self.contracting_conv4(x2))
        x2_pooled = self.pool(x2)

        x3 = F.relu(self.contracting_conv5(x2_pooled))
        x3 = F.relu(self.contracting_conv6(x3))
        x3_pooled = self.pool(x3)

        x4 = F.relu(self.contracting_conv7(x3_pooled))
        x4 = F.relu(self.contracting_conv8(x4))
        x4_pooled = self.pool(x4)

        x5 = F.relu(self.contracting_conv9(x4_pooled))
        x5 = F.relu(self.contracting_conv10(x5))

        # Expanding Path
        # In each expanding block, we upsample the spatial dimensions by a factor of 2
        # After each transposed convolution, we concatenate the feature maps from the corresponding contracting block
        # After the concatenation, we apply two convolution layers with ReLU activation
        # to reduce the number of channels by a factor of 2

        x = self.up_conv1(x5)
        x = torch.cat([x, x4], dim=1)
        x = F.relu(self.expanding_conv1(x))
        x = F.relu(self.expanding_conv2(x))

        x = self.up_conv2(x)
        x = torch.cat([x, x3], dim=1)
        x = F.relu(self.expanding_conv3(x))
        x = F.relu(self.expanding_conv4(x))

        x = self.up_conv3(x)
        x = torch.cat([x, x2], dim=1)
        x = F.relu(self.expanding_conv5(x))
        x = F.relu(self.expanding_conv6(x))

        x = self.up_conv4(x)
        x = torch.cat([x, x1], dim=1)
        x = F.relu(self.expanding_conv7(x))
        x = F.relu(self.expanding_conv8(x))

        # Final classification layer
        # The final convolutional layer reduces the number of channels to the number of output channels
        # that are used for classification and applies a sigmoid activation function
        x = self.final_conv(x)
        x = self.sigmoid(x)

        return x

    def training_step(self, batch, batch_idx):
        features, labels = batch
        labels = labels.unsqueeze(1).float()

        # Forward pass
        outputs = self(features)

        # Exclude unlabeled (=0) pixels from loss computation
        outputs, labels = self._prepare_inputs_for_loss_computation(outputs, labels)

        # Calculate loss for labeled pixels
        loss = self.loss_fn(outputs, labels)
        # Calculate metrics for labeled pixels
        labels = labels.long()
        for metric in self.train_metrics.values():
            metric(outputs, labels)

        # Log loss to W&B
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        features, labels = batch
        labels = labels.unsqueeze(1).float()

        # Forward pass
        outputs = self(features)

        # Exclude unlabeled (=0) pixels from loss computation
        outputs, labels = self._prepare_inputs_for_loss_computation(outputs, labels)

        # Calculate loss for labeled pixels
        loss = self.loss_fn(outputs, labels)
        # Calculate metrics for labeled pixels
        labels = labels.long()
        for metric in self.val_metrics.values():
            metric(outputs, labels)

        # Log loss to W&B
        self.log("val_loss", loss)

        return loss

    def on_train_epoch_end(self):
        # Log training metrics
        for name, metric in self.train_metrics.items():
            self.log(name, metric.compute())
            metric.reset()

    def on_validation_epoch_end(self):
        # Log validation metrics
        for name, metric in self.val_metrics.items():
            self.log(name, metric.compute())
            metric.reset()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return optimizer

    def _init_contracting_path(self, in_channels: int):
        """
        Initialize the weights of the contracting path of the U-Net model.

        :param in_channels: Number of input channels.
        """
        # After each double convolution, we perform max pooling with kernel size 2 and stride 2
        # to reduce the spatial dimensions by half
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # The convolutional layers have kernel size 3 and padding 1 to maintain the spatial dimensions
        # After each double convolution, we increase the number of channels by a factor of 2
        self.contracting_conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.contracting_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.contracting_conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.contracting_conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.contracting_conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.contracting_conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.contracting_conv7 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.contracting_conv8 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.contracting_conv9 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.contracting_conv10 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)

    def _init_expansive_path(self, out_channels: int):
        """
        Initialize the weights of the expansive path of the U-Net model.

        :param out_channels: Number of output channels.
        """
        # After each transposed convolution, which upsamples the spatial dimensions by a factor of 2,
        # we perform a double convolution to reduce the number of channels by a factor of 2
        self.up_conv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.expanding_conv1 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.expanding_conv2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.up_conv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.expanding_conv3 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.expanding_conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.up_conv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.expanding_conv5 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.expanding_conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.up_conv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.expanding_conv7 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.expanding_conv8 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # Final Convolution for classification
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    @staticmethod
    def _prepare_inputs_for_loss_computation(
        outputs: torch.Tensor,
        labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare the outputs and labels for loss computation by excluding unlabeled pixels.

        :param outputs: The network outputs.
        :param labels: The ground truth labels.
        :return: The prepared outputs and labels.
        """
        # Exclude unlabeled (=0) pixels from loss computation
        mask = (labels != 0)
        outputs = outputs[mask]
        labels = labels[mask]

        # BCE loss expects the target to be 0 or 1
        labels = (labels + 1) / 2

        return outputs, labels
