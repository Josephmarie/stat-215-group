import random

import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as transforms

# Set batch size such that memory consumption is maximized on a 16GB GPU
BATCH_SIZE = 16


class CloudDataset(Dataset):
    def __init__(
        self,
        file_paths: list[str],
        augment: bool = False,
        train: bool = True,
        use_engineered_features: bool = True,
        use_raw_features: bool = True,
    ):
        """
        Initialize the CloudDataset, which loads the data from the given file paths and prepares it for training.

        :param file_paths: List of file paths to the data files.
        :param augment: Whether to apply data augmentation by transforming the image data.
        :param train: Whether the dataset is used for training. If False, the dataset is used for validation or testing.
        :param use_engineered_features: Whether to use the engineered features CORR, SD, and NDAI.
        :param use_raw_features: Whether to use the raw radiance angle features.
        """
        if not use_engineered_features and not use_raw_features:
            raise ValueError("At least one of use_engineered_features and use_raw_features must be True.")

        self.train = train
        self.num_images = len(file_paths)
        self.features = []
        self.labels = []

        for file_path in file_paths:
            # Load the data
            data = np.loadtxt(file_path)

            # Extract x and y coordinates
            x_coords = data[:, 1].astype(int)
            y_coords = data[:, 0].astype(int)
            min_x = np.min(x_coords)
            min_y = np.min(y_coords)

            # Shift coordinates to start from 0
            x_coords = x_coords - min_x
            y_coords = y_coords - min_y

            # Determine the size of the image based on x and y coordinates
            image_height = np.max(y_coords) + 1
            image_width = np.max(x_coords) + 1
            # Image height and width should be divisible by 16 for the UNet architecture
            image_height = int(np.ceil(image_height / 16) * 16)
            image_width = int(np.ceil(image_width / 16) * 16)

            # Extract features into a height x width x n_features Tensor
            # Some coordinates especially at the edges may not have data, so we initialize the features with zeros
            # x-coordinates, y-coordinates, and labels are not features, so we use don't include them
            feature_indices = []
            if use_engineered_features:
                feature_indices += [3, 4, 5]
            if use_raw_features:
                feature_indices += [6, 7, 8, 9, 10]
            features = np.zeros((image_height, image_width, len(feature_indices)))
            features[y_coords, x_coords] = data[:, feature_indices]

            # Extract labels into a height x width Tensor
            labels = np.zeros((image_height, image_width))
            labels[y_coords, x_coords] = data[:, 2]

            # Convert features and labels to torch tensors
            # We permute the dimensions to have the features in the order (channels, height, width), which is the format
            # expected by PyTorch for CNNs
            features = torch.tensor(features, dtype=torch.float32).permute(2, 0, 1)
            labels = torch.tensor(labels, dtype=torch.long)

            self.features.append(features)
            self.labels.append(labels)

        # Enable or disable augmentation
        self.augment = augment

    @staticmethod
    def _transform(features: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply the same transformation to the features and labels.

        :param features: The features to transform.
        :param labels: The labels to transform.
        :return: The transformed features and labels.
        """
        # Random horizontal flip
        if random.random() > 0.5:
            features = transforms.hflip(features)
            labels = transforms.hflip(labels)

        # Random vertical flip
        if random.random() > 0.5:
            features = transforms.vflip(features)
            labels = transforms.vflip(labels)

        # Random rotation
        if random.random() > 0.5:
            angle = random.uniform(-30, 30)
            features = transforms.rotate(features, angle)
            # Add one dimension to labels to make it a 3D tensor for rotation
            labels = labels.unsqueeze(0)
            labels = transforms.rotate(labels, angle)
            # Remove the added dimension
            labels = labels.squeeze(0)

        return features, labels

    def __len__(self):
        # If the dataset is used for training, we want to use the batch size defined above
        # The dataset is then filled with transformed versions of the data
        if self.train:
            return BATCH_SIZE
        else:
            return len(self.features)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # If the dataset is used for training, we want to use the batch size defined above
        # The dataset is then filled with transformed versions of the data
        if self.train:
            idx = idx % self.num_images

        features = self.features[idx]
        labels = self.labels[idx]

        # Apply augmentation if enabled
        if self.augment:
            features, labels = self._transform(features, labels)

        return features, labels
