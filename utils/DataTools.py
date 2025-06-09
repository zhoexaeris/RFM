"""Data loading and processing utilities for deep learning models.

This module provides tools for loading and processing image datasets, including
custom dataset classes and data transformation utilities. It supports various
image formats and includes functionality for data augmentation and preprocessing.
"""

from torchvision.datasets.vision import VisionDataset
from PIL import Image
import cv2
import numpy as np
import os
import random


class imgdataset(VisionDataset):
    """Custom image dataset class for loading and processing images.

    This class extends VisionDataset to provide functionality for loading and
    processing images from a list of directories, with support for data
    transformation and preprocessing.

    Args:
        rootlist (list): List of tuples containing (root_path, label) pairs.
        process (callable, optional): Additional processing function. Defaults to None.
        transform (callable, optional): Image transformation function. Defaults to None.
        randomdrop (int, optional): Random drop probability. Defaults to 0.
    """

    def __init__(self, rootlist, process=None, transform=None, randomdrop=0):
        """Initialize the image dataset.

        Args:
            rootlist (list): List of tuples containing (root_path, label) pairs.
            process (callable, optional): Additional processing function. Defaults to None.
            transform (callable, optional): Image transformation function. Defaults to None.
            randomdrop (int, optional): Random drop probability. Defaults to 0.
        """
        super(imgdataset, self).__init__(root="", transform=transform)
        self.rootlist = rootlist
        self.randomdrop = randomdrop
        self.dataset = []
        self.process = process
        for root, label in self.rootlist:
            imglist = os.listdir(root)
            print("Loading %s" % (root), end="\r")
            for p in imglist:
                self.dataset.append((os.path.join(root, p), label))
            print("Loaded %s=>%d" % (root, len(imglist)))

    def shuffle(self):
        """Randomly shuffle the dataset."""
        random.shuffle(self.dataset)

    def reset(self):
        """Reset the dataset to its initial state."""
        self.dataset = []
        for root, label in self.rootlist:
            imglist = os.listdir(root)
            for p in imglist:
                self.dataset.append((os.path.join(root, p), label))

    def __getitem__(self, index):
        """Get an item from the dataset.

        Args:
            index (int): Index of the item to get.

        Returns:
            tuple: A tuple containing:
                - PIL.Image: Transformed image
                - int: Image label
        """
        img, label = self.dataset[index]
        img = Image.open(img)
        img = np.array(img)
        img = cv2.resize(img, (256, 256))
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        """Get the length of the dataset.

        Returns:
            int: Number of items in the dataset.
        """
        return len(self.dataset)

    def __add__(self, other):
        """Combine two datasets.

        Args:
            other (imgdataset): Another dataset to combine with.

        Returns:
            imgdataset: Combined dataset.
        """
        self.dataset.extend(other.dataset)
        return self
