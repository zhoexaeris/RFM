"""Custom dataset implementation for deep learning model training.

This module provides a custom dataset class for handling real and fake image datasets,
with support for different data splits (train, validation, test) and data augmentation
transforms. It includes functionality to load and process images with appropriate
transformations for training and testing.
"""

import torchvision
import os
import utils.DataTools as dt
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import cv2
from torchvision import transforms
import torchvision.transforms.functional as TF
from functools import lru_cache

# Keep the same augmentation transforms as the original
aug_train = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    #torchvision.transforms.RandomCrop(224),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        [0.5, 0.5, 0.5], [0.25, 0.25, 0.25])
])

aug_test = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    #torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        [0.5, 0.5, 0.5], [0.25, 0.25, 0.25])
])

# Cache for loaded images
@lru_cache(maxsize=1000)
def load_image(image_path):
    """Load and cache an image efficiently.
    
    Args:
        image_path (str): Path to the image file.
        
    Returns:
        PIL.Image: Loaded image.
    """
    try:
        img = Image.open(image_path)
        img = np.array(img)
        img = cv2.resize(img, (256, 256))
        return Image.fromarray(img)
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

class CustomDataset(Dataset):
    """Custom dataset class for handling real and fake image datasets.

    This class provides methods to load and process real and fake images from
    different dataset splits (train, validation, test) with appropriate data
    augmentation transforms.

    Args:
        folder_path (str): Root path to the dataset directory.
    """

    def __init__(self, folder_path="./dataset_root"):
        """Initialize the custom dataset.

        Args:
            folder_path (str): Root path to the dataset directory.
        """
        self.folder_path = folder_path
        self.R_dir = ["real"]  # Real images directory name
        self.F_dir = ["fake"]  # Fake images directory name
        
        # Define path functions for each split
        self.trainpath = lambda path, file: os.path.join(self.folder_path, "train", file)
        self.validpath = lambda path, file: os.path.join(self.folder_path, "val", file)
        self.testpath = lambda path, file: os.path.join(self.folder_path, "test", file)

        # Cache for dataset paths
        self._path_cache = {}

    def _get_paths(self, base_path, label):
        """Get and cache paths for a dataset split.
        
        Args:
            base_path (str): Base path for the dataset split.
            label (str): Label directory name.
            
        Returns:
            list: List of (image_path, label) tuples.
        """
        cache_key = f"{base_path}_{label}"
        if cache_key not in self._path_cache:
            path = os.path.join(base_path, label)
            if not os.path.exists(path):
                return []
            
            paths = []
            for img_name in os.listdir(path):
                img_path = os.path.join(path, img_name)
                if os.path.isfile(img_path):
                    paths.append((img_path, 1 if label == self.F_dir else 0))
            
            self._path_cache[cache_key] = paths
        return self._path_cache[cache_key]

    def getDatasets(self, base_path, dataset_configs, transform, process=None, datasetfunc=None):
        """Get dataset for a specific configuration.
        
        Args:
            base_path (str): Base path for the dataset.
            dataset_configs (list): List of dataset configurations.
            transform (callable): Transform to apply to images.
            process (callable, optional): Additional processing function.
            datasetfunc (callable, optional): Custom dataset creation function.
            
        Returns:
            torch.utils.data.Dataset: Dataset for the specified configuration.
        """
        if datasetfunc is not None:
            return datasetfunc(base_path, dataset_configs, transform, process)
            
        paths = []
        for config in dataset_configs:
            name, label, _ = config
            paths.extend(self._get_paths(base_path, label))
            
        return ImageDataset(paths, transform=transform, process=process)

    def getsetlist(self, real, setType, process=None, datasetfunc=None):
        """Get a list of datasets for real or fake images.

        Args:
            real (bool): If True, get real image datasets; if False, get fake image datasets.
            setType (int): Type of dataset split (0: train, 1: validation, 2: test).
            process (callable, optional): Additional processing function. Defaults to None.
            datasetfunc (callable, optional): Custom dataset creation function. Defaults to None.

        Returns:
            tuple: A tuple containing:
                - list: List of datasets
                - list: List of directory names
        """
        setdir = self.R_dir if real is True else self.F_dir
        label = 0 if real is True else 1
        aug = aug_train if setType == 0 else aug_test
        pathfunc = self.trainpath if setType == 0 else self.validpath if setType == 1 else self.testpath
        setlist = []
        for setname in setdir:
            datalist = [(pathfunc(self.folder_path, setname), label)]
            if datasetfunc is not None:
                tmptestset = datasetfunc(datalist, transform=aug, process=process)
            else:
                tmptestset = dt.imgdataset(datalist, transform=aug, process=process)
            setlist.append(tmptestset)
        return setlist, setdir

    def getTrainsetR(self, process=None, datasetfunc=None):
        """Get training dataset for real images.

        Args:
            process (callable, optional): Additional processing function. Defaults to None.
            datasetfunc (callable, optional): Custom dataset creation function. Defaults to None.

        Returns:
            torch.utils.data.Dataset: Training dataset for real images.
        """
        return self.getDatasets(self.trainpath, [[self.__class__.__name__+" TrainsetR", self.R_dir, 0]], aug_train, process=process, datasetfunc=datasetfunc)

    def getTrainsetF(self, process=None, datasetfunc=None):
        """Get training dataset for fake images.

        Args:
            process (callable, optional): Additional processing function. Defaults to None.
            datasetfunc (callable, optional): Custom dataset creation function. Defaults to None.

        Returns:
            torch.utils.data.Dataset: Training dataset for fake images.
        """
        return self.getDatasets(self.trainpath, [[self.__class__.__name__+" TrainsetF", self.F_dir, 1]], aug_train, process=process, datasetfunc=datasetfunc)

    def getTrainset(self, process=None, datasetfunc=None):
        """Get combined training dataset for real and fake images.

        Args:
            process (callable, optional): Additional processing function. Defaults to None.
            datasetfunc (callable, optional): Custom dataset creation function. Defaults to None.

        Returns:
            torch.utils.data.Dataset: Combined training dataset.
        """
        return self.getDatasets(self.trainpath, [[self.__class__.__name__+" TrainsetR", self.R_dir, 0], [self.__class__.__name__+" TrainsetF", self.F_dir, 1]], aug_train, process=process, datasetfunc=datasetfunc)

    def getValidsetR(self, process=None, datasetfunc=None):
        """Get validation dataset for real images.

        Args:
            process (callable, optional): Additional processing function. Defaults to None.
            datasetfunc (callable, optional): Custom dataset creation function. Defaults to None.

        Returns:
            torch.utils.data.Dataset: Validation dataset for real images.
        """
        return self.getDatasets(self.validpath, [[self.__class__.__name__+" ValidsetR", self.R_dir, 0]], aug_test, process=process, datasetfunc=datasetfunc)

    def getValidsetF(self, process=None, datasetfunc=None):
        """Get validation dataset for fake images.

        Args:
            process (callable, optional): Additional processing function. Defaults to None.
            datasetfunc (callable, optional): Custom dataset creation function. Defaults to None.

        Returns:
            torch.utils.data.Dataset: Validation dataset for fake images.
        """
        return self.getDatasets(self.validpath, [[self.__class__.__name__+" ValidsetF", self.F_dir, 1]], aug_test, process=process, datasetfunc=datasetfunc)

    def getValidset(self, process=None, datasetfunc=None):
        """Get combined validation dataset for real and fake images.

        Args:
            process (callable, optional): Additional processing function. Defaults to None.
            datasetfunc (callable, optional): Custom dataset creation function. Defaults to None.

        Returns:
            torch.utils.data.Dataset: Combined validation dataset.
        """
        return self.getDatasets(self.validpath, [[self.__class__.__name__+" ValidsetR", self.R_dir, 0], [self.__class__.__name__+" ValidsetF", self.F_dir, 1]], aug_test, process=process, datasetfunc=datasetfunc)

    def getTestsetR(self, process=None, datasetfunc=None):
        """Get test dataset for real images.

        Args:
            process (callable, optional): Additional processing function. Defaults to None.
            datasetfunc (callable, optional): Custom dataset creation function. Defaults to None.

        Returns:
            torch.utils.data.Dataset: Test dataset for real images.
        """
        return self.getDatasets(self.testpath, [[self.__class__.__name__+" TestsetR", self.R_dir, 0]], aug_test, process=process, datasetfunc=datasetfunc)

    def getTestsetF(self, process=None, datasetfunc=None):
        """Get test dataset for fake images.

        Args:
            process (callable, optional): Additional processing function. Defaults to None.
            datasetfunc (callable, optional): Custom dataset creation function. Defaults to None.

        Returns:
            torch.utils.data.Dataset: Test dataset for fake images.
        """
        return self.getDatasets(self.testpath, [[self.__class__.__name__+" TestsetF", self.F_dir, 1]], aug_test, process=process, datasetfunc=datasetfunc)

    def getTestset(self, process=None, datasetfunc=None):
        """Get combined test dataset for real and fake images.

        Args:
            process (callable, optional): Additional processing function. Defaults to None.
            datasetfunc (callable, optional): Custom dataset creation function. Defaults to None.

        Returns:
            torch.utils.data.Dataset: Combined test dataset.
        """
        return self.getDatasets(self.testpath, [[self.__class__.__name__+" TestsetR", self.R_dir, 0], [self.__class__.__name__+" TestsetF", self.F_dir, 1]], aug_test, process=process, datasetfunc=datasetfunc)

class ImageDataset(Dataset):
    def __init__(self, paths, transform=None, process=None):
        """Initialize the image dataset.
        
        Args:
            paths (list): List of (image_path, label) tuples.
            transform (callable, optional): Transform to apply to images.
            process (callable, optional): Additional processing function.
        """
        self.paths = paths
        self.transform = transform
        self.process = process
        
    def __len__(self):
        return len(self.paths)
        
    def __getitem__(self, idx):
        img_path, label = self.paths[idx]
        
        # Use cached image loading
        img = load_image(img_path)
        if img is None:
            # Return a black image if loading fails
            img = Image.new('RGB', (256, 256))
            
        if self.transform:
            img = self.transform(img)
            
        if self.process:
            img = self.process(img)
            
        return img, label 