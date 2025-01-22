import os
import requests
import tarfile

import torch
import torchvision
import torchvision.transforms as transforms

from Params.params_datasets import *


class Dataset:
    """
    Load and configure a dataset. The valid values for the arguments can be found in Params/params_dataset.py.
    The first time a specific dataset is called, it is downloaded and saved locally.

    Args:
        name (str): Name of the dataset.
        split (str): Split to load ("train" or "test").
        network (str): Network that will be used to process the dataset.
        data_transforms (str or None): transform applied to images before going through the network.
        data_adv_attack (str or None): attack applied to images before going through the network.
        batch_size (int): Batch size used to process the dataset.

    Attributes (public):
        name (str): Name.
        split (str): Split.
        network (str): Network.
        data_transforms (str or None): Transform.
        data_adv_attack (str or None): Attack.
        batch_size (int): Batch size.
        dataloader (torch dataloader): Dataloader object used by the neural network to process the dataset
    """
    def __init__(self, dataset_name, dataset_split, network, data_transforms=None, data_adv_attack=None, batch_size=1000):
        """Initializes dataset."""

        # Public attributes
        self.name = dataset_name
        self.split = dataset_split
        self.network = network
        self.batch_size = batch_size
        self.data_transforms = data_transforms
        self.data_adv_attack = data_adv_attack



        # Create Data folder if required
        if not os.path.exists(path_to_saved_datasets):
            os.makedirs(path_to_saved_datasets)

        # Ensure dataset is valid
        self._check_accepted_datasets()
        self._check_accepted_networks()
        self._check_accepted_transforms()
        self._check_accepted_adv_attack()

        # Private attributes
        self._mean_transform = mean_transform[network]
        self._std_transform = std_transform[network]
        self._set_transforms()

        # Create dataloader
        self._load_dataset()
        self.dataloader = torch.utils.data.DataLoader(
            self.name, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=2
        )

    def _check_accepted_datasets(self):
        """Ensures that the queried dataset is valid."""
        data_split = self.name + "_" + self.split
        if data_split not in accepted_datasets:
            raise ValueError("Accepted dataset/split pairs are: %s" % str(accepted_datasets)[1:-1])

    def _check_accepted_networks(self):
        """Ensures that the queried neural network is valid."""
        accepted_networks = list(mean_transform.keys())
        if self.network not in accepted_networks:
            raise ValueError("Accepted network architectures are: %s" % str(accepted_networks)[1:-1])

    def _check_accepted_transforms(self):
        """Ensures that the queried transform is valid."""
        accepted_transforms = list(additional_transforms.keys()) + [None]
        if self.data_transforms not in accepted_transforms:
            raise ValueError("Accepted data transforms are: %s" % str(accepted_transforms)[1:-1])

    def _check_accepted_adv_attack(self):
        """Ensures that the queried adversarial attack is valid."""
        if self.data_adv_attack not in accepted_attacks + [None]:
            raise ValueError("Accepted attacks are: %s" % str(accepted_attacks)[1:-1])

    def _set_transforms(self):
        """Sets the transforms."""
        transform_list = [transforms.ToTensor(),
                          transforms.Normalize(self._mean_transform, self._std_transform)]
        if self.data_transforms is not None:
            transform_list.insert(0, additional_transforms[self.data_transforms])

        self._transform = transforms.Compose(transform_list)

    def _load_dataset(self):
        """Loads the dataset."""
        if self.name == "cifar10":
            self._load_cifar10()
        elif self.name == "cifar100":
            self._load_cifar100()
        elif self.name == "svhn":
            self._load_svhn()
        elif self.name == "tiny_imagenet":
            self._load_tinyimagenet()
        elif self.name == "lsun":
            self._load_lsun()
        else:
            pass

    def _load_cifar10(self):
        """Load CIFAR10."""
        is_train = (self.split == "train")
        self.dataset = torchvision.datasets.CIFAR10(
            root=path_to_saved_datasets, 
            train=is_train,
            download=True, 
            transform=self._transform
        )

    def _load_cifar100(self):
        """Load CIFAR100."""
        is_train = (self.split == "train")
        self.dataset = torchvision.datasets.CIFAR100(
            root=path_to_saved_datasets, 
            train=is_train,
            download=True, 
            transform=self._transform
        )

    def _load_svhn(self):
        """Load SVHN."""
        self.dataset = torchvision.datasets.SVHN(
            root=path_to_saved_datasets, 
            split=self.split,
            download=True, 
            transform=self._transform
        )

    def _load_tinyimagenet(self):
        """Load Tiny ImageNet."""
        if not os.path.exists(path_to_tinyImagenet):
            r = requests.get(url_tinyImagenet, allow_redirects=True)
            open(path_to_tinyImagenet[:-1] + ".tar.gz", 'wb').write(r.content)
            tar = tarfile.open(path_to_tinyImagenet[:-1] + ".tar.gz")
            tar.extractall(path=path_to_saved_datasets)
            tar.close()

        self.dataset = torchvision.datasets.ImageFolder(path_to_tinyImagenet, transform=self._transform)

    def _load_lsun(self):
        """Load LSUN."""
        if not os.path.exists(path_to_lsun):
            r = requests.get(url_lsun, allow_redirects=True)
            open(path_to_lsun[:-1] + ".tar.gz", 'wb').write(r.content)
            tar = tarfile.open(path_to_lsun[:-1] + ".tar.gz")
            tar.extractall(path=path_to_saved_datasets)
            tar.close()

        self.dataset = torchvision.datasets.ImageFolder(path_to_lsun, transform=self._transform)
