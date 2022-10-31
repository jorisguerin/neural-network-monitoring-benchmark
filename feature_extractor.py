import os

from tqdm import tqdm
import h5py
from Utils.utils_nn import download_file_from_google_drive

import numpy as np
from scipy.special import softmax
import torch
from torchvision.models.feature_extraction import create_feature_extractor
import torchattacks

import models

from Params.params_network import *
from Params.params_dataset import *

# import warnings
# warnings.filterwarnings("ignore")


class FeatureExtractor:
    """
    Load and configure a neural network classifier.
    The valid values for the arguments can be found in Params/params_network.py.
    The first time a specific neural network is called, it is downloaded and saved locally.

    Args:
       network (str): Name of the network.
       id_dataset (str): Name of the dataset used to train the network.
       layers_ids (List[int]): List of layers IDs that we wish to extract.
       device_name (str): The device to use to run the neural network inference.

    Attributes (public):
       network (str): Network.
       id_dataset (str): Training dataset.
       layers_id (List[int]): IDs of extracted layers.
       n_classes_id (int): Number of output classes
       layers (List[str]): names of extracted layers.
       linear_weights (np.array): Values of the linear weights (last layer).
       linear_bias (np.array): Values of the linear biases (last layer).
       model (pytorch model): The model itself.

    Methods (public):
        get_features: Extract features from desired layers.
    """
    def __init__(self, network, id_dataset, layers_ids, device_name=None):
        """Initializes dataset."""

        # Public attributes
        self.network = network
        self.id_dataset = id_dataset
        self.layers_id = layers_ids
        self.layers_id.sort()
        self.n_classes_id = n_classes_dataset[id_dataset]

        self._check_accepted_network()
        self.layers = layers[network]

        self._check_accepted_id_dataset()
        self._check_accepted_layers_id()

        if device_name is None:
            device_name = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.device_name = device_name
        self._device = torch.device(self.device_name)

        self.linear_weights, self.linear_bias = None, None

        if not os.path.exists(models_path):
            os.makedirs(models_path)
        self._model_dataset_name = self.network + "_" + self.id_dataset
        self._load_model()
        self.model.eval()
        self.model.to(self._device)

    def get_features(self, dataset, save=True):
        """
        Extract features from desired layers

        Args:
            dataset (Dataset): The dataset from which features are extracted. It can differ from the training dataset.
            save (bool): If True, the features are saved and will be loaded instead of extracted in the future.

        Returns:
            features, logits, softmax_values, predictions, labels (np.arrays)
        """
        if not os.path.exists(save_features_path):
            os.makedirs(save_features_path)

        predictions = None
        logits = None
        softmax_values = None
        labels = None

        layers_names = [list(self.layers.items())[i][0] for i in self.layers_id]
        to_extract, file_names_to_extract = [], []
        features = [[] for _ in range(len(self.layers_id))]
        for i, l in enumerate(layers_names):
            perturbations = ""
            if dataset.additional_transform is not None and dataset.adversarial_attack is not None:
                perturbations += "_" + dataset.additional_transform + "_" + dataset.adversarial_attack
            elif dataset.additional_transform is not None:
                perturbations += "_" + dataset.additional_transform
            elif dataset.adversarial_attack is not None:
                perturbations += "_" + dataset.adversarial_attack

            file_name = save_features_path + "%s_%s%s__%s_%s_%s.h5" % (dataset.name, dataset.split,
                                                                       perturbations,
                                                                       dataset.network, self.id_dataset, l)
            if os.path.exists(file_name):
                features[i], logits, softmax_values, predictions, labels = self._load_features(file_name)
            else:
                to_extract.append(i)
                file_names_to_extract.append(file_name)

        torch.cuda.empty_cache()
        if len(to_extract) != 0:
            if dataset.adversarial_attack is None:
                features_extracted, logits, softmax_values, predictions, labels = self._extract_features(dataset,
                                                                                                         to_extract,
                                                                                                         layers_names)
            else:
                features_extracted, logits, \
                    softmax_values, predictions, labels = self._extract_features_adv(dataset, to_extract, layers_names)
            for i in range(len(layers_names)):
                if i in to_extract:
                    features[i] = features_extracted[0]
                    if save:
                        self._save_features(features[i], logits, softmax_values,
                                            predictions, labels, file_names_to_extract[0])
                    features_extracted.pop(0)
                    file_names_to_extract.pop(0)

        return features, logits, softmax_values, predictions, labels

    def _extract_features(self, dataset, to_extract, layers_names):
        """
        Extracts features that have not been saved before, when no adversarial attack is applied.

        Args:
            dataset (Dataset): The dataset from which features are extracted. It can differ from the training dataset.
            to_extract (List[int]): IDs of layers to extract that were not saved.
            layers_names (List[str]): names of layers to extract that were not saved.

        Returns: features, all_logits, all_softmax, predicted_classes, all_labels
        """

        print('Extracting layers: %s' % str([layers_names[i] for i in to_extract])[1:-1])
        features = [[] for _ in range(len(to_extract))]
        predicted_classes = []
        all_labels = []
        all_logits = []
        all_softmax = []

        layers_selected = dict([list(self.layers.items())[self.layers_id[i]] for i in to_extract])
        layers_refs = [list(self.layers.items())[self.layers_id[i]][1] for i in to_extract]

        feature_extractor = create_feature_extractor(self.model, return_nodes=layers_selected)

        with torch.no_grad():
            for data in tqdm(dataset.dataloader):
                images, labels = data[0].to(self._device), data[1].to(self._device)
                outputs = feature_extractor(images)

                for i, l in enumerate(layers_refs):
                    features[i].append(torch.mean(outputs[l], (2, 3)))

                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                predicted_classes.append(predicted)
                all_labels.append(labels)
                all_logits.append(outputs.data)
                all_softmax.append(softmax(outputs.data.cpu().detach().numpy(), axis=1))

        for i in range(len(features)):
            features[i] = torch.cat(features[i], dim=0)
            features[i] = features[i].cpu().detach().numpy()

        predicted_classes = torch.cat(predicted_classes)
        predicted_classes = predicted_classes.cpu().detach().numpy()
        all_labels = torch.cat(all_labels)
        all_labels = all_labels.cpu().detach().numpy()
        all_logits = torch.cat(all_logits)
        all_logits = all_logits.cpu().detach().numpy()
        all_softmax = np.concatenate(all_softmax)

        return features, all_logits, all_softmax, predicted_classes, all_labels

    def _extract_features_adv(self, dataset, to_extract, layers_names):
        """
        Extracts features that have not been saved before, when an adversarial attack is applied.

        Args:
            dataset (Dataset): The dataset from which features are extracted. It can differ from the training dataset.
            to_extract (List[int]): IDs of layers to extract that were not saved.
            layers_names (List[str]): names of layers to extract that were not saved.

        Returns: features, all_logits, all_softmax, predicted_classes, all_labels
        """
        print('Extracting layers: %s' % str([layers_names[i] for i in to_extract])[1:-1])

        attacker = AdversarialAttack(dataset.adversarial_attack, self.model)

        features = [[] for _ in range(len(to_extract))]
        predicted_classes = []
        all_labels = []
        all_logits = []
        all_softmax = []

        layers_selected = dict([list(self.layers.items())[self.layers_id[i]] for i in to_extract])
        layers_refs = [list(self.layers.items())[self.layers_id[i]][1] for i in to_extract]

        feature_extractor = create_feature_extractor(self.model, return_nodes=layers_selected)

        for data in tqdm(dataset.dataloader):
            images, labels = data[0].to(self._device), data[1].to(self._device)
            images = attacker.run(images, labels).to(self._device)
            outputs = feature_extractor(images)

            for i, l in enumerate(layers_refs):
                features[i].append(torch.mean(outputs[l], (2, 3)).cpu().detach().numpy())

            outputs = self.model(images)
            _, predicted = torch.max(outputs.data, 1)
            predicted_classes.append(predicted)
            all_labels.append(labels)
            all_logits.append(outputs.data)
            all_softmax.append(softmax(outputs.data.cpu().detach().numpy(), axis=1))

        for i in range(len(features)):
            features[i] = np.concatenate(features[i], axis=0)

        predicted_classes = torch.cat(predicted_classes)
        predicted_classes = predicted_classes.cpu().detach().numpy()
        all_labels = torch.cat(all_labels)
        all_labels = all_labels.cpu().detach().numpy()
        all_logits = torch.cat(all_logits)
        all_logits = all_logits.cpu().detach().numpy()
        all_softmax = np.concatenate(all_softmax)

        return features, all_logits, all_softmax, predicted_classes, all_labels

    @staticmethod
    def _save_features(features, logits, softmax_values, predictions, labels, file_name):
        """Saves extracted features to file."""
        hf = h5py.File(file_name, 'w')
        hf.create_dataset('features', data=features)
        hf.create_dataset('logits', data=logits)
        hf.create_dataset('softmax', data=softmax_values)
        hf.create_dataset('predictions', data=predictions)
        hf.create_dataset('labels', data=labels)
        hf.close()

    @staticmethod
    def _load_features(file_name):
        """Loads features from file."""
        hf = h5py.File(file_name, 'r')
        features = np.array(hf.get("features"))
        logits = np.array(hf.get("logits"))
        softmax_values = np.array(hf.get("softmax"))
        predictions = np.array(hf.get("predictions"))
        labels = np.array(hf.get("labels"))
        hf.close()

        return features, logits, softmax_values, predictions, labels

    def _check_accepted_network(self):
        """Verifies that the network is handled by the library."""
        accepted_networks = list(mean_transform.keys())
        if self.network not in accepted_networks:
            raise ValueError("Accepted network architectures are: %s" % str(accepted_networks)[1:-1])

    def _check_accepted_id_dataset(self):
        """Verifies that the network/training dataset pair is handled by the library."""
        accepted_dataset = list(n_classes_dataset.keys())
        if self.id_dataset not in accepted_dataset:
            raise ValueError("Accepted ID datasets are: %s" % str(accepted_dataset)[1:-1])

    def _check_accepted_layers_id(self):
        """Verifies that the layer IDs exist for the network."""
        n_layers = len(list(self.layers.items()))
        if self.layers_id[0] < 0:
            raise ValueError("All layers IDs must be >= 0")
        elif self.layers_id[-1] > n_layers:
            raise ValueError("All layers IDs must be <= %i" % n_layers)

    def _load_model(self):
        """Loads the network."""
        if not os.path.exists(models_path + self._model_dataset_name + ".pth"):
            gd_id = models_gdrive_ids[self._model_dataset_name]
            destination = models_path + self._model_dataset_name + ".pth"
            download_file_from_google_drive(gd_id, destination)

        if self.network == "resnet":
            self._load_resnet()
        elif self.network == "densenet":
            self._load_densenet()
        else:
            pass

    def _load_resnet(self):
        """Loads ResNet models."""
        self.model = models.ResNet34(num_c=self.n_classes_id)
        self.model.load_state_dict(torch.load(models_path + self._model_dataset_name + ".pth",
                                              map_location=self.device_name))
        self.linear_weights = self.model.linear.weight.cpu().detach().numpy()
        self.linear_bias = self.model.linear.bias.cpu().detach().numpy()

    def _load_densenet(self):
        """Loads DenseNet models."""
        self.model = models.DenseNet3(100, self.n_classes_id)
        self.model.load_state_dict(torch.load(models_path + self._model_dataset_name + ".pth",
                                              map_location=self.device_name))

        self.linear_weights = self.model.fc.weight.cpu().detach().numpy()
        self.linear_bias = self.model.fc.bias.cpu().detach().numpy()


class AdversarialAttack:
    """
    Load and configure an adversarial attack.
    The valid values for adversarial attack can be found in Params/params_dataset.py.

    Args:
       attack_type (str): Name of the attack.
       model (pytorch model): Model to attack.

    Attributes (public):
       attack_type (str): Name of the attack.
       model (pytorch model): Model to attack.
       attack (torchattacks Attack): The actual attack.

    Methods (public):
        run: Attacks the model on a batch of images.
    """
    def __init__(self, attack_type, model):
        """Initializes the attack."""
        self.attack_type = attack_type
        self.model = model
        self._load_attack()

    def run(self, images, labels):
        """
        Attacks the model on a batch of images.

        Args:
            images (np.array): Batch of images on which the attack will be applied.
            labels (np.array): Labels corresponding to the images.

        Returns: the attacked images.
        """
        adv_images = self.attack(images, labels)

        return adv_images

    def _load_attack(self):
        """Configures and loads the attack"""
        if self.attack_type == "fgsm":
            self.attack = torchattacks.FGSM(self.model, eps=0.007)
        if self.attack_type == "deepfool":
            self.attack = torchattacks.DeepFool(self.model, steps=10, overshoot=0.02)
        if self.attack_type == "pgd":
            self.attack = torchattacks.PGD(self.model, eps=8/255, alpha=1/255, steps=40, random_start=True)
