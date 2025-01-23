import torch

from dataset import Dataset
from feature_extractor import FeatureExtractor

batch_size = 10
device_name = 'cuda:0' if torch.cuda.is_available() else 'cpu'

all_models = ["resnet", "densenet"]

all_layers_ids = [[0, 5, 8, 12, 15, 19, 22, 26, 29, 32],
                  [0, 19, 29, 40, 50, 60, 69, 79, 89, 98]]

all_id_dataset = ["cifar10", "svhn", "cifar100"]

all_ood_dataset = [["cifar100", "svhn", "lsun"],
               ["cifar10", "tiny_imagenet", "lsun"],
               ["cifar10", "svhn", "lsun"]]

all_additional_transform = ["brightness", "blur", "pixelization"]

all_adversarial_attack = ["fgsm", "deepfool", "pgd"]

for i in range(len(all_models)):
    model = all_models[i]
    layers_ids = all_layers_ids[i]
    for j in range(len(all_id_dataset)):
        id_dataset = all_id_dataset[j]
        for k in range(len(all_ood_dataset[j])):
            ood_dataset = all_ood_dataset[j][k]

            print(model, id_dataset, ood_dataset)

            dataset_train = Dataset(id_dataset, "train", model, batch_size=batch_size)
            dataset_test = Dataset(id_dataset, "test", model, batch_size=batch_size)
            dataset_ood = Dataset(ood_dataset, "test", model, None, None, batch_size=batch_size)

            feature_extractor = FeatureExtractor(model, id_dataset, layers_ids, device_name)

            _, _, _, _, _ = feature_extractor.get_features(dataset_train)
            _, _, _, _, _ = feature_extractor.get_features(dataset_test)
            _, _, _, _, _ = feature_extractor.get_features(dataset_ood)

        for k in range(len(all_additional_transform)):
            additional_transform = all_additional_transform[k]

            print(model, id_dataset, additional_transform)

            dataset_train = Dataset(id_dataset, "train", model, batch_size=batch_size)
            dataset_test = Dataset(id_dataset, "test", model, batch_size=batch_size)
            dataset_ood = Dataset(id_dataset, "test", model, additional_transform, None, batch_size=batch_size)

            feature_extractor = FeatureExtractor(model, id_dataset, layers_ids, device_name)

            _, _, _, _, _ = feature_extractor.get_features(dataset_train)
            _, _, _, _, _ = feature_extractor.get_features(dataset_test)
            _, _, _, _, _ = feature_extractor.get_features(dataset_ood)

        for k in range(len(all_adversarial_attack)):
            adversarial_attack = all_adversarial_attack[k]

            print(model, id_dataset, adversarial_attack)

            dataset_train = Dataset(id_dataset, "train", model, batch_size=batch_size)
            dataset_test = Dataset(id_dataset, "test", model, batch_size=batch_size)
            dataset_ood = Dataset(id_dataset, "test", model, None, adversarial_attack, batch_size=batch_size)

            feature_extractor = FeatureExtractor(model, id_dataset, layers_ids, device_name)

            _, _, _, _, _ = feature_extractor.get_features(dataset_train)
            _, _, _, _, _ = feature_extractor.get_features(dataset_test)
            _, _, _, _, _ = feature_extractor.get_features(dataset_ood)