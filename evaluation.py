from dataset import Dataset
from Monitors import *
from evaluator import Evaluator
from feature_extractor import FeatureExtractor

import os
import csv
from sklearn.metrics import accuracy_score



all_models = ["resnet", "densenet"]
all_id_datasets = [
    "cifar10",
    "svhn",
    "cifar100",
]
all_ood_datasets = [
    ["cifar100", "svhn", "lsun"],
    ["cifar10", "tiny_imagenet", "lsun"],
    ["cifar10", "svhn", "lsun"],
]
all_perturbations = [
    "brightness",
    "blur",
    "pixelization",
]
all_adver_attacks = [
    "fgsm",
    "deepfool",
    "pgd",
]
all_monitors = [
    [MahalanobisMonitor, {}],
    [OTBMonitor, {}],
    [MSPMonitor, {}],
    [EnergyMonitor, {'T': 1}],
    [ReActMonitor, {'quantile_value': 0.99, 'mode':'MSP'}],
    [ReActMonitor, {'quantile_value': 0.99, 'mode':'energy'}],
]


def evaluate_one_monitor(
        model,
        id_dataset,
        ood_dataset,
        perturbation=None,
        adver_attack=None,
        eval_mode="oms",
        batch_size=10,
        layer_ids=None,
        device_name=None,
):
    """
    """
    dataset_train = Dataset(id_dataset, "train", model, batch_size=batch_size)
    dataset_test = Dataset(id_dataset, "test", model, batch_size=batch_size)
    dataset_ood = Dataset(ood_dataset, "test", model, perturbation, adver_attack, batch_size=batch_size)

    feature_extractor = FeatureExtractor(model, id_dataset, layer_ids, device_name)

    (features_train, logits_train, softmax_train, 
     preds_train, labels_train) = feature_extractor.get_features(dataset_train)
    (features_test, logits_test, softmax_test,
     preds_test, labels_test) = feature_extractor.get_features(dataset_test)
    (features_ood, logits_ood, softmax_ood,
     preds_ood, labels_ood) = feature_extractor.get_features(dataset_ood)
    
    evaluator = Evaluator(eval_mode, is_novelty=(id_dataset != ood_dataset))
    evaluator.fit_ground_truth(labels_test, labels_ood, preds_test, preds_ood)



def evaluate_all_monitors(
        model,
        layer_ids,
        batch_size,
        id_dataset,
        ood_dataset,
        eval_settings,
        perturbation=None,
        adver_attack=None,
        device_name=None,
):
    """
    """
    dataset_train = Dataset(id_dataset, "train", model, batch_size=batch_size)
    dataset_test = Dataset(id_dataset, "test", model, batch_size=batch_size)
    dataset_ood = Dataset(ood_dataset, "test", model, perturbation, adver_attack, batch_size=batch_size)

    feature_extractor = FeatureExtractor(model, id_dataset, layer_ids, device_name)

    (features_train, logits_train, softmax_train, 
     preds_train, labels_train) = feature_extractor.get_features(dataset_train)
    (features_test, logits_test, softmax_test,
     preds_test, labels_test) = feature_extractor.get_features(dataset_test)
    (features_ood, logits_ood, softmax_ood,
     preds_ood, labels_ood) = feature_extractor.get_features(dataset_ood)
    
    # Compute the model accuracy scores
    id_acc = accuracy_score(labels_test, preds_test)
    if id_dataset == ood_dataset:
        ood_acc = accuracy_score(labels_ood, preds_ood)
    else:
        ood_acc = 0

    # Define the OOD and OMS evaluators
    if 'oms' in eval_settings:
        eval_oms = Evaluator("oms", is_novelty=(id_dataset!=ood_dataset))
        eval_oms.fit_ground_truth(labels_test, labels_ood, preds_test, preds_ood)
    if 'ood' in eval_settings:
        eval_ood = Evaluator("ood", is_novelty=(id_dataset!=ood_dataset))
        eval_ood.fit_ground_truth(labels_test, labels_ood, preds_test, preds_ood)

    # Get the perfect precision, recall and f1-score
    prec_star, recall_star, f1_star = eval_oms.get_metrics()

    

def evaluate_all_scenarios(
        all_models,
        all_monitored_layers_ids,
        all_id_datasets,
        all_ood_datasets,
        all_perturbations,
        all_adver_attacks,
        all_eval_settings,
        all_monitors,
):
    """
    """
    for i in range(len(all_models)):
        model = all_models[i]
        layer_ids = all_monitored_layers_ids[i]

        for j in range(len(all_id_datasets)):
            id_dataset = all_id_datasets[j]

            for k in range(len(all_ood_datasets)):
                ood_dataset = all_ood_datasets[k]

                ## Evaluate the monitors with OOD as novelty

            for k in range(len(all_perturbations)):
                ood_dataset = id_dataset
                data_transforms = all_perturbations[k]

                ## Evaluate the monitors with OOD as cov shift

            for k in range(len(all_adver_attacks)):
                ood_dataset = id_dataset
                data_adv_attack = all_adver_attacks[k]

                ## Evaluate the monitors with OOD as adv attacks


