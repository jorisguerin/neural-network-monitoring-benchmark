"""
Script to generate the results seen in the paper

"""

import csv
import os
import sys
import torch
from sklearn.metrics import accuracy_score

## Adjuste PATH variable to launch script from project root ##
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
## -------------------------------------------------------- ##

from dataset import Dataset
from feature_extractor import FeatureExtractor
from evaluator import Evaluator
from Monitors import (
    EnergyMonitor,
    MSPMonitor,
    OTBMonitor,
    ReActMonitor,
    MahalanobisMonitor,
)

import warnings
warnings.filterwarnings('ignore')


batch_size = 10
TORCH_DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

all_networks = ["resnet", "densenet"]
all_networks_layers = [[0, 32], 
                       [0, 98]]
all_datasets = ["cifar10", "svhn", "cifar100"]
all_datasets_ood = [["cifar100", "svhn", "lsun"],
                    ["cifar10", "tiny_imagenet", "lsun"],
                    ["cifar10", "svhn", "lsun"]]
all_perturbations = ["brightness", "blur", "pixelization"]
all_adver_attacks = ["fgsm", "deepfool", "pgd"]

path_to_save_results = "Results/base_study/"
path_to_results_file = path_to_save_results + "full_results_v1.csv"
if not os.path.exists(path_to_save_results):
    os.makedirs(path_to_save_results)

f = open(path_to_results_file, "w", encoding="UTF8")
writer = csv.writer(f)
header = [
    "Network", "Network Layer",
    "Dataset", "Dataset OOD", "Perturbation", "Attack",
    "Precision OMS@OOD*", "Recall OMS@OOD*", "F1 OMS@OOD*"
    "Monitor",
    "Precision OOD", "Recall OOD", "F1 OOD"
    "Precision OMS", "Recall OMS", "F1 OMS"
]
writer.writerow(header)


def evaluate_monitors(
        network,
        network_layers,
        dataset_ID,
        dataset_OOD,
        perturbation=None,
        adver_attack=None,
):
    dataset_train = Dataset(dataset_ID, "train", network, batch_size=batch_size)
    dataset_test = Dataset(dataset_ID, "test", network, batch_size=batch_size)
    dataset_ood = Dataset(dataset_OOD, "test", network, perturbation, adver_attack, batch_size=batch_size)

    feature_extractor = FeatureExtractor(network, dataset_ID, network_layers, TORCH_DEVICE)

    deep_features_train = feature_extractor.get_features(dataset_train)
    deep_features_test = feature_extractor.get_features(dataset_test)
    deep_features_ood = feature_extractor.get_features(dataset_ood)

    features_train, logits_train, softmax_train, \
        pred_train, lab_train = deep_features_train
    features_test, logits_test, softmax_test, \
        pred_test, lab_test = deep_features_test
    features_ood, logits_ood, softmax_ood, \
        pred_ood, lab_ood = deep_features_ood

    accuracy_id = accuracy_score(lab_test, pred_test)
    accuracy_ood = 0
    if dataset_ID == dataset_OOD:
        accuracy_ood = accuracy_score(lab_ood, pred_ood) 

    eval_oms = Evaluator("oms", is_novelty=(dataset_ID!=dataset_OOD))
    eval_ood = Evaluator("ood", is_novelty=(dataset_ID!=dataset_OOD))
    eval_oms.fit_ground_truth(lab_test, lab_ood, pred_test, pred_ood)
    eval_ood.fit_ground_truth(lab_test, lab_ood, pred_test, pred_ood)

    prec_star, recall_star, f1_star = eval_oms.get_metrics_at_f1_opt(
        eval_ood.y_true[:lab_test.shape[0]].astype(bool),
        eval_ood.y_true[lab_test.shape[0]:].astype(bool) 
    )

    # # Evaluate MSP
    # metrics = _evaluate_MSP(
    #     deep_features_train,
    #     deep_features_test,
    #     deep_features_ood,
    #     eval_oms, eval_ood,
    #     monitor_args=[],
    #     monitor_kwargs={})
    # results = [network, 1,
    #     dataset_ID, dataset_OOD, str(perturbation), str(adver_attack),
    #     accuracy_id, accuracy_ood,
    #     prec_star, recall_star, f1_star,
    #     "MSP",
    #     metrics[0], metrics[1], metrics[2],
    #     metrics[3], metrics[4], metrics[5]]
    # writer.writerow(results)

    # Evaluate MSP
    monitor = MSPMonitor()
    monitor.fit()

    scores_test = monitor.predict(softmax_test)
    scores_ood  = monitor.predict(softmax_ood)

    prec_ood, recall_ood, f1_ood = eval_ood.get_metrics_at_f1_opt(scores_test, scores_ood)
    prec_oms, recall_oms, f1_oms = eval_oms.get_metrics_at_f1_opt(scores_test, scores_ood)

    result = [
        network, 1,
        dataset_ID, dataset_OOD, str(perturbation), str(adver_attack),
        accuracy_id, accuracy_ood,
        prec_star, recall_star, f1_star,
        "MSP",
        prec_ood, recall_ood, f1_ood,
        prec_oms, recall_oms, f1_oms]
    writer.writerow(result)
    
    # Evaluate Ene
    monitor = EnergyMonitor(T=1)
    monitor.fit()

    scores_test = monitor.predict(logits_test)
    scores_ood  = monitor.predict(logits_ood)

    prec_ood, recall_ood, f1_ood = eval_ood.get_metrics_at_f1_opt(scores_test, scores_ood)
    prec_oms, recall_oms, f1_oms = eval_oms.get_metrics_at_f1_opt(scores_test, scores_ood)

    result = [
        network, 1,
        dataset_ID, dataset_OOD, str(perturbation), str(adver_attack),
        accuracy_id, accuracy_ood,
        prec_star, recall_star, f1_star,
        "Ene",
        prec_ood, recall_ood, f1_ood,
        prec_oms, recall_oms, f1_oms]
    writer.writerow(result)

    # Evaluate Re-MSP
    monitor = ReActMonitor(quantile_value=0.99, mode="MSP")
    monitor.fit(feature_extractor, features_train[-1])

    scores_test = monitor.predict(features_test[-1])
    scores_ood  = monitor.predict(features_ood[-1])

    prec_ood, recall_ood, f1_ood = eval_ood.get_metrics_at_f1_opt(scores_test, scores_ood)
    prec_oms, recall_oms, f1_oms = eval_oms.get_metrics_at_f1_opt(scores_test, scores_ood)

    result = [
        network, 1,
        dataset_ID, dataset_OOD, str(perturbation), str(adver_attack),
        accuracy_id, accuracy_ood,
        prec_star, recall_star, f1_star,
        "Re-MSP",
        prec_ood, recall_ood, f1_ood,
        prec_oms, recall_oms, f1_oms]
    writer.writerow(result)

    # Evaluate Re-Ene
    monitor = ReActMonitor(quantile_value=0.99, mode="energy")
    monitor.fit(feature_extractor, features_train[-1])

    scores_test = monitor.predict(features_test[-1])
    scores_ood  = monitor.predict(features_ood[-1])

    prec_ood, recall_ood, f1_ood = eval_ood.get_metrics_at_f1_opt(scores_test, scores_ood)
    prec_oms, recall_oms, f1_oms = eval_oms.get_metrics_at_f1_opt(scores_test, scores_ood)

    result = [
        network, 1,
        dataset_ID, dataset_OOD, str(perturbation), str(adver_attack),
        accuracy_id, accuracy_ood,
        prec_star, recall_star, f1_star,
        "Re-Ene",
        prec_ood, recall_ood, f1_ood,
        prec_oms, recall_oms, f1_oms]
    writer.writerow(result)

    # Evaluate OTB
    for i_layer in range(len(network_layers)):
        monitor = OTBMonitor(n_clusters=10)
        monitor.fit(features_train[i_layer], pred_train, lab_train, save=False)

        scores_test = monitor.predict(features_test[i_layer], pred_test)
        scores_ood  = monitor.predict(features_ood[i_layer], pred_ood)

        prec_ood, recall_ood, f1_ood = eval_ood.get_metrics_at_f1_opt(scores_test, scores_ood)
        prec_oms, recall_oms, f1_oms = eval_oms.get_metrics_at_f1_opt(scores_test, scores_ood)

        result = [
            network, i_layer,
            dataset_ID, dataset_OOD, str(perturbation), str(adver_attack),
            accuracy_id, accuracy_ood,
            prec_star, recall_star, f1_star,
            "OTB",
            prec_ood, recall_ood, f1_ood,
            prec_oms, recall_oms, f1_oms]
        writer.writerow(result)

    # Evaluate Maha
    for i_layer in range(len(network_layers)):
        monitor = MahalanobisMonitor(dataset_ID, network, i_layer, is_tied=True)
        monitor.fit(features_train[i_layer], pred_train, lab_train, save=False)

        scores_test = monitor.predict(features_test[i_layer], pred_test)
        scores_ood  = monitor.predict(features_ood[i_layer], pred_ood)

        prec_ood, recall_ood, f1_ood = eval_ood.get_metrics_at_f1_opt(scores_test, scores_ood)
        prec_oms, recall_oms, f1_oms = eval_oms.get_metrics_at_f1_opt(scores_test, scores_ood)

        result = [
            network, i_layer,
            dataset_ID, dataset_OOD, str(perturbation), str(adver_attack),
            accuracy_id, accuracy_ood,
            prec_star, recall_star, f1_star,
            "OTB",
            prec_ood, recall_ood, f1_ood,
            prec_oms, recall_oms, f1_oms]
        writer.writerow(result)


# def _evaluate_MSP(
#         deep_features_train,
#         deep_features_test,
#         deep_features_ood,
#         eval_oms=None, 
#         eval_ood=None,
#         monitor_args=[],
#         monitor_kwargs={}
# ):
#     monitor = MSPMonitor()
#     monitor.fit()

#     scores_test = monitor.predict(deep_features_test[2])
#     scores_ood  = monitor.predict(deep_features_ood[2])

#     prec_oms, recall_oms, f1_oms = eval_oms.get_metrics_at_f1_opt(scores_test, scores_ood)
#     prec_ood, recall_ood, f1_ood = eval_ood.get_metrics_at_f1_opt(scores_test, scores_ood)
    
#     return (prec_ood, recall_ood, f1_ood,
#             prec_oms, recall_oms, f1_oms)


for i_network in range(len(all_networks)):
    network = all_networks[i_network]
    network_layers = all_networks_layers[i_network]

    for i_dataset in range(len(all_datasets)):
        dataset = all_datasets[i_dataset]
        
        # Test with OOD as novelty
        for j_dataset in range(len(all_datasets_ood)):
            dataset_ood = all_datasets_ood[i_dataset][j_dataset]

            print("Evaluating %s, for dataset %s and OOD dataset %s." % (network, dataset, (dataset_ood, None, None)), flush=True)
            evaluate_monitors(
                network, network_layers,
                dataset, dataset_ood
            )

        # Test with OOD as cov shift
        for j in range(len(all_perturbations)):
            dataset_ood = dataset
            perturbation = all_perturbations[j]

            print("Evaluating %s, for dataset %s and OOD dataset %s." % (network, dataset, (dataset_ood, perturbation, None)), flush=True)
            evaluate_monitors(
                network, network_layers,
                dataset, dataset_ood,
                perturbation=perturbation
            )

        # Test with OOD as adversarial attack
        for j in range(len(all_adver_attacks)):
            dataset_ood = dataset
            adver_attack = all_adver_attacks[j]

            print("Evaluating on %s, for dataset %s and OOD dataset %s." % (network, dataset, (dataset_ood, None, adver_attack)), flush=True)
            evaluate_monitors(
                network, network_layers,
                dataset, dataset_ood,
                adver_attack=adver_attack
            )

f.close()