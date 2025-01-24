"""
Script to generate the results seen in the paper

"""

import csv
import os
import torch

from Monitors import (
    EnergyMonitor,
    MSPMonitor,
    OTBMonitor,
    ReActMonitor,
    MahalanobisMonitor,
)
from evaluation import evaluate_monitor

import warnings
warnings.filterwarnings('ignore')


batch_size = 100
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
if not os.path.exists(path_to_save_results):
    os.makedirs(path_to_save_results)

header = [
    "Network", "Network Layer",
    "Dataset", "Dataset OOD", "Perturbation", "Attack",
    "Precision OMS@OOD*", "Recall OMS@OOD*", "F1 OMS@OOD*"
    "Monitor",
    "Precision OOD", "Recall OOD", "F1 OOD"
    "Precision OMS", "Recall OMS", "F1 OMS"
]
config = {
    'network': None,
    'network_layers': None,
    'dataset': None,
    'dataset_ood': None,
    'monitor': None,
    'monitor_train_params': True,
    'monitor_infer_params': 'features',
    'perturbation': None,
    'adver_attack': None,
    'batch_size': batch_size,
    'TORCH_DEVICE': TORCH_DEVICE,
    'evaluation_mode': 'oms',
    'evaluation_metrics': ['aupr_score', 'auroc_score', 'tnr_frac_tpr'],
    'is_train_timed': False,
    'is_infer_timed': False,
}


def _evaluate():
    pass


def _evaluate_MSP(
        network,
        network_layers,
        dataset,
        dataset_ood,
        perturbation=None,
        adver_attack=None,
):
    monitor = MSPMonitor()
    


for i_network in range(len(all_networks)):
    network = all_networks[i_network]
    network_layers = all_networks_layers[i_network]

    config['network'] = network
    config['network_layers'] = network_layers

    for i_dataset in range(len(all_datasets)):
        dataset = all_datasets[i_dataset]
        
        # Test with OOD as novelty
        for j_dataset in range(len(all_datasets_ood)):
            dataset_ood = all_datasets_ood[i_dataset][j_dataset]

            config['dataset'] = dataset
            config['dataset_ood'] = dataset_ood
            config['perturbation'] = None
            config['adver_attack'] = None

            print("Evaluating on %s, for dataset %s and OOD dataset %s." % (network, dataset, (dataset_ood, None, None)), flush=True)
            path_to_results_file = path_to_save_results + "%s_%s_%s.csv" % (network, dataset, dataset_ood)

            if (not os.path.exists(path_to_results_file)) or len(list(csv.reader(open(path_to_results_file)))) < 15:
                f = open(path_to_results_file, "w", encoding="UTF8")
                writer = csv.writer(f)
                writer.writerow(header)

                # Try MSP monitor
                monitor = MSPMonitor()
                config['monitor'] = monitor
                config['monitor_train_params'] = False
                config['monitor_infer_params'] = "softmax"
                result = evaluate_monitor(config)
                data = [
                    network, 1,
                    dataset, dataset_ood, str(None), str(None),
                    "MSP",
                    result["aupr_score"],
                    result["auroc_score"],
                    result["tnr_frac_tpr"]
                ]
                writer.writerow(data)

                # Try Ene monitor
                monitor = EnergyMonitor(T=1)
                config['monitor'] = monitor
                config['monitor_train_params'] = False
                config['monitor_infer_params'] = "logits"
                result = evaluate_monitor(config)
                data = [
                    network, 1,
                    dataset, dataset_ood, str(None), str(None),
                    "Ene",
                    result["aupr_score"],
                    result["auroc_score"],
                    result["tnr_frac_tpr"]
                ]
                writer.writerow(data)

                # Try OtB monitor
                monitor = OTBMonitor(dataset, network, network_layers, n_clusters=10)
                config['monitor'] = monitor
                config['monitor_train_params'] = False
                config['monitor_infer_params'] = "features"
                result = evaluate_monitor(config)
                data = [
                    network, network_layers[0],
                    dataset, dataset_ood, str(None), str(None),
                    "OTB",
                    result["aupr_score"],
                    result["auroc_score"],
                    result["tnr_frac_tpr"]
                ]
                writer.writerow(data)

                # Try Maha monitor
                monitor = MahalanobisMonitor(dataset, network, network_layers, is_tied=True)
                config['monitor'] = monitor
                config['monitor_train_params'] = False
                config['monitor_infer_params'] = "features"
                result = evaluate_monitor(config)
                data = [
                    network, network_layers,
                    dataset, dataset_ood, str(None), str(None),
                    "Maha",
                    result["aupr_score"],
                    result["auroc_score"],
                    result["tnr_frac_tpr"]
                ]
                writer.writerow(data)
                f.close()

        # Test with OOD as cov shift
        for j in range(len(all_perturbations)):
            dataset_ood = dataset
            perturbation = all_perturbations[j]

            config['dataset'] = dataset
            config['dataset_ood'] = dataset_ood
            config['perturbation'] = perturbation
            config['adver_attack'] = None

            print("Evaluating on %s, for dataset %s and OOD dataset %s." % (network, dataset, (dataset_ood, perturbation, None)), flush=True)
            path_to_results_file = path_to_save_results + "%s_%s_%s.csv" % (network, dataset, perturbation)

            if (not os.path.exists(path_to_results_file)) or len(list(csv.reader(open(path_to_results_file)))) < 15:
                f = open(path_to_results_file, "w", encoding="UTF8")
                writer = csv.writer(f)
                writer.writerow(header)

                # Try MSP monitor
                monitor = MSPMonitor()
                config['monitor'] = monitor
                config['monitor_train_params'] = False
                config['monitor_infer_params'] = "softmax"
                result = evaluate_monitor(config)
                data = [
                    network, 1,
                    dataset, dataset_ood, perturbation, str(None),
                    "MSP",
                    result["aupr_score"],
                    result["auroc_score"],
                    result["tnr_frac_tpr"]
                ]
                writer.writerow(data)

                # Try Ene monitor
                monitor = EnergyMonitor(T=1)
                config['monitor'] = monitor
                config['monitor_train_params'] = False
                config['monitor_infer_params'] = "logits"
                result = evaluate_monitor(config)
                data = [
                    network, 1,
                    dataset, dataset_ood, perturbation, str(None),
                    "Ene",
                    result["aupr_score"],
                    result["auroc_score"],
                    result["tnr_frac_tpr"]
                ]
                writer.writerow(data)

                # Try OtB monitor
                monitor = OTBMonitor(dataset, network, network_layers, n_clusters=10)
                config['monitor'] = monitor
                config['monitor_train_params'] = False
                config['monitor_infer_params'] = "features"
                result = evaluate_monitor(config)
                data = [
                    network, network_layers[0],
                    dataset, dataset_ood, perturbation, str(None),
                    "OTB",
                    result["aupr_score"],
                    result["auroc_score"],
                    result["tnr_frac_tpr"]
                ]
                writer.writerow(data)

                # Try Maha monitor
                monitor = MahalanobisMonitor(dataset, network, network_layers, is_tied=True)
                config['monitor'] = monitor
                config['monitor_train_params'] = False
                config['monitor_infer_params'] = "features"
                result = evaluate_monitor(config)
                data = [
                    network, network_layers,
                    dataset, dataset_ood, perturbation, str(None),
                    "Maha",
                    result["aupr_score"],
                    result["auroc_score"],
                    result["tnr_frac_tpr"]
                ]
                writer.writerow(data)
                f.close()

        # Test with OOD as adversarial attack
        for j in range(len(all_adver_attacks)):
            dataset_ood = dataset
            adver_attack = all_adver_attacks[j]

            config['dataset'] = dataset
            config['dataset_ood'] = dataset_ood
            config['perturbation'] = None
            config['adver_attack'] = adver_attack

            print("Evaluating on %s, for dataset %s and OOD dataset %s." % (network, dataset, (dataset_ood, None, adver_attack)), flush=True)
            path_to_results_file = path_to_save_results + "%s_%s_%s.csv" % (network, dataset, adver_attack)

            if (not os.path.exists(path_to_results_file)) or len(list(csv.reader(open(path_to_results_file)))) < 15:
                f = open(path_to_results_file, "w", encoding="UTF8")
                writer = csv.writer(f)
                writer.writerow(header)

                # Try MSP monitor
                monitor = MSPMonitor()
                config['monitor'] = monitor
                config['monitor_train_params'] = False
                config['monitor_infer_params'] = "softmax"
                result = evaluate_monitor(config)
                data = [
                    network, 1,
                    dataset, dataset_ood, str(None), adver_attack,
                    "MSP",
                    result["aupr_score"],
                    result["auroc_score"],
                    result["tnr_frac_tpr"]
                ]
                writer.writerow(data)

                # Try Ene monitor
                monitor = EnergyMonitor(T=1)
                config['monitor'] = monitor
                config['monitor_train_params'] = False
                config['monitor_infer_params'] = "logits"
                result = evaluate_monitor(config)
                data = [
                    network, 1,
                    dataset, dataset_ood, str(None), adver_attack,
                    "Ene",
                    result["aupr_score"],
                    result["auroc_score"],
                    result["tnr_frac_tpr"]
                ]
                writer.writerow(data)

                # Try OtB monitor
                monitor = OTBMonitor(dataset, network, network_layers, n_clusters=10)
                config['monitor'] = monitor
                config['monitor_train_params'] = False
                config['monitor_infer_params'] = "features"
                result = evaluate_monitor(config)
                data = [
                    network, network_layers[0],
                    dataset, dataset_ood, str(None), adver_attack,
                    "OTB",
                    result["aupr_score"],
                    result["auroc_score"],
                    result["tnr_frac_tpr"]
                ]
                writer.writerow(data)

                # Try Maha monitor
                monitor = MahalanobisMonitor(dataset, network, network_layers, is_tied=True)
                config['monitor'] = monitor
                config['monitor_train_params'] = False
                config['monitor_infer_params'] = "features"
                result = evaluate_monitor(config)
                data = [
                    network, network_layers,
                    dataset, dataset_ood, str(None), adver_attack,
                    "Maha",
                    result["aupr_score"],
                    result["auroc_score"],
                    result["tnr_frac_tpr"]
                ]
                writer.writerow(data)
                f.close()