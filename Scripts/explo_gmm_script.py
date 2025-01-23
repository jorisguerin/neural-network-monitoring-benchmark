import csv
import os
import torch

from Monitors import GMMMonitor
from Monitors import MahalanobisMonitor
from evaluation import evaluate_monitor

import warnings
warnings.filterwarnings('ignore')


batch_size = 100
TORCH_DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

all_networks = ["resnet", "densenet"]
all_networks_layers = [[32], [98]]
all_datasets = ["cifar10"]
all_datasets_ood = [["cifar100", "svhn", "lsun"]]
all_perturbations = ["brightness", "blur", "pixelization"]
all_adver_attacks = ["fgsm", "deepfool", "pgd"]

all_hp_n_components = ["auto_bic", "auto_knee"]
all_hp_t_covariance = ["full", "diag", "tied"]

path_to_save_results = "Results/GMM_study/"
if not os.path.exists(path_to_save_results):
    os.makedirs(path_to_save_results)

header = [
    "Network", "Network Layer",
    "Dataset", "Dataset OOD", "Perturbation", "Attack",
    "Monitor", "GMM n_components", "GMM t_covariance", "GMM architecture",
    "AUPR (OMS)", "AUROC (OMS)", "TNR @95 TPR (OMS)",
    "Train time", "Infer time"
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
    'is_train_timed': True,
    'is_infer_timed': True,
}

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

                for hp_n in all_hp_n_components:
                    for hp_c in all_hp_t_covariance:
                        print("... HPs n=%s, c=%s" % (hp_n, hp_c), flush=True)
                        monitor = GMMMonitor(
                            dataset,
                            network,
                            network_layers[0],
                            n_components=hp_n,
                            t_covariance=hp_c)
                        
                        config['monitor'] = monitor
                        result = evaluate_monitor(config)

                        print("results:", result)
                        data = [
                            network, network_layers[0],
                            dataset, dataset_ood, str(None), str(None),
                            "GMM_%s_%s" % (hp_n, hp_c), hp_n, hp_c, 
                            str([gmm.n_components for gmm in monitor.gmm]),
                            result["aupr_score"],
                            result["auroc_score"],
                            result["tnr_frac_tpr"],
                            result["train_time"],
                            result["infer_time"]
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

                for hp_n in all_hp_n_components:
                    for hp_c in all_hp_t_covariance:
                        monitor = GMMMonitor(
                            dataset,
                            network,
                            network_layers[0],
                            n_components=hp_n,
                            t_covariance=hp_c)
                        
                        config['monitor'] = monitor
                        result = evaluate_monitor(config)

                        data = [
                            network, network_layers[0],
                            dataset, dataset_ood, perturbation, str(None),
                            "GMM_%s_%s" % (hp_n, hp_c),
                            str([gmm.n_components for gmm in monitor.gmm]),
                            result["aupr_score"],
                            result["auroc_score"],
                            result["tnr_frac_tpr"],
                            result["train_time"],
                            result["infer_time"]
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

                for hp_n in all_hp_n_components:
                    for hp_c in all_hp_t_covariance:
                        monitor = GMMMonitor(
                            dataset,
                            network,
                            network_layers[0],
                            n_components=hp_n,
                            t_covariance=hp_c)
                        
                        config['monitor'] = monitor
                        result = evaluate_monitor(config)

                        data = [
                            network, network_layers[0],
                            dataset, dataset_ood, str(None), adver_attack,
                            "GMM_%s_%s" % (hp_n, hp_c),
                            str([gmm.n_components for gmm in monitor.gmm]),
                            result["aupr_score"],
                            result["auroc_score"],
                            result["tnr_frac_tpr"],
                            result["train_time"],
                            result["infer_time"]
                        ]
                        writer.writerow(data)
                f.close()