import os
import csv

import torch

from dataset import Dataset
from feature_extractor import FeatureExtractor
from evaluation import Evaluator
from monitors import OutsideTheBoxMonitor, GaussianMixtureMonitor, MaxSoftmaxProbabilityMonitor, MaxLogitMonitor, \
    EnergyMonitor, ReActMonitor, MahalanobisMonitor

from sklearn.metrics import accuracy_score

from termcolor import colored

batch_size = 10
device_name = 'cuda:0' if torch.cuda.is_available() else 'cpu'

all_models = ["resnet", "densenet"]

all_layers_ids = [[32], [98]]

all_id_datasets = ["cifar10", "svhn", "cifar100"]

all_ood_datasets = [["cifar100", "svhn", "lsun"],
                    ["cifar10", "tiny_imagenet", "lsun"],
                    ["cifar10", "svhn", "lsun"]]

all_perturbations = ["brightness", "blur", "pixelization"]

all_attacks = ["fgsm", "deepfool", "pgd"]

header = ["Model", "Layer",
          "ID dataset", "OOD dataset", "Perturbation", "Attack",
          "ID accuracy", "OOD accuracy",
          "Monitor", "Constraint Covariance", "N clusters",
          "AUPR", "AUROC", "TNR@95TPR"]

save_results_path = "Results/ICCV/"
if not os.path.exists(save_results_path):
    os.makedirs(save_results_path)
f = open(save_results_path + "full_results.csv", "w", encoding="UTF8")
writer = csv.writer(f)
writer.writerow(header)

# Parameters experiments
cov_constraints = ["full", "diag", "tied", "spherical"]
n_clusters = [1, 2, 3, 5, 7]
react_clip = [0.8, 0.9, 0.95, 0.99]


for i in range(len(all_models)):
    model = all_models[i]
    for layer in all_layers_ids[i]:
        for j in range(len(all_id_datasets)):
            id_dataset = all_id_datasets[j]
            for k in range(len(all_ood_datasets[j])):
                ood_dataset = all_ood_datasets[j][k]

                print(colored("%s, %i, %s, %s" % (model, layer, id_dataset, ood_dataset), "blue"))

                dataset_train = Dataset(id_dataset, "train", model, batch_size=batch_size)
                dataset_test = Dataset(id_dataset, "test", model, batch_size=batch_size)
                dataset_ood = Dataset(ood_dataset, "test", model, None, None, batch_size=batch_size)

                feature_extractor = FeatureExtractor(model, id_dataset, [layer], device_name)

                features_train, logits_train, softmax_train, \
                    pred_train, lab_train = feature_extractor.get_features(dataset_train)
                features_test, logits_test, softmax_test, \
                    pred_test, lab_test = feature_extractor.get_features(dataset_test)
                features_ood, logits_ood, softmax_ood, \
                    pred_ood, lab_ood = feature_extractor.get_features(dataset_ood)

                eval_oms = Evaluator("oms", is_novelty=(id_dataset != ood_dataset))
                eval_oms.fit_ground_truth(lab_test, lab_ood, pred_test, pred_ood)

                id_accuracy = accuracy_score(lab_test, pred_test)
                if id_dataset == ood_dataset:
                    ood_accuracy = accuracy_score(lab_ood, pred_ood)
                else:
                    ood_accuracy = 0

                for cc in cov_constraints:
                    for nc in n_clusters:

                        monitor = GaussianMixtureMonitor(id_dataset, model, layer, n_components=nc,
                                                         constraint=cc, is_cv=True)
                        monitor.fit(features_train[0], pred_train, lab_train, save=True)

                        scores_test = monitor.predict(features_test[0], pred_test)
                        scores_ood = monitor.predict(features_ood[0], pred_ood)

                        aupr = eval_oms.get_average_precision(-scores_test, -scores_ood)
                        auroc = eval_oms.get_auroc(-scores_test, -scores_ood)
                        tnr95tpr = eval_oms.get_tnr_frac_tpr_oms(-scores_test, -scores_ood, frac=0.95)

                        data = [model, layer,
                                id_dataset, ood_dataset, str(None), str(None),
                                id_accuracy, ood_accuracy,
                                "GMM", cc, nc,
                                aupr, auroc, tnr95tpr]
                        writer.writerow(data)

                for nc in n_clusters:
                    monitor = OutsideTheBoxMonitor(id_dataset, model, layer, n_clusters=nc, is_cv=True)
                    monitor.fit(features_train[0], pred_train, lab_train, save=True)

                    scores_test = monitor.predict(features_test[0], pred_test)
                    scores_ood = monitor.predict(features_ood[0], pred_ood)

                    aupr = eval_oms.get_average_precision(scores_test, scores_ood)
                    auroc = eval_oms.get_auroc(scores_test, scores_ood)
                    tnr95tpr = eval_oms.get_tnr_frac_tpr_oms(scores_test, scores_ood, frac=0.95)

                    data = [model, layer,
                            id_dataset, ood_dataset, str(None), str(None),
                            id_accuracy, ood_accuracy,
                            "OTB", str(None), nc,
                            aupr, auroc, tnr95tpr]
                    writer.writerow(data)

                monitor = MahalanobisMonitor(id_dataset, model, layer)
                monitor.fit(features_train[0], pred_train, lab_train, save=True)

                scores_test = monitor.predict(features_test[0], pred_test)
                scores_ood = monitor.predict(features_ood[0], pred_ood)

                aupr = eval_oms.get_average_precision(-scores_test, -scores_ood)
                auroc = eval_oms.get_auroc(-scores_test, -scores_ood)
                tnr95tpr = eval_oms.get_tnr_frac_tpr_oms(-scores_test, -scores_ood, frac=0.95)

                data = [model, layer,
                        id_dataset, ood_dataset, str(None), str(None),
                        id_accuracy, ood_accuracy,
                        "Mahalanobis", str(None), str(None),
                        aupr, auroc, tnr95tpr]
                writer.writerow(data)

                monitor = MaxSoftmaxProbabilityMonitor()
                monitor.fit()

                scores_test = monitor.predict(softmax_test)
                scores_ood = monitor.predict(softmax_ood)

                aupr = eval_oms.get_average_precision(-scores_test, -scores_ood)
                auroc = eval_oms.get_auroc(-scores_test, -scores_ood)
                tnr95tpr = eval_oms.get_tnr_frac_tpr_oms(-scores_test, -scores_ood, frac=0.95)

                data = [model, layer,
                        id_dataset, ood_dataset, str(None), str(None),
                        id_accuracy, ood_accuracy,
                        "MSP", str(None), str(None),
                        aupr, auroc, tnr95tpr]
                writer.writerow(data)

                for clip in react_clip:
                    monitor = ReActMonitor(quantile_value=clip, mode="msp")
                    monitor.fit(feature_extractor, features_train[-1])

                    scores_test = monitor.predict(features_test[-1])
                    scores_ood = monitor.predict(features_ood[-1])

                    aupr = eval_oms.get_average_precision(-scores_test, -scores_ood)
                    auroc = eval_oms.get_auroc(-scores_test, -scores_ood)
                    tnr95tpr = eval_oms.get_tnr_frac_tpr_oms(-scores_test, -scores_ood, frac=0.95)

                    data = [model, layer,
                            id_dataset, ood_dataset, str(None), str(None),
                            id_accuracy, ood_accuracy,
                            "ReAct_MSP", str(None), clip,
                            aupr, auroc, tnr95tpr]
                    writer.writerow(data)

                monitor = EnergyMonitor(temperature=1)
                monitor.fit()

                scores_test = monitor.predict(logits_test)
                scores_ood = monitor.predict(logits_ood)

                aupr = eval_oms.get_average_precision(-scores_test, -scores_ood)
                auroc = eval_oms.get_auroc(-scores_test, -scores_ood)
                tnr95tpr = eval_oms.get_tnr_frac_tpr_oms(-scores_test, -scores_ood, frac=0.95)

                data = [model, layer,
                        id_dataset, ood_dataset, str(None), str(None),
                        id_accuracy, ood_accuracy,
                        "Energy", str(None), str(None),
                        aupr, auroc, tnr95tpr]
                writer.writerow(data)

                for clip in react_clip:
                    monitor = ReActMonitor(quantile_value=clip)
                    monitor.fit(feature_extractor, features_train[-1])

                    scores_test = monitor.predict(features_test[-1])
                    scores_ood = monitor.predict(features_ood[-1])

                    aupr = eval_oms.get_average_precision(-scores_test, -scores_ood)
                    auroc = eval_oms.get_auroc(-scores_test, -scores_ood)
                    tnr95tpr = eval_oms.get_tnr_frac_tpr_oms(-scores_test, -scores_ood, frac=0.95)

                    data = [model, layer,
                            id_dataset, ood_dataset, str(None), str(None),
                            id_accuracy, ood_accuracy,
                            "ReAct_Energy", str(None), clip,
                            aupr, auroc, tnr95tpr]
                    writer.writerow(data)

            for k in range(len(all_perturbations)):
                ood_dataset = id_dataset

                additional_transform = all_perturbations[k]

                print(colored("%s, %i, %s, %s" % (model, layer, id_dataset, additional_transform), "blue"))

                dataset_train = Dataset(id_dataset, "train", model, batch_size=batch_size)
                dataset_test = Dataset(id_dataset, "test", model, batch_size=batch_size)
                dataset_ood = Dataset(id_dataset, "test", model, additional_transform, None, batch_size=batch_size)

                feature_extractor = FeatureExtractor(model, id_dataset, [layer], device_name)

                features_train, logits_train, softmax_train, \
                    pred_train, lab_train = feature_extractor.get_features(dataset_train)
                features_test, logits_test, softmax_test, \
                    pred_test, lab_test = feature_extractor.get_features(dataset_test)
                features_ood, logits_ood, softmax_ood, \
                    pred_ood, lab_ood = feature_extractor.get_features(dataset_ood)

                eval_oms = Evaluator("oms", is_novelty=(id_dataset != ood_dataset))
                eval_oms.fit_ground_truth(lab_test, lab_ood, pred_test, pred_ood)

                id_accuracy = accuracy_score(lab_test, pred_test)
                if id_dataset == ood_dataset:
                    ood_accuracy = accuracy_score(lab_ood, pred_ood)
                else:
                    ood_accuracy = 0

                for cc in cov_constraints:
                    for nc in n_clusters:
                        monitor = GaussianMixtureMonitor(id_dataset, model, layer, n_components=nc,
                                                         constraint=cc, is_cv=True)
                        monitor.fit(features_train[0], pred_train, lab_train, save=True)

                        scores_test = monitor.predict(features_test[0], pred_test)
                        scores_ood = monitor.predict(features_ood[0], pred_ood)

                        aupr = eval_oms.get_average_precision(-scores_test, -scores_ood)
                        auroc = eval_oms.get_auroc(-scores_test, -scores_ood)
                        tnr95tpr = eval_oms.get_tnr_frac_tpr_oms(-scores_test, -scores_ood, frac=0.95)

                        data = [model, layer,
                                id_dataset, ood_dataset, str(additional_transform), str(None),
                                id_accuracy, ood_accuracy,
                                "GMM", cc, nc,
                                aupr, auroc, tnr95tpr]
                        writer.writerow(data)

                for nc in n_clusters:
                    monitor = OutsideTheBoxMonitor(id_dataset, model, layer, n_clusters=nc, is_cv=True)
                    monitor.fit(features_train[0], pred_train, lab_train, save=True)

                    scores_test = monitor.predict(features_test[0], pred_test)
                    scores_ood = monitor.predict(features_ood[0], pred_ood)

                    aupr = eval_oms.get_average_precision(scores_test, scores_ood)
                    auroc = eval_oms.get_auroc(scores_test, scores_ood)
                    tnr95tpr = eval_oms.get_tnr_frac_tpr_oms(scores_test, scores_ood, frac=0.95)

                    data = [model, layer,
                            id_dataset, ood_dataset, str(additional_transform), str(None),
                            id_accuracy, ood_accuracy,
                            "OTB", str(None), nc,
                            aupr, auroc, tnr95tpr]
                    writer.writerow(data)

                monitor = MahalanobisMonitor(id_dataset, model, layer)
                monitor.fit(features_train[0], pred_train, lab_train, save=True)

                scores_test = monitor.predict(features_test[0], pred_test)
                scores_ood = monitor.predict(features_ood[0], pred_ood)

                aupr = eval_oms.get_average_precision(-scores_test, -scores_ood)
                auroc = eval_oms.get_auroc(-scores_test, -scores_ood)
                tnr95tpr = eval_oms.get_tnr_frac_tpr_oms(-scores_test, -scores_ood, frac=0.95)

                data = [model, layer,
                        id_dataset, ood_dataset, str(additional_transform), str(None),
                        id_accuracy, ood_accuracy,
                        "Mahalanobis", str(None), str(None),
                        aupr, auroc, tnr95tpr]
                writer.writerow(data)

                monitor = MaxSoftmaxProbabilityMonitor()
                monitor.fit()

                scores_test = monitor.predict(softmax_test)
                scores_ood = monitor.predict(softmax_ood)

                aupr = eval_oms.get_average_precision(-scores_test, -scores_ood)
                auroc = eval_oms.get_auroc(-scores_test, -scores_ood)
                tnr95tpr = eval_oms.get_tnr_frac_tpr_oms(-scores_test, -scores_ood, frac=0.95)

                data = [model, layer,
                        id_dataset, ood_dataset, str(additional_transform), str(None),
                        id_accuracy, ood_accuracy,
                        "MSP", str(None), str(None),
                        aupr, auroc, tnr95tpr]
                writer.writerow(data)

                for clip in react_clip:
                    monitor = ReActMonitor(quantile_value=clip, mode="msp")
                    monitor.fit(feature_extractor, features_train[-1])

                    scores_test = monitor.predict(features_test[-1])
                    scores_ood = monitor.predict(features_ood[-1])

                    aupr = eval_oms.get_average_precision(-scores_test, -scores_ood)
                    auroc = eval_oms.get_auroc(-scores_test, -scores_ood)
                    tnr95tpr = eval_oms.get_tnr_frac_tpr_oms(-scores_test, -scores_ood, frac=0.95)

                    data = [model, layer,
                            id_dataset, ood_dataset, str(additional_transform), str(None),
                            id_accuracy, ood_accuracy,
                            "ReAct_MSP", str(None), clip,
                            aupr, auroc, tnr95tpr]
                    writer.writerow(data)

                monitor = EnergyMonitor(temperature=1)
                monitor.fit()

                scores_test = monitor.predict(logits_test)
                scores_ood = monitor.predict(logits_ood)

                aupr = eval_oms.get_average_precision(-scores_test, -scores_ood)
                auroc = eval_oms.get_auroc(-scores_test, -scores_ood)
                tnr95tpr = eval_oms.get_tnr_frac_tpr_oms(-scores_test, -scores_ood, frac=0.95)

                data = [model, layer,
                        id_dataset, ood_dataset, str(additional_transform), str(None),
                        id_accuracy, ood_accuracy,
                        "Energy", str(None), str(None),
                        aupr, auroc, tnr95tpr]
                writer.writerow(data)

                for clip in react_clip:
                    monitor = ReActMonitor(quantile_value=clip)
                    monitor.fit(feature_extractor, features_train[-1])

                    scores_test = monitor.predict(features_test[-1])
                    scores_ood = monitor.predict(features_ood[-1])

                    aupr = eval_oms.get_average_precision(-scores_test, -scores_ood)
                    auroc = eval_oms.get_auroc(-scores_test, -scores_ood)
                    tnr95tpr = eval_oms.get_tnr_frac_tpr_oms(-scores_test, -scores_ood, frac=0.95)

                    data = [model, layer,
                            id_dataset, ood_dataset, str(additional_transform), str(None),
                            id_accuracy, ood_accuracy,
                            "ReAct_Energy", str(None), clip,
                            aupr, auroc, tnr95tpr]
                    writer.writerow(data)

            for k in range(len(all_attacks)):
                ood_dataset = id_dataset

                adversarial_attack = all_attacks[k]

                print(colored("%s, %i, %s, %s" % (model, layer, id_dataset, adversarial_attack), "blue"))

                dataset_train = Dataset(id_dataset, "train", model, batch_size=batch_size)
                dataset_test = Dataset(id_dataset, "test", model, batch_size=batch_size)
                dataset_ood = Dataset(id_dataset, "test", model, None, adversarial_attack, batch_size=batch_size)

                feature_extractor = FeatureExtractor(model, id_dataset, [layer], device_name)

                features_train, logits_train, softmax_train, \
                    pred_train, lab_train = feature_extractor.get_features(dataset_train)
                features_test, logits_test, softmax_test, \
                    pred_test, lab_test = feature_extractor.get_features(dataset_test)
                features_ood, logits_ood, softmax_ood, \
                    pred_ood, lab_ood = feature_extractor.get_features(dataset_ood)

                eval_oms = Evaluator("oms", is_novelty=(id_dataset != ood_dataset))
                eval_oms.fit_ground_truth(lab_test, lab_ood, pred_test, pred_ood)

                id_accuracy = accuracy_score(lab_test, pred_test)
                if id_dataset == ood_dataset:
                    ood_accuracy = accuracy_score(lab_ood, pred_ood)
                else:
                    ood_accuracy = 0

                for cc in cov_constraints:
                    for nc in n_clusters:
                        monitor = GaussianMixtureMonitor(id_dataset, model, layer, n_components=nc,
                                                         constraint=cc, is_cv=True)
                        monitor.fit(features_train[0], pred_train, lab_train, save=True)

                        scores_test = monitor.predict(features_test[0], pred_test)
                        scores_ood = monitor.predict(features_ood[0], pred_ood)

                        aupr = eval_oms.get_average_precision(-scores_test, -scores_ood)
                        auroc = eval_oms.get_auroc(-scores_test, -scores_ood)
                        tnr95tpr = eval_oms.get_tnr_frac_tpr_oms(-scores_test, -scores_ood, frac=0.95)

                        data = [model, layer,
                                id_dataset, ood_dataset, str(None), str(adversarial_attack),
                                id_accuracy, ood_accuracy,
                                "GMM", cc, nc,
                                aupr, auroc, tnr95tpr]
                        writer.writerow(data)

                for nc in n_clusters:
                    monitor = OutsideTheBoxMonitor(id_dataset, model, layer, n_clusters=nc, is_cv=True)
                    monitor.fit(features_train[0], pred_train, lab_train, save=True)

                    scores_test = monitor.predict(features_test[0], pred_test)
                    scores_ood = monitor.predict(features_ood[0], pred_ood)

                    aupr = eval_oms.get_average_precision(scores_test, scores_ood)
                    auroc = eval_oms.get_auroc(scores_test, scores_ood)
                    tnr95tpr = eval_oms.get_tnr_frac_tpr_oms(scores_test, scores_ood, frac=0.95)

                    data = [model, layer,
                            id_dataset, ood_dataset, str(None), str(adversarial_attack),
                            id_accuracy, ood_accuracy,
                            "OTB", str(None), nc,
                            aupr, auroc, tnr95tpr]
                    writer.writerow(data)

                monitor = MahalanobisMonitor(id_dataset, model, layer)
                monitor.fit(features_train[0], pred_train, lab_train, save=True)

                scores_test = monitor.predict(features_test[0], pred_test)
                scores_ood = monitor.predict(features_ood[0], pred_ood)

                aupr = eval_oms.get_average_precision(-scores_test, -scores_ood)
                auroc = eval_oms.get_auroc(-scores_test, -scores_ood)
                tnr95tpr = eval_oms.get_tnr_frac_tpr_oms(-scores_test, -scores_ood, frac=0.95)

                data = [model, layer,
                        id_dataset, ood_dataset, str(None), str(adversarial_attack),
                        id_accuracy, ood_accuracy,
                        "Mahalanobis", str(None), str(None),
                        aupr, auroc, tnr95tpr]
                writer.writerow(data)

                monitor = MaxSoftmaxProbabilityMonitor()
                monitor.fit()

                scores_test = monitor.predict(softmax_test)
                scores_ood = monitor.predict(softmax_ood)

                aupr = eval_oms.get_average_precision(-scores_test, -scores_ood)
                auroc = eval_oms.get_auroc(-scores_test, -scores_ood)
                tnr95tpr = eval_oms.get_tnr_frac_tpr_oms(-scores_test, -scores_ood, frac=0.95)

                data = [model, layer,
                        id_dataset, ood_dataset, str(None), str(adversarial_attack),
                        id_accuracy, ood_accuracy,
                        "MSP", str(None), str(None),
                        aupr, auroc, tnr95tpr]
                writer.writerow(data)

                for clip in react_clip:
                    monitor = ReActMonitor(quantile_value=clip, mode="msp")
                    monitor.fit(feature_extractor, features_train[-1])

                    scores_test = monitor.predict(features_test[-1])
                    scores_ood = monitor.predict(features_ood[-1])

                    aupr = eval_oms.get_average_precision(-scores_test, -scores_ood)
                    auroc = eval_oms.get_auroc(-scores_test, -scores_ood)
                    tnr95tpr = eval_oms.get_tnr_frac_tpr_oms(-scores_test, -scores_ood, frac=0.95)

                    data = [model, layer,
                            id_dataset, ood_dataset, str(None), str(adversarial_attack),
                            id_accuracy, ood_accuracy,
                            "ReAct_MSP", str(None), clip,
                            aupr, auroc, tnr95tpr]
                    writer.writerow(data)

                monitor = EnergyMonitor(temperature=1)
                monitor.fit()

                scores_test = monitor.predict(logits_test)
                scores_ood = monitor.predict(logits_ood)

                aupr = eval_oms.get_average_precision(-scores_test, -scores_ood)
                auroc = eval_oms.get_auroc(-scores_test, -scores_ood)
                tnr95tpr = eval_oms.get_tnr_frac_tpr_oms(-scores_test, -scores_ood, frac=0.95)

                data = [model, layer,
                        id_dataset, ood_dataset, str(None), str(adversarial_attack),
                        id_accuracy, ood_accuracy,
                        "Energy", str(None), str(None),
                        aupr, auroc, tnr95tpr]
                writer.writerow(data)

                for clip in react_clip:
                    monitor = ReActMonitor(quantile_value=clip)
                    monitor.fit(feature_extractor, features_train[-1])

                    scores_test = monitor.predict(features_test[-1])
                    scores_ood = monitor.predict(features_ood[-1])

                    aupr = eval_oms.get_average_precision(-scores_test, -scores_ood)
                    auroc = eval_oms.get_auroc(-scores_test, -scores_ood)
                    tnr95tpr = eval_oms.get_tnr_frac_tpr_oms(-scores_test, -scores_ood, frac=0.95)

                    data = [model, layer,
                            id_dataset, ood_dataset, str(None), str(adversarial_attack),
                            id_accuracy, ood_accuracy,
                            "ReAct_Energy", str(None), clip,
                            aupr, auroc, tnr95tpr]
                    writer.writerow(data)

f.close()
