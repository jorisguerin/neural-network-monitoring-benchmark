import os
import csv

import torch
from sklearn.metrics import accuracy_score

from dataset import Dataset
from feature_extractor import FeatureExtractor
from evaluation import Evaluator
from monitors import MahalanobisMonitor, OutsideTheBoxMonitor, GaussianMixtureMonitor

from Params.params_monitors import gmm_n_comp_values

batch_size = 10
device_name = 'cuda:0' if torch.cuda.is_available() else 'cpu'

all_models = ["resnet", "densenet"]

all_layers_ids = [[31],
                  [97]]

all_id_datasets = ["cifar10", "svhn", "cifar100"]

all_ood_datasets = [["cifar100", "svhn", "lsun"],
                    ["cifar10", "tiny_imagenet", "lsun"],
                    ["cifar10", "svhn", "lsun"]]

all_perturbations = ["brightness", "blur", "pixelization"]

all_attacks = ["fgsm", "deepfool", "pgd"]

header = ["Model", "Layer",
          "ID dataset", "OOD dataset", "Perturbation", "Attack",
          "ID accuracy", "OOD accuracy",
          "Monitor",
          "Precision OMS", "Recall OMS", "F1 OMS",
          "GMM components"]

save_results_path = "Results/GMM/"
if not os.path.exists(save_results_path):
    os.makedirs(save_results_path)

for i in range(len(all_models)):
    model = all_models[i]
    layers_ids = all_layers_ids[i]
    for j in range(len(all_id_datasets)):
        id_dataset = all_id_datasets[j]
        for k in range(len(all_ood_datasets[j])):
            ood_dataset = all_ood_datasets[j][k]

            print(model, id_dataset, ood_dataset)

            result_file_path = save_results_path + "%s_%s_%s.csv" % (model, id_dataset, ood_dataset)

            if (not os.path.exists(result_file_path)) or len(list(csv.reader(open(result_file_path)))) < 14:
                f = open(result_file_path, "w", encoding="UTF8")
                writer = csv.writer(f)
                writer.writerow(header)

                dataset_train = Dataset(id_dataset, "train", model, batch_size=batch_size)
                dataset_test = Dataset(id_dataset, "test", model, batch_size=batch_size)
                dataset_ood = Dataset(ood_dataset, "test", model, None, None, batch_size=batch_size)

                feature_extractor = FeatureExtractor(model, id_dataset, layers_ids, device_name)

                features_train, logits_train, softmax_train, \
                    pred_train, lab_train = feature_extractor.get_features(dataset_train)
                features_test, logits_test, softmax_test, \
                    pred_test, lab_test = feature_extractor.get_features(dataset_test)
                features_ood, logits_ood, softmax_ood, \
                    pred_ood, lab_ood = feature_extractor.get_features(dataset_ood)

                id_accuracy = accuracy_score(lab_test, pred_test)
                if id_dataset == ood_dataset:
                    ood_accuracy = accuracy_score(lab_ood, pred_ood)
                else:
                    ood_accuracy = 0

                eval_oms = Evaluator("oms", is_novelty=(id_dataset != ood_dataset))

                eval_oms.fit_ground_truth(lab_test, lab_ood, pred_test, pred_ood)

                for n_comp in gmm_n_comp_values:
                    monitor = GaussianMixtureMonitor(id_dataset, model, layers_ids[0], n_components=n_comp,
                                                     constraint="full", is_cv=True)
                    monitor.fit(features_train[0][pred_train == lab_train],
                                pred_train[[pred_train == lab_train]], save=True)

                    scores_test = monitor.predict(features_test[0], pred_test)
                    scores_ood = monitor.predict(features_ood[0], pred_ood)

                    precision_oms, recall_oms, f1_oms = eval_oms.get_metrics(scores_test, scores_ood)

                    data = [model, layers_ids[0],
                            id_dataset, ood_dataset, str(None), str(None),
                            id_accuracy, ood_accuracy,
                            "GMM_" + str(n_comp),
                            precision_oms, recall_oms, f1_oms,
                            str([gmm.n_components for gmm in monitor.gmm])]
                    writer.writerow(data)

                monitor = GaussianMixtureMonitor(id_dataset, model, layers_ids[0], n_components="auto_knee",
                                                 constraint="full", is_cv=True)
                monitor.fit(features_train[0][pred_train == lab_train],
                            pred_train[[pred_train == lab_train]], save=True)

                scores_test = monitor.predict(features_test[0], pred_test)
                scores_ood = monitor.predict(features_ood[0], pred_ood)

                precision_oms, recall_oms, f1_oms = eval_oms.get_metrics(scores_test, scores_ood)

                data = [model, layers_ids[0],
                        id_dataset, ood_dataset, str(None), str(None),
                        id_accuracy, ood_accuracy,
                        "GMM_knee",
                        precision_oms, recall_oms, f1_oms,
                        str([gmm.n_components for gmm in monitor.gmm])]
                writer.writerow(data)

                monitor = GaussianMixtureMonitor(id_dataset, model, layers_ids[0], n_components="auto_bic",
                                                 constraint="full", is_cv=True)
                monitor.fit(features_train[0][pred_train == lab_train],
                            pred_train[[pred_train == lab_train]], save=True)

                scores_test = monitor.predict(features_test[0], pred_test)
                scores_ood = monitor.predict(features_ood[0], pred_ood)

                precision_oms, recall_oms, f1_oms = eval_oms.get_metrics(scores_test, scores_ood)

                data = [model, layers_ids[0],
                        id_dataset, ood_dataset, str(None), str(None),
                        id_accuracy, ood_accuracy,
                        "GMM_bic",
                        precision_oms, recall_oms, f1_oms,
                        str([gmm.n_components for gmm in monitor.gmm])]
                writer.writerow(data)

                monitor = MahalanobisMonitor(id_dataset, model, layers_ids[0], is_tied=True)
                monitor.fit(features_train[0][pred_train == lab_train],
                            pred_train[[pred_train == lab_train]], save=True)

                scores_test = monitor.predict(features_test[0], pred_test)
                scores_ood = monitor.predict(features_ood[0], pred_ood)

                precision_oms, recall_oms, f1_oms = eval_oms.get_metrics(scores_test, scores_ood)

                data = [model, layers_ids[0],
                        id_dataset, ood_dataset, str(None), str(None),
                        id_accuracy, ood_accuracy,
                        "mahalanobis",
                        precision_oms, recall_oms, f1_oms,
                        str(None)]
                writer.writerow(data)

                monitor = OutsideTheBoxMonitor(n_clusters=10)
                monitor.fit(features_train[0][pred_train == lab_train],
                            pred_train[pred_train == lab_train])

                scores_test = monitor.predict(features_test[0], pred_test)
                scores_ood = monitor.predict(features_ood[0], pred_ood)

                precision_oms, recall_oms, f1_oms = eval_oms.get_metrics(scores_test, scores_ood)

                data = [model, layers_ids[0],
                        id_dataset, ood_dataset, str(None), str(None),
                        id_accuracy, ood_accuracy,
                        "oob",
                        precision_oms, recall_oms, f1_oms,
                        str(None)]
                writer.writerow(data)
                f.close()

        for k in range(len(all_perturbations)):
            ood_dataset = id_dataset

            additional_transform = all_perturbations[k]

            print(model, id_dataset, additional_transform)

            result_file_path = save_results_path + "%s_%s_%s.csv" % (model, id_dataset, additional_transform)

            if (not os.path.exists(result_file_path)) or len(list(csv.reader(open(result_file_path)))) < 14:
                f = open(result_file_path, "w", encoding="UTF8")
                writer = csv.writer(f)
                writer.writerow(header)

                dataset_train = Dataset(id_dataset, "train", model, batch_size=batch_size)
                dataset_test = Dataset(id_dataset, "test", model, batch_size=batch_size)
                dataset_ood = Dataset(id_dataset, "test", model, additional_transform, None, batch_size=batch_size)

                feature_extractor = FeatureExtractor(model, id_dataset, layers_ids, device_name)

                features_train, logits_train, softmax_train, \
                    pred_train, lab_train = feature_extractor.get_features(dataset_train)
                features_test, logits_test, softmax_test, \
                    pred_test, lab_test = feature_extractor.get_features(dataset_test)
                features_ood, logits_ood, softmax_ood, \
                    pred_ood, lab_ood = feature_extractor.get_features(dataset_ood)

                id_accuracy = accuracy_score(lab_test, pred_test)
                if id_dataset == ood_dataset:
                    ood_accuracy = accuracy_score(lab_ood, pred_ood)
                else:
                    ood_accuracy = 0

                eval_oms = Evaluator("oms", is_novelty=(id_dataset != ood_dataset))

                eval_oms.fit_ground_truth(lab_test, lab_ood, pred_test, pred_ood)

                for n_comp in gmm_n_comp_values:
                    monitor = GaussianMixtureMonitor(id_dataset, model, layers_ids[0], n_components=n_comp,
                                                     constraint="full", is_cv=True)
                    monitor.fit(features_train[0][pred_train == lab_train],
                                pred_train[[pred_train == lab_train]], save=True)

                    scores_test = monitor.predict(features_test[0], pred_test)
                    scores_ood = monitor.predict(features_ood[0], pred_ood)

                    precision_oms, recall_oms, f1_oms = eval_oms.get_metrics(scores_test, scores_ood)

                    data = [model, layers_ids[0],
                            id_dataset, ood_dataset, str(additional_transform), str(None),
                            id_accuracy, ood_accuracy,
                            "GMM_" + str(n_comp),
                            precision_oms, recall_oms, f1_oms,
                            str([gmm.n_components for gmm in monitor.gmm])]
                    writer.writerow(data)

                monitor = GaussianMixtureMonitor(id_dataset, model, layers_ids[0], n_components="auto_knee",
                                                 constraint="full", is_cv=True)
                monitor.fit(features_train[0][pred_train == lab_train],
                            pred_train[[pred_train == lab_train]], save=True)

                scores_test = monitor.predict(features_test[0], pred_test)
                scores_ood = monitor.predict(features_ood[0], pred_ood)

                precision_oms, recall_oms, f1_oms = eval_oms.get_metrics(scores_test, scores_ood)

                data = [model, layers_ids[0],
                        id_dataset, ood_dataset, str(additional_transform), str(None),
                        id_accuracy, ood_accuracy,
                        "GMM_knee",
                        precision_oms, recall_oms, f1_oms,
                        str([gmm.n_components for gmm in monitor.gmm])]
                writer.writerow(data)

                monitor = GaussianMixtureMonitor(id_dataset, model, layers_ids[0], n_components="auto_bic",
                                                 constraint="full", is_cv=True)
                monitor.fit(features_train[0][pred_train == lab_train],
                            pred_train[[pred_train == lab_train]], save=True)

                scores_test = monitor.predict(features_test[0], pred_test)
                scores_ood = monitor.predict(features_ood[0], pred_ood)

                precision_oms, recall_oms, f1_oms = eval_oms.get_metrics(scores_test, scores_ood)

                data = [model, layers_ids[0],
                        id_dataset, ood_dataset, str(additional_transform), str(None),
                        id_accuracy, ood_accuracy,
                        "GMM_bic",
                        precision_oms, recall_oms, f1_oms,
                        str([gmm.n_components for gmm in monitor.gmm])]
                writer.writerow(data)

                monitor = MahalanobisMonitor(id_dataset, model, layers_ids[0], is_tied=True)
                monitor.fit(features_train[0][pred_train == lab_train],
                            pred_train[[pred_train == lab_train]], save=True)

                scores_test = monitor.predict(features_test[0], pred_test)
                scores_ood = monitor.predict(features_ood[0], pred_ood)

                precision_oms, recall_oms, f1_oms = eval_oms.get_metrics(scores_test, scores_ood)

                data = [model, layers_ids[0],
                        id_dataset, ood_dataset, str(additional_transform), str(None),
                        id_accuracy, ood_accuracy,
                        "mahalanobis",
                        precision_oms, recall_oms, f1_oms,
                        str(None)]
                writer.writerow(data)

                monitor = OutsideTheBoxMonitor(n_clusters=10)
                monitor.fit(features_train[0][pred_train == lab_train],
                            pred_train[pred_train == lab_train])

                scores_test = monitor.predict(features_test[0], pred_test)
                scores_ood = monitor.predict(features_ood[0], pred_ood)

                precision_oms, recall_oms, f1_oms = eval_oms.get_metrics(scores_test, scores_ood)

                data = [model, layers_ids[0],
                        id_dataset, ood_dataset, str(additional_transform), str(None),
                        id_accuracy, ood_accuracy,
                        "oob",
                        precision_oms, recall_oms, f1_oms,
                        str(None)]
                writer.writerow(data)
                f.close()

        for k in range(len(all_attacks)):
            ood_dataset = id_dataset

            adversarial_attack = all_attacks[k]

            print(model, id_dataset, adversarial_attack)

            result_file_path = save_results_path + "%s_%s_%s.csv" % (model, id_dataset, adversarial_attack)

            if (not os.path.exists(result_file_path)) or len(list(csv.reader(open(result_file_path)))) < 14:
                f = open(result_file_path, "w", encoding="UTF8")
                writer = csv.writer(f)
                writer.writerow(header)

                dataset_train = Dataset(id_dataset, "train", model, batch_size=batch_size)
                dataset_test = Dataset(id_dataset, "test", model, batch_size=batch_size)
                dataset_ood = Dataset(id_dataset, "test", model, None, adversarial_attack, batch_size=batch_size)

                feature_extractor = FeatureExtractor(model, id_dataset, layers_ids, device_name)

                features_train, logits_train, softmax_train, \
                    pred_train, lab_train = feature_extractor.get_features(dataset_train)
                features_test, logits_test, softmax_test, \
                    pred_test, lab_test = feature_extractor.get_features(dataset_test)
                features_ood, logits_ood, softmax_ood, \
                    pred_ood, lab_ood = feature_extractor.get_features(dataset_ood)

                id_accuracy = accuracy_score(lab_test, pred_test)
                if id_dataset == ood_dataset:
                    ood_accuracy = accuracy_score(lab_ood, pred_ood)
                else:
                    ood_accuracy = 0

                eval_oms = Evaluator("oms", is_novelty=(id_dataset != ood_dataset))

                eval_oms.fit_ground_truth(lab_test, lab_ood, pred_test, pred_ood)

                for n_comp in gmm_n_comp_values:
                    monitor = GaussianMixtureMonitor(id_dataset, model, layers_ids[0], n_components=n_comp,
                                                     constraint="full", is_cv=True)
                    monitor.fit(features_train[0][pred_train == lab_train],
                                pred_train[[pred_train == lab_train]], save=True)

                    scores_test = monitor.predict(features_test[0], pred_test)
                    scores_ood = monitor.predict(features_ood[0], pred_ood)

                    precision_oms, recall_oms, f1_oms = eval_oms.get_metrics(scores_test, scores_ood)

                    data = [model, layers_ids[0],
                            id_dataset, ood_dataset, str(None), str(adversarial_attack),
                            id_accuracy, ood_accuracy,
                            "GMM_" + str(n_comp),
                            precision_oms, recall_oms, f1_oms,
                            str([gmm.n_components for gmm in monitor.gmm])]
                    writer.writerow(data)

                monitor = GaussianMixtureMonitor(id_dataset, model, layers_ids[0], n_components="auto_knee",
                                                 constraint="full", is_cv=True)
                monitor.fit(features_train[0][pred_train == lab_train],
                            pred_train[[pred_train == lab_train]], save=True)

                scores_test = monitor.predict(features_test[0], pred_test)
                scores_ood = monitor.predict(features_ood[0], pred_ood)

                precision_oms, recall_oms, f1_oms = eval_oms.get_metrics(scores_test, scores_ood)

                data = [model, layers_ids[0],
                        id_dataset, ood_dataset, str(None), str(adversarial_attack),
                        id_accuracy, ood_accuracy,
                        "GMM_knee",
                        precision_oms, recall_oms, f1_oms,
                        str([gmm.n_components for gmm in monitor.gmm])]
                writer.writerow(data)

                monitor = GaussianMixtureMonitor(id_dataset, model, layers_ids[0], n_components="auto_bic",
                                                 constraint="full", is_cv=True)
                monitor.fit(features_train[0][pred_train == lab_train],
                            pred_train[[pred_train == lab_train]], save=True)

                scores_test = monitor.predict(features_test[0], pred_test)
                scores_ood = monitor.predict(features_ood[0], pred_ood)

                precision_oms, recall_oms, f1_oms = eval_oms.get_metrics(scores_test, scores_ood)

                data = [model, layers_ids[0],
                        id_dataset, ood_dataset, str(None), str(adversarial_attack),
                        id_accuracy, ood_accuracy,
                        "GMM_bic",
                        precision_oms, recall_oms, f1_oms,
                        str([gmm.n_components for gmm in monitor.gmm])]
                writer.writerow(data)

                monitor = MahalanobisMonitor(id_dataset, model, layers_ids[0], is_tied=True)
                monitor.fit(features_train[0][pred_train == lab_train],
                            pred_train[[pred_train == lab_train]], save=True)

                scores_test = monitor.predict(features_test[0], pred_test)
                scores_ood = monitor.predict(features_ood[0], pred_ood)

                precision_oms, recall_oms, f1_oms = eval_oms.get_metrics(scores_test, scores_ood)

                data = [model, layers_ids[0],
                        id_dataset, ood_dataset, str(None), str(adversarial_attack),
                        id_accuracy, ood_accuracy,
                        "mahalanobis",
                        precision_oms, recall_oms, f1_oms,
                        str(None)]
                writer.writerow(data)

                monitor = OutsideTheBoxMonitor(n_clusters=10)
                monitor.fit(features_train[0][pred_train == lab_train],
                            pred_train[pred_train == lab_train])

                scores_test = monitor.predict(features_test[0], pred_test)
                scores_ood = monitor.predict(features_ood[0], pred_ood)

                precision_oms, recall_oms, f1_oms = eval_oms.get_metrics(scores_test, scores_ood)

                data = [model, layers_ids[0],
                        id_dataset, ood_dataset, str(None), str(adversarial_attack),
                        id_accuracy, ood_accuracy,
                        "oob",
                        precision_oms, recall_oms, f1_oms,
                        str(None)]
                writer.writerow(data)
                f.close()
