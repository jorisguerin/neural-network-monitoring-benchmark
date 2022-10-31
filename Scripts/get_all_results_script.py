import os
import csv

import torch
from sklearn.metrics import accuracy_score

from dataset import Dataset
from feature_extractor import FeatureExtractor
from evaluation import Evaluator
from monitors import MahalanobisMonitor, OutsideTheBoxMonitor, MaxSoftmaxProbabilityMonitor, \
                     EnergyMonitor, ReActMonitor

batch_size = 10
device_name = 'cuda:0' if torch.cuda.is_available() else 'cpu'

all_models = ["resnet", "densenet"]

all_layers_ids = [[0, 32],
                  [0, 98]]

all_id_datasets = ["cifar10", "svhn", "cifar100"]

all_ood_datasets = [["cifar100", "svhn", "lsun"],
                    ["cifar10", "tiny_imagenet", "lsun"],
                    ["cifar10", "svhn", "lsun"]]

all_perturbations = ["brightness", "blur", "pixelization"]

all_attacks = ["fgsm", "deepfool", "pgd"]

header = ["Model", "Layer",
          "ID dataset", "OOD dataset", "Perturbation", "Attack",
          "ID accuracy", "OOD accuracy",
          "Precision OMS@OOD*", "Recall OMS@OOD*", "F1 OMS@OOD*",
          "Monitor",
          "Precision OOD", "Recall OOD", "F1 OOD",
          "Precision OMS", "Recall OMS", "F1 OMS"]

save_results_path = "Results/"
if not os.path.exists(save_results_path):
    os.makedirs(save_results_path)
f = open(save_results_path + "full_results_onlyCorrectTrainingExamples.csv", "w", encoding="UTF8")
writer = csv.writer(f)
writer.writerow(header)

for i in range(len(all_models)):
    model = all_models[i]
    layers_ids = all_layers_ids[i]
    for j in range(len(all_id_datasets)):
        id_dataset = all_id_datasets[j]
        for k in range(len(all_ood_datasets[j])):
            ood_dataset = all_ood_datasets[j][k]

            print(model, id_dataset, ood_dataset)

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
            eval_ood = Evaluator("ood", is_novelty=(id_dataset != ood_dataset))

            eval_oms.fit_ground_truth(lab_test, lab_ood, pred_test, pred_ood)
            eval_ood.fit_ground_truth(lab_test, lab_ood, pred_test, pred_ood)

            precision_star, recall_star, f1_star = eval_oms.get_metrics(
                eval_ood.y_true[:lab_test.shape[0]].astype(bool),
                eval_ood.y_true[lab_test.shape[0]:].astype(bool))

            for lay in range(2):
                monitor = MahalanobisMonitor(id_dataset, model, lay, is_tied=True)
                monitor.fit(features_train[lay][pred_train == lab_train],
                            pred_train[[pred_train == lab_train]], save=False)

                scores_test = monitor.predict(features_test[lay], pred_test)
                scores_ood = monitor.predict(features_ood[lay], pred_ood)

                precision_ood, recall_ood, f1_ood = eval_ood.get_metrics(scores_test, scores_ood)
                precision_oms, recall_oms, f1_oms = eval_oms.get_metrics(scores_test, scores_ood)

                data = [model, lay,
                        id_dataset, ood_dataset, str(None), str(None),
                        id_accuracy, ood_accuracy,
                        precision_star, recall_star, f1_star,
                        "mahalanobis",
                        precision_ood, recall_ood, f1_ood,
                        precision_oms, recall_oms, f1_oms]
                writer.writerow(data)

                monitor = OutsideTheBoxMonitor(n_clusters=10)
                monitor.fit(features_train[lay][pred_train == lab_train],
                            pred_train[pred_train == lab_train])

                scores_test = monitor.predict(features_test[lay], pred_test)
                scores_ood = monitor.predict(features_ood[lay], pred_ood)

                precision_ood, recall_ood, f1_ood = eval_ood.get_metrics(scores_test, scores_ood)
                precision_oms, recall_oms, f1_oms = eval_oms.get_metrics(scores_test, scores_ood)

                data = [model, lay,
                        id_dataset, ood_dataset, str(None), str(None),
                        id_accuracy, ood_accuracy,
                        precision_star, recall_star, f1_star,
                        "oob",
                        precision_ood, recall_ood, f1_ood,
                        precision_oms, recall_oms, f1_oms]
                writer.writerow(data)

            monitor = MaxSoftmaxProbabilityMonitor()
            monitor.fit()

            scores_test = monitor.predict(softmax_test)
            scores_ood = monitor.predict(softmax_ood)

            precision_ood, recall_ood, f1_ood = eval_ood.get_metrics(scores_test, scores_ood)
            precision_oms, recall_oms, f1_oms = eval_oms.get_metrics(scores_test, scores_ood)

            data = [model, 1,
                    id_dataset, ood_dataset, str(None), str(None),
                    id_accuracy, ood_accuracy,
                    precision_star, recall_star, f1_star,
                    "msp",
                    precision_ood, recall_ood, f1_ood,
                    precision_oms, recall_oms, f1_oms]
            writer.writerow(data)

            monitor = ReActMonitor(quantile_value=0.99, mode="msp")
            monitor.fit(feature_extractor, features_train[-1])

            scores_test = monitor.predict(features_test[-1])
            scores_ood = monitor.predict(features_ood[-1])

            precision_ood, recall_ood, f1_ood = eval_ood.get_metrics(scores_test, scores_ood)
            precision_oms, recall_oms, f1_oms = eval_oms.get_metrics(scores_test, scores_ood)

            data = [model, 1,
                    id_dataset, ood_dataset, str(None), str(None),
                    id_accuracy, ood_accuracy,
                    precision_star, recall_star, f1_star,
                    "react_msp",
                    precision_ood, recall_ood, f1_ood,
                    precision_oms, recall_oms, f1_oms]
            writer.writerow(data)

            monitor = EnergyMonitor(temperature=1)
            monitor.fit()

            scores_test = monitor.predict(logits_test)
            scores_ood = monitor.predict(logits_ood)

            precision_ood, recall_ood, f1_ood = eval_ood.get_metrics(scores_test, scores_ood)
            precision_oms, recall_oms, f1_oms = eval_oms.get_metrics(scores_test, scores_ood)

            data = [model, 1,
                    id_dataset, ood_dataset, str(None), str(None),
                    id_accuracy, ood_accuracy,
                    precision_star, recall_star, f1_star,
                    "energy",
                    precision_ood, recall_ood, f1_ood,
                    precision_oms, recall_oms, f1_oms]
            writer.writerow(data)

            monitor = ReActMonitor(quantile_value=0.99)
            monitor.fit(feature_extractor, features_train[-1])

            scores_test = monitor.predict(features_test[-1])
            scores_ood = monitor.predict(features_ood[-1])

            precision_ood, recall_ood, f1_ood = eval_ood.get_metrics(scores_test, scores_ood)
            precision_oms, recall_oms, f1_oms = eval_oms.get_metrics(scores_test, scores_ood)

            data = [model, 1,
                    id_dataset, ood_dataset, str(None), str(None),
                    id_accuracy, ood_accuracy,
                    precision_star, recall_star, f1_star,
                    "react_energy",
                    precision_ood, recall_ood, f1_ood,
                    precision_oms, recall_oms, f1_oms]
            writer.writerow(data)

        for k in range(len(all_perturbations)):
            ood_dataset = id_dataset

            additional_transform = all_perturbations[k]

            print(model, id_dataset, additional_transform)

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
            eval_ood = Evaluator("ood", is_novelty=(id_dataset != ood_dataset))

            eval_oms.fit_ground_truth(lab_test, lab_ood, pred_test, pred_ood)
            eval_ood.fit_ground_truth(lab_test, lab_ood, pred_test, pred_ood)

            precision_star, recall_star, f1_star = eval_oms.get_metrics(
                eval_ood.y_true[:lab_test.shape[0]].astype(bool),
                eval_ood.y_true[lab_test.shape[0]:].astype(bool))

            for lay in range(2):
                monitor = MahalanobisMonitor(id_dataset, model, lay, is_tied=True)
                monitor.fit(features_train[lay][pred_train == lab_train],
                            pred_train[pred_train == lab_train], save=False)

                scores_test = monitor.predict(features_test[lay], pred_test)
                scores_ood = monitor.predict(features_ood[lay], pred_ood)

                precision_ood, recall_ood, f1_ood = eval_ood.get_metrics(scores_test, scores_ood)
                precision_oms, recall_oms, f1_oms = eval_oms.get_metrics(scores_test, scores_ood)

                data = [model, lay,
                        id_dataset, ood_dataset, str(additional_transform), str(None),
                        id_accuracy, ood_accuracy,
                        precision_star, recall_star, f1_star,
                        "mahalanobis",
                        precision_ood, recall_ood, f1_ood,
                        precision_oms, recall_oms, f1_oms]
                writer.writerow(data)

                monitor = OutsideTheBoxMonitor(n_clusters=10)
                monitor.fit(features_train[lay][pred_train == lab_train],
                            pred_train[pred_train == lab_train])

                scores_test = monitor.predict(features_test[lay], pred_test)
                scores_ood = monitor.predict(features_ood[lay], pred_ood)

                precision_ood, recall_ood, f1_ood = eval_ood.get_metrics(scores_test, scores_ood)
                precision_oms, recall_oms, f1_oms = eval_oms.get_metrics(scores_test, scores_ood)

                data = [model, lay,
                        id_dataset, ood_dataset, str(additional_transform), str(None),
                        id_accuracy, ood_accuracy,
                        precision_star, recall_star, f1_star,
                        "oob",
                        precision_ood, recall_ood, f1_ood,
                        precision_oms, recall_oms, f1_oms]
                writer.writerow(data)

            monitor = MaxSoftmaxProbabilityMonitor()
            monitor.fit()

            scores_test = monitor.predict(softmax_test)
            scores_ood = monitor.predict(softmax_ood)

            precision_ood, recall_ood, f1_ood = eval_ood.get_metrics(scores_test, scores_ood)
            precision_oms, recall_oms, f1_oms = eval_oms.get_metrics(scores_test, scores_ood)

            data = [model, 1,
                    id_dataset, ood_dataset, str(additional_transform), str(None),
                    id_accuracy, ood_accuracy,
                    precision_star, recall_star, f1_star,
                    "msp",
                    precision_ood, recall_ood, f1_ood,
                    precision_oms, recall_oms, f1_oms]
            writer.writerow(data)

            monitor = ReActMonitor(quantile_value=0.99, mode="msp")
            monitor.fit(feature_extractor, features_train[-1])

            scores_test = monitor.predict(features_test[-1])
            scores_ood = monitor.predict(features_ood[-1])

            precision_ood, recall_ood, f1_ood = eval_ood.get_metrics(scores_test, scores_ood)
            precision_oms, recall_oms, f1_oms = eval_oms.get_metrics(scores_test, scores_ood)

            data = [model, 1,
                    id_dataset, ood_dataset, str(additional_transform), str(None),
                    id_accuracy, ood_accuracy,
                    precision_star, recall_star, f1_star,
                    "react_msp",
                    precision_ood, recall_ood, f1_ood,
                    precision_oms, recall_oms, f1_oms]
            writer.writerow(data)

            monitor = EnergyMonitor(temperature=1)
            monitor.fit()

            scores_test = monitor.predict(logits_test)
            scores_ood = monitor.predict(logits_ood)

            precision_ood, recall_ood, f1_ood = eval_ood.get_metrics(scores_test, scores_ood)
            precision_oms, recall_oms, f1_oms = eval_oms.get_metrics(scores_test, scores_ood)

            data = [model, 1,
                    id_dataset, ood_dataset, str(additional_transform), str(None),
                    id_accuracy, ood_accuracy,
                    precision_star, recall_star, f1_star,
                    "energy",
                    precision_ood, recall_ood, f1_ood,
                    precision_oms, recall_oms, f1_oms]
            writer.writerow(data)

            monitor = ReActMonitor(quantile_value=0.99)
            monitor.fit(feature_extractor, features_train[-1])

            scores_test = monitor.predict(features_test[-1])
            scores_ood = monitor.predict(features_ood[-1])

            precision_ood, recall_ood, f1_ood = eval_ood.get_metrics(scores_test, scores_ood)
            precision_oms, recall_oms, f1_oms = eval_oms.get_metrics(scores_test, scores_ood)

            data = [model, 1,
                    id_dataset, ood_dataset, str(additional_transform), str(None),
                    id_accuracy, ood_accuracy,
                    precision_star, recall_star, f1_star,
                    "react_energy",
                    precision_ood, recall_ood, f1_ood,
                    precision_oms, recall_oms, f1_oms]
            writer.writerow(data)

        for k in range(len(all_attacks)):
            ood_dataset = id_dataset

            adversarial_attack = all_attacks[k]

            print(model, id_dataset, adversarial_attack)

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
            eval_ood = Evaluator("ood", is_novelty=(id_dataset != ood_dataset))

            eval_oms.fit_ground_truth(lab_test, lab_ood, pred_test, pred_ood)
            eval_ood.fit_ground_truth(lab_test, lab_ood, pred_test, pred_ood)

            precision_star, recall_star, f1_star = eval_oms.get_metrics(
                eval_ood.y_true[:lab_test.shape[0]].astype(bool),
                eval_ood.y_true[lab_test.shape[0]:].astype(bool))

            for lay in range(2):
                monitor = MahalanobisMonitor(id_dataset, model, lay, is_tied=True)
                monitor.fit(features_train[lay][pred_train == lab_train],
                            pred_train[pred_train == lab_train], save=False)

                scores_test = monitor.predict(features_test[lay], pred_test)
                scores_ood = monitor.predict(features_ood[lay], pred_ood)

                precision_ood, recall_ood, f1_ood = eval_ood.get_metrics(scores_test, scores_ood)
                precision_oms, recall_oms, f1_oms = eval_oms.get_metrics(scores_test, scores_ood)

                data = [model, lay,
                        id_dataset, ood_dataset, str(None), str(adversarial_attack),
                        id_accuracy, ood_accuracy,
                        precision_star, recall_star, f1_star,
                        "mahalanobis",
                        precision_ood, recall_ood, f1_ood,
                        precision_oms, recall_oms, f1_oms]
                writer.writerow(data)

                monitor = OutsideTheBoxMonitor(n_clusters=10)
                monitor.fit(features_train[lay][pred_train == lab_train],
                            pred_train[pred_train == lab_train])

                scores_test = monitor.predict(features_test[lay], pred_test)
                scores_ood = monitor.predict(features_ood[lay], pred_ood)

                precision_ood, recall_ood, f1_ood = eval_ood.get_metrics(scores_test, scores_ood)
                precision_oms, recall_oms, f1_oms = eval_oms.get_metrics(scores_test, scores_ood)

                data = [model, lay,
                        id_dataset, ood_dataset, str(None), str(adversarial_attack),
                        id_accuracy, ood_accuracy,
                        precision_star, recall_star, f1_star,
                        "oob",
                        precision_ood, recall_ood, f1_ood,
                        precision_oms, recall_oms, f1_oms]
                writer.writerow(data)

            monitor = MaxSoftmaxProbabilityMonitor()
            monitor.fit()

            scores_test = monitor.predict(softmax_test)
            scores_ood = monitor.predict(softmax_ood)

            precision_ood, recall_ood, f1_ood = eval_ood.get_metrics(scores_test, scores_ood)
            precision_oms, recall_oms, f1_oms = eval_oms.get_metrics(scores_test, scores_ood)

            data = [model, 1,
                    id_dataset, ood_dataset, str(None), str(adversarial_attack),
                    id_accuracy, ood_accuracy,
                    precision_star, recall_star, f1_star,
                    "msp",
                    precision_ood, recall_ood, f1_ood,
                    precision_oms, recall_oms, f1_oms]
            writer.writerow(data)

            monitor = ReActMonitor(quantile_value=0.99, mode="msp")
            monitor.fit(feature_extractor, features_train[-1])

            scores_test = monitor.predict(features_test[-1])
            scores_ood = monitor.predict(features_ood[-1])

            precision_ood, recall_ood, f1_ood = eval_ood.get_metrics(scores_test, scores_ood)
            precision_oms, recall_oms, f1_oms = eval_oms.get_metrics(scores_test, scores_ood)

            data = [model, 1,
                    id_dataset, ood_dataset, str(None), str(adversarial_attack),
                    id_accuracy, ood_accuracy,
                    precision_star, recall_star, f1_star,
                    "react_msp",
                    precision_ood, recall_ood, f1_ood,
                    precision_oms, recall_oms, f1_oms]
            writer.writerow(data)

            monitor = EnergyMonitor(temperature=1)
            monitor.fit()

            scores_test = monitor.predict(logits_test)
            scores_ood = monitor.predict(logits_ood)

            precision_ood, recall_ood, f1_ood = eval_ood.get_metrics(scores_test, scores_ood)
            precision_oms, recall_oms, f1_oms = eval_oms.get_metrics(scores_test, scores_ood)

            data = [model, 1,
                    id_dataset, ood_dataset, str(None), str(adversarial_attack),
                    id_accuracy, ood_accuracy,
                    precision_star, recall_star, f1_star,
                    "energy",
                    precision_ood, recall_ood, f1_ood,
                    precision_oms, recall_oms, f1_oms]
            writer.writerow(data)

            monitor = ReActMonitor(quantile_value=0.99)
            monitor.fit(feature_extractor, features_train[-1])

            scores_test = monitor.predict(features_test[-1])
            scores_ood = monitor.predict(features_ood[-1])

            precision_ood, recall_ood, f1_ood = eval_ood.get_metrics(scores_test, scores_ood)
            precision_oms, recall_oms, f1_oms = eval_oms.get_metrics(scores_test, scores_ood)

            data = [model, 1,
                    id_dataset, ood_dataset, str(None), str(adversarial_attack),
                    id_accuracy, ood_accuracy,
                    precision_star, recall_star, f1_star,
                    "react_energy",
                    precision_ood, recall_ood, f1_ood,
                    precision_oms, recall_oms, f1_oms]
            writer.writerow(data)

f.close()
