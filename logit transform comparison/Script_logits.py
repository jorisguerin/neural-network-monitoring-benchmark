from dataset import Dataset
from feature_extractor import *
from monitors_logits import *
from evaluation import Evaluator
import torch
from models import *
import pandas as pd
import os
import csv


batch_size = 10
device_name = 'cuda:0' if torch.cuda.is_available() else 'cpu'

#Data  id
all_id_datasets = ["cifar10", "svhn", "cifar100"]

all_ood_dataset = [["cifar100", "svhn", "lsun"],
               ["cifar10", "tiny_imagenet", "lsun"],
               ["cifar10", "svhn", "lsun"]]

all_perturbations = ["brightness", "blur", "pixelization"]

all_attacks = ["fgsm", "deepfool", "pgd"]

all_models = ["resnet", "densenet"]

all_layers_ids = [[32], [98]]
header = ["Model", "Layer", "ID dataset", "OOD dataset", "Perturbation", "Attack",
          "Monitor", "AUPR", "AUROC", "TNR@95TPR"]

save_results_path = "Results/ICCV/"
if not os.path.exists(save_results_path):
    os.makedirs(save_results_path)
f_novelty1 = open(save_results_path + "Expe_score_test_novelty1.csv", "w", encoding="UTF8")
writer_novelty1 = csv.writer(f_novelty1, delimiter =',')
writer_novelty1.writerow(header)

save_results_path = "Results/ICCV/"
if not os.path.exists(save_results_path):
    os.makedirs(save_results_path)
f_attack1 = open(save_results_path + "Expe_score_test_attack1.csv", "w", encoding="UTF8")
writer_attack1 = csv.writer(f_attack1, delimiter =',')
writer_attack1.writerow(header)


save_results_path = "Results/ICCV/"
if not os.path.exists(save_results_path):
    os.makedirs(save_results_path)
f_pertu1 = open(save_results_path + "Expe_score_test_pertu1.csv", "w", encoding="UTF8")
writer_pertu1 = csv.writer(f_pertu1, delimiter =',')
writer_pertu1.writerow(header)

for h in range(len(all_models)):
    model = all_models[h]
    for layer in all_layers_ids[h]:
        for n in range(len(all_id_datasets)):
            id_dataset = all_id_datasets[n]
            for i in range(len(all_ood_datasets)):
                ood_dataset = all_ood_datasets[i]

                dataset_train = Dataset(id_dataset, "train", model, batch_size=batch_size)
                dataset_test = Dataset(id_dataset, "test", model, batch_size=batch_size)
                dataset_ood = Dataset(ood_dataset, "test", model, None, None, batch_size=batch_size)

                feature_extractor = FeatureExtractor(model, id_dataset, [layer], device_name)
                features_train, logits_train, softmax_train, pred_train, lab_train = feature_extractor.get_features(dataset_train)
                features_test, logits_test, softmax_test, pred_test, lab_test = feature_extractor.get_features(dataset_test)
                features_ood, logits_ood, softmax_ood, pred_ood, lab_ood = feature_extractor.get_features(dataset_ood)

                eval_oms = Evaluator("oms", is_novelty=(id_dataset != ood_dataset))
                eval_oms.fit_ground_truth(lab_test, lab_ood, pred_test, pred_ood)


                # Moniteur MaxSoftmaxProbability
                monitor_max_softmax = MaxSoftmaxProbabilityMonitor()
                monitor_max_softmax.fit()

                scores_test_max_softmax = monitor_max_softmax.predict(softmax_test)
                scores_ood_max_softmax = monitor_max_softmax.predict(softmax_ood)

                aupr_max_softmax = eval_oms.get_average_precision(scores_test_max_softmax, scores_ood_max_softmax)
                auroc_max_softmax = eval_oms.get_auroc(scores_test_max_softmax, scores_ood_max_softmax)
                tnr95tpr_max_softmax = eval_oms.get_tnr_frac_tpr_oms(scores_test_max_softmax, scores_ood_max_softmax, frac=0.95)

                data = [model, layer, id_dataset, ood_dataset, str(None), str(None), "MaxSoftmaxProbability", aupr_max_softmax, auroc_max_softmax, tnr95tpr_max_softmax]
                writer_novelty1.writerow(data)


                # Moniteur Maxlogits
                monitor = MaxLogitMonitor()
                monitor.fit()

                scores_test_maxlogits = monitor.predict(logits_test)
                scores_ood_maxlogits= monitor.predict(logits_ood)

                aupr_maxlogits = eval_oms.get_average_precision(scores_test_maxlogits, scores_ood_maxlogits)
                auroc_maxlogits = eval_oms.get_auroc(scores_test_maxlogits, scores_ood_maxlogits)
                tnr95tpr_maxlogits = eval_oms.get_tnr_frac_tpr_oms(scores_test_maxlogits, scores_ood_maxlogits, frac=0.95)

                data = [model, layer, id_dataset, ood_dataset, str(None), str(None), "Maxlogits", aupr_maxlogits, auroc_maxlogits, tnr95tpr_maxlogits]
                writer_novelty1.writerow(data)
                
                
                # Moniteur EnergyMonitor
                monitor_energy = EnergyMonitor()
                monitor_energy.fit()

                scores_test_energy = monitor_energy.predict(logits_test)
                scores_ood_energy = monitor_energy.predict(logits_ood)

                aupr_energy = eval_oms.get_average_precision(scores_test_energy, scores_ood_energy)
                auroc_energy = eval_oms.get_auroc(scores_test_energy, scores_ood_energy)
                tnr95tpr_energy = eval_oms.get_tnr_frac_tpr_oms(scores_test_energy, scores_ood_energy, frac=0.95)

                data = [model, layer, id_dataset, ood_dataset, str(None), str(None), "EnergyMonitor", aupr_energy, auroc_energy, tnr95tpr_energy]
                writer_novelty1.writerow(data)

           


                # Moniteur ODIN

                monitor_ODIN = ODINMonitor()
                monitor_ODIN.fit()

                scores_test_ODIN = monitor_ODIN.predict(logits_test)
                scores_ood_ODIN= monitor_ODIN.predict(logits_ood)

                aupr_ODIN = eval_oms.get_average_precision(scores_test_ODIN, scores_ood_ODIN)
                auroc_ODIN = eval_oms.get_auroc(scores_test_ODIN, scores_ood_ODIN)
                tnr95tpr_ODIN = eval_oms.get_tnr_frac_tpr_oms(scores_test_ODIN, scores_ood_ODIN, frac=0.95)

                data = [model, layer, id_dataset, ood_dataset, str(None), str(None), "ODINMonitor", aupr_ODIN, auroc_ODIN, tnr95tpr_ODIN]
                writer_novelty1.writerow(data)

                


                # Moniteur DOCTOR mode="alpha"
                monitor_doctor_alpha = Doctor(mode="alpha")
                monitor_doctor_alpha.fit()

                scores_test_doctor_alpha = monitor_doctor_alpha.predict(softmax_test)
                scores_ood_doctor_alpha = monitor_doctor_alpha.predict(softmax_ood)

                aupr_doctor_alpha = eval_oms.get_average_precision(scores_test_doctor_alpha, scores_ood_doctor_alpha)
                auroc_doctor_alpha = eval_oms.get_auroc(scores_test_doctor_alpha, scores_ood_doctor_alpha)
                tnr95tpr_doctor_alpha = eval_oms.get_tnr_frac_tpr_oms(scores_test_doctor_alpha, scores_ood_doctor_alpha, frac=0.95)

                data = [model, layer, id_dataset, ood_dataset, str(None), str(None), "DOCTOR_alpha", aupr_doctor_alpha, auroc_doctor_alpha, tnr95tpr_doctor_alpha]
                writer_novelty1.writerow(data)
                
                

                # Moniteur DOCTOR mode="beta"
                monitor_doctor = Doctor(mode="beta")
                monitor_doctor.fit()

                scores_test_doctor = monitor_doctor.predict(softmax_test)
                scores_ood_doctor = monitor_doctor.predict(softmax_ood)

                aupr_doctor = eval_oms.get_average_precision(scores_test_doctor, scores_ood_doctor)
                auroc_doctor = eval_oms.get_auroc(scores_test_doctor, scores_ood_doctor)
                tnr95tpr_doctor = eval_oms.get_tnr_frac_tpr_oms(scores_test_doctor, scores_ood_doctor, frac=0.95)

                data = [model, layer, id_dataset, ood_dataset, str(None), str(None), "DOCTOR_beta", aupr_doctor, auroc_doctor, tnr95tpr_doctor]
                writer_novelty1.writerow(data)




                #moniteur react mode msp

                monitor_react = ReActMonitor(quantile_value=0.9, mode="msp")
                monitor_react.fit(feature_extractor, features_train[-1])

                scores_test_react = monitor_react.predict(features_test[-1])
                scores_ood_react = monitor_react.predict(features_ood[-1])
                
                aupr_react = eval_oms.get_average_precision(scores_test_react, scores_ood_react)
                auroc_react = eval_oms.get_auroc(scores_test_react, scores_ood_react)
                tnr95tpr_react = eval_oms.get_tnr_frac_tpr_oms(scores_test_react, scores_ood_react, frac=0.95)

                data = [model, layer, id_dataset, ood_dataset, str(None), str(None), "ReAct mode(MSP) ", aupr_react, auroc_react, tnr95tpr_react]
                writer_novelty1.writerow(data)
                
                
                
                
                # Moniteur ReActMonitor mode maxlogits
                monitor_react = ReActMonitor(quantile_value=0.9, mode="maxlogits")
                monitor_react.fit(feature_extractor, features_train[-1])

                scores_test_react = monitor_react.predict(features_test[-1])
                scores_ood_react = monitor_react.predict(features_ood[-1])

                aupr_react = eval_oms.get_average_precision(scores_test_react, scores_ood_react)
                auroc_react = eval_oms.get_auroc(scores_test_react, scores_ood_react)
                tnr95tpr_react = eval_oms.get_tnr_frac_tpr_oms(scores_test_react, scores_ood_react, frac=0.95)

                
                data = [model, layer, id_dataset, ood_dataset, str(None), str(None), "ReAct mode(Maxlogits) ", aupr_react, auroc_react, tnr95tpr_react]
                writer_novelty1.writerow(data)
                
                
                                

                # Moniteur ReActMonitor mode energy
                monitor = ReActMonitor(quantile_value=0.9, mode = "energy")
                monitor.fit(feature_extractor, features_train[-1])

                scores_test = monitor.predict(features_test[-1])
                scores_ood = monitor.predict(features_ood[-1])

                aupr = eval_oms.get_average_precision(scores_test, scores_ood)
                auroc = eval_oms.get_auroc(scores_test, scores_ood)
                tnr95tpr = eval_oms.get_tnr_frac_tpr_oms(scores_test, scores_ood, frac=0.95)


                data = [model, layer, id_dataset, ood_dataset, str(None), str(None), "ReAct mode(Energy)", aupr_react, auroc_react, tnr95tpr_react]
                writer_novelty1.writerow(data)



                # Moniteur ReActMonitor mode "ODIN"

                monitor_react_odin = ReActMonitor(quantile_value=0.9, mode = "ODIN")
                monitor_react_odin.fit(feature_extractor, features_train[-1])

                scores_test_react_odin = monitor_react_odin.predict(features_test[-1])
                scores_ood_react_odin = monitor_react_odin.predict(features_ood[-1])

                aupr_react_odin = eval_oms.get_average_precision(scores_test_react_odin, scores_ood_react_odin)
                auroc_react_odin = eval_oms.get_auroc(scores_test_react_odin, scores_ood_react_odin)
                tnr95tpr_react_odin = eval_oms.get_tnr_frac_tpr_oms(scores_test_react_odin, scores_ood_react_odin, frac=0.95)


                data = [model, layer, id_dataset, ood_dataset, str(None), str(None), "ReAct mode(ODIN)", aupr_react_odin, auroc_react_odin, tnr95tpr_react_odin]
                writer_novelty1.writerow(data)


                # Moniteur ReActMonitor mode "doctor alpha"

                monitor_react_doctor = ReActMonitor(quantile_value=0.9, mode = "doctor alpha")
                monitor_react_doctor.fit(feature_extractor, features_train[-1])

                scores_test_react_doctor = monitor_react_doctor.predict(features_test[-1])
                scores_ood_react_doctor = monitor_react_doctor.predict(features_ood[-1])

                aupr_react_doctor = eval_oms.get_average_precision(scores_test_react_doctor, scores_ood_react_doctor)
                auroc_react_doctor = eval_oms.get_auroc(scores_test_react_doctor, scores_ood_react_doctor)
                tnr95tpr_react_doctor = eval_oms.get_tnr_frac_tpr_oms(scores_test_react_doctor, scores_ood_react_doctor, frac=0.95)


                data = [model, layer, id_dataset, ood_dataset, str(None), str(None), "ReAct mode(DOCTOR)", aupr_react_doctor, auroc_react_doctor, tnr95tpr_react_doctor]
                writer_novelty1.writerow(data)
                    


    
            for k in range(len(all_attacks)):
                ood_dataset = id_dataset

                adversarial_attack = all_attacks[k]


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



                # Moniteur MSP
                monitor = MaxSoftmaxProbabilityMonitor()
                monitor.fit()

                scores_test = monitor.predict(softmax_test)
                scores_ood = monitor.predict(softmax_ood)

                aupr = eval_oms.get_average_precision(scores_test, scores_ood)
                auroc = eval_oms.get_auroc(scores_test, scores_ood)
                tnr95tpr = eval_oms.get_tnr_frac_tpr_oms(scores_test, scores_ood, frac=0.95)

                data = [model, layer, id_dataset, ood_dataset, str(None), str(adversarial_attack), "MSP",  aupr, auroc, tnr95tpr]
                writer_attack1.writerow(data)

                
                # Moniteur Maxlogits
                monitor = MaxLogitMonitor()
                monitor.fit()

                scores_test_maxlogits = monitor.predict(logits_test)
                scores_ood_maxlogits= monitor.predict(logits_ood)

                aupr_maxlogits = eval_oms.get_average_precision(scores_test_maxlogits, scores_ood_maxlogits)
                auroc_maxlogits = eval_oms.get_auroc(scores_test_maxlogits, scores_ood_maxlogits)
                tnr95tpr_maxlogits = eval_oms.get_tnr_frac_tpr_oms(scores_test_maxlogits, scores_ood_maxlogits, frac=0.95)

                data = [model, layer, id_dataset, ood_dataset, str(None), str(adversarial_attack), "Maxlogits", aupr_maxlogits, auroc_maxlogits, tnr95tpr_maxlogits]
                writer_attack1.writerow(data)
                
      
                

                # Moniteur EnergyMonitor
                monitor_energy = EnergyMonitor()
                monitor_energy.fit()

                scores_test_energy = monitor_energy.predict(logits_test)
                scores_ood_energy = monitor_energy.predict(logits_ood)

                aupr_energy = eval_oms.get_average_precision(scores_test_energy, scores_ood_energy)
                auroc_energy = eval_oms.get_auroc(scores_test_energy, scores_ood_energy)
                tnr95tpr_energy = eval_oms.get_tnr_frac_tpr_oms(scores_test_energy, scores_ood_energy, frac=0.95)

                data = [model, layer, id_dataset, ood_dataset, str(None), str(adversarial_attack), "EnergyMonitor", aupr_energy, auroc_energy, tnr95tpr_energy]
                writer_attack1.writerow(data)

                # Moniteur ODIN
                monitor_ODIN = ODINMonitor()
                monitor_ODIN.fit()

                scores_test_ODIN = monitor_ODIN.predict(logits_test)
                scores_ood_ODIN= monitor_ODIN.predict(logits_ood)

                aupr_ODIN = eval_oms.get_average_precision(scores_test_ODIN, scores_ood_ODIN)
                auroc_ODIN = eval_oms.get_auroc(scores_test_ODIN, scores_ood_ODIN)
                tnr95tpr_ODIN = eval_oms.get_tnr_frac_tpr_oms(scores_test_ODIN, scores_ood_ODIN, frac=0.95)

                data = [model, layer, id_dataset, ood_dataset, str(None), str(adversarial_attack), "ODINMonitor", aupr_ODIN, auroc_ODIN, tnr95tpr_ODIN]
                writer_attack1.writerow(data)



                # Moniteur DOCTOR mode="alpha"
                monitor_doctor_alpha = Doctor(mode="alpha")
                monitor_doctor_alpha.fit()

                scores_test_doctor_alpha = monitor_doctor_alpha.predict(softmax_test)
                scores_ood_doctor_alpha = monitor_doctor_alpha.predict(softmax_ood)

                aupr_doctor_alpha = eval_oms.get_average_precision(scores_test_doctor_alpha, scores_ood_doctor_alpha)
                auroc_doctor_alpha = eval_oms.get_auroc(scores_test_doctor_alpha, scores_ood_doctor_alpha)
                tnr95tpr_doctor_alpha = eval_oms.get_tnr_frac_tpr_oms(scores_test_doctor_alpha, scores_ood_doctor_alpha, frac=0.95)

                data = [model, layer, id_dataset, ood_dataset, str(None), str(adversarial_attack), "DOCTOR_alpha", aupr_doctor_alpha, auroc_doctor_alpha, tnr95tpr_doctor_alpha]
                writer_attack1.writerow(data)


                # Moniteur DOCTOR mode="beta"
                monitor_doctor = Doctor(mode="beta")
                monitor_doctor.fit()

                scores_test_doctor = monitor_doctor.predict(softmax_test)
                scores_ood_doctor = monitor_doctor.predict(softmax_ood)

                aupr_doctor = eval_oms.get_average_precision(scores_test_doctor, scores_ood_doctor)
                auroc_doctor = eval_oms.get_auroc(scores_test_doctor, scores_ood_doctor)
                tnr95tpr_doctor = eval_oms.get_tnr_frac_tpr_oms(scores_test_doctor, scores_ood_doctor, frac=0.95)

                data = [model, layer, id_dataset, ood_dataset, str(None), str(adversarial_attack), "DOCTOR_beta",  aupr_doctor, auroc_doctor, tnr95tpr_doctor]
                writer_attack1.writerow(data)


               
                # Moniteur ReActMonitor MSP
                monitor_react = ReActMonitor(quantile_value=0.9, mode="msp")
                monitor_react.fit(feature_extractor, features_train[-1])

                scores_test_react = monitor_react.predict(features_test[-1])
                scores_ood_react = monitor_react.predict(features_ood[-1])

                aupr_react = eval_oms.get_average_precision(scores_test_react, scores_ood_react)
                auroc_react = eval_oms.get_auroc(scores_test_react, scores_ood_react)
                tnr95tpr_react = eval_oms.get_tnr_frac_tpr_oms(scores_test_react, scores_ood_react, frac=0.95)

                data = [model, layer, id_dataset, ood_dataset, str(None), str(adversarial_attack), "ReAct mode(MSP)", aupr_react, auroc_react, tnr95tpr_react]
                writer_attack1.writerow(data)
                                
                # Moniteur ReActMonitor mode maxlogits
                monitor_react = ReActMonitor(quantile_value=0.9, mode="maxlogits")
                monitor_react.fit(feature_extractor, features_train[-1])

                scores_test_react = monitor_react.predict(features_test[-1])
                scores_ood_react = monitor_react.predict(features_ood[-1])

                aupr_react = eval_oms.get_average_precision(scores_test_react, scores_ood_react)
                auroc_react = eval_oms.get_auroc(scores_test_react, scores_ood_react)
                tnr95tpr_react = eval_oms.get_tnr_frac_tpr_oms(scores_test_react, scores_ood_react, frac=0.95)

                
                data = [model, layer, id_dataset, ood_dataset, str(None), str(adversarial_attack), "ReAct mode(Maxlogits) ", aupr_react, auroc_react, tnr95tpr_react]
                writer_attack1.writerow(data)


                # Moniteur ReActMonitor mode energy

                monitor = ReActMonitor(quantile_value=0.9, mode = "energy")
                monitor.fit(feature_extractor, features_train[-1])

                scores_test = monitor.predict(features_test[-1])
                scores_ood = monitor.predict(features_ood[-1])

                aupr = eval_oms.get_average_precision(scores_test, scores_ood)
                auroc = eval_oms.get_auroc(scores_test, scores_ood)
                tnr95tpr = eval_oms.get_tnr_frac_tpr_oms(scores_test, scores_ood, frac=0.95)

                data = [model, layer, id_dataset, ood_dataset, str(None), str(adversarial_attack), "ReAct mode(Energy)", aupr, auroc, tnr95tpr]
                writer_attack1.writerow(data) 


                # Moniteur ReActMonitor mode "ODIN"

                monitor_react_odin = ReActMonitor(quantile_value=0.9, mode = "ODIN")
                monitor_react_odin.fit(feature_extractor, features_train[-1])

                scores_test_react_odin = monitor_react_odin.predict(features_test[-1])
                scores_ood_react_odin = monitor_react_odin.predict(features_ood[-1])

                aupr_react_odin = eval_oms.get_average_precision(scores_test_react_odin, scores_ood_react_odin)
                auroc_react_odin = eval_oms.get_auroc(scores_test_react_odin, scores_ood_react_odin)
                tnr95tpr_react_odin = eval_oms.get_tnr_frac_tpr_oms(scores_test_react_odin, scores_ood_react_odin, frac=0.95)


                data = [model, layer, id_dataset, ood_dataset, str(None), str(adversarial_attack), "ReAct mode(ODIN)", aupr_react_odin, auroc_react_odin, tnr95tpr_react_odin]
                writer_attack1.writerow(data)


                # Moniteur ReActMonitor mode "doctor alpha"

                monitor_react_doctor = ReActMonitor(quantile_value=0.9, mode = "doctor alpha")
                monitor_react_doctor.fit(feature_extractor, features_train[-1])

                scores_test_react_doctor = monitor_react_doctor.predict(features_test[-1])
                scores_ood_react_doctor = monitor_react_doctor.predict(features_ood[-1])

                aupr_react_doctor = eval_oms.get_average_precision(scores_test_react_doctor, scores_ood_react_doctor)
                auroc_react_doctor = eval_oms.get_auroc(scores_test_react_doctor, scores_ood_react_doctor)
                tnr95tpr_react_doctor = eval_oms.get_tnr_frac_tpr_oms(scores_test_react_doctor, scores_ood_react_doctor, frac=0.95)


                data = [model, layer, id_dataset, ood_dataset, str(None), str(adversarial_attack), "ReAct mode(DOCTOR)", aupr_react_doctor, auroc_react_doctor, tnr95tpr_react_doctor]
                writer_attack1.writerow(data)



            for j in range(len(all_perturbations)):
                ood_dataset = id_dataset

                additional_transform = all_perturbations[j]


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


                # Moniteur MSP
                monitor = MaxSoftmaxProbabilityMonitor()
                monitor.fit()

                scores_test = monitor.predict(softmax_test)
                scores_ood = monitor.predict(softmax_ood)

                aupr = eval_oms.get_average_precision(scores_test, scores_ood)
                auroc = eval_oms.get_auroc(scores_test, scores_ood)
                tnr95tpr = eval_oms.get_tnr_frac_tpr_oms(scores_test, scores_ood, frac=0.95)

                data = [model, layer, id_dataset, ood_dataset, str(additional_transform), str(None), "MSP", aupr, auroc, tnr95tpr]
                writer_pertu1.writerow(data)
                

                # Moniteur Maxlogits
                monitor = MaxLogitMonitor()
                monitor.fit()

                scores_test_maxlogits = monitor.predict(logits_test)
                scores_ood_maxlogits= monitor.predict(logits_ood)

                aupr_maxlogits = eval_oms.get_average_precision(scores_test_maxlogits, scores_ood_maxlogits)
                auroc_maxlogits = eval_oms.get_auroc(scores_test_maxlogits, scores_ood_maxlogits)
                tnr95tpr_maxlogits = eval_oms.get_tnr_frac_tpr_oms(scores_test_maxlogits, scores_ood_maxlogits, frac=0.95)

                data = [model, layer, id_dataset, ood_dataset, str(additional_transform), str(None), "Maxlogits", aupr_maxlogits, auroc_maxlogits, tnr95tpr_maxlogits]
                writer_pertu1.writerow(data)
                


               
                # Moniteur EnergyMonitor
                monitor_energy = EnergyMonitor(temperature=1)
                monitor_energy.fit()

                scores_test_energy = monitor_energy.predict(logits_test)
                scores_ood_energy = monitor_energy.predict(logits_ood)

                aupr_energy = eval_oms.get_average_precision(scores_test_energy, scores_ood_energy)
                auroc_energy = eval_oms.get_auroc(scores_test_energy, scores_ood_energy)
                tnr95tpr_energy = eval_oms.get_tnr_frac_tpr_oms(scores_test_energy, scores_ood_energy, frac=0.95)

                data = [model, layer, id_dataset, ood_dataset, str(additional_transform), str(None), "EnergyMonitor", aupr_energy, auroc_energy, tnr95tpr_energy]
                writer_pertu1.writerow(data)

                # Moniteur ODIN
                monitor_ODIN = ODINMonitor(temperature=100)
                monitor_ODIN.fit()

                scores_test_ODIN = monitor_ODIN.predict(logits_test)
                scores_ood_ODIN= monitor_ODIN.predict(logits_ood)

                aupr_ODIN = eval_oms.get_average_precision(scores_test_ODIN, scores_ood_ODIN)
                auroc_ODIN = eval_oms.get_auroc(scores_test_ODIN, scores_ood_ODIN)
                tnr95tpr_ODIN = eval_oms.get_tnr_frac_tpr_oms(scores_test_ODIN, scores_ood_ODIN, frac=0.95)

                data = [model, layer, id_dataset, ood_dataset, str(additional_transform), str(None), "ODINMonitor", aupr_ODIN, auroc_ODIN, tnr95tpr_ODIN]
                writer_pertu1.writerow(data)


                # Moniteur DOCTOR mode="alpha"
                monitor_doctor_alpha = Doctor(mode="alpha")
                monitor_doctor_alpha.fit()

                scores_test_doctor_alpha = monitor_doctor_alpha.predict(softmax_test)
                scores_ood_doctor_alpha = monitor_doctor_alpha.predict(softmax_ood)

                aupr_doctor_alpha = eval_oms.get_average_precision(scores_test_doctor_alpha, scores_ood_doctor_alpha)
                auroc_doctor_alpha = eval_oms.get_auroc(scores_test_doctor_alpha, scores_ood_doctor_alpha)
                tnr95tpr_doctor_alpha = eval_oms.get_tnr_frac_tpr_oms(scores_test_doctor_alpha, scores_ood_doctor_alpha, frac=0.95)

                data = [model, layer, id_dataset, ood_dataset, str(additional_transform), str(None), "DOCTOR_alpha", aupr_doctor_alpha, auroc_doctor_alpha, tnr95tpr_doctor_alpha]
                writer_pertu1.writerow(data)


                # Moniteur DOCTOR mode="beta"
                monitor_doctor = Doctor(mode="beta")
                monitor_doctor.fit()

                scores_test_doctor = monitor_doctor.predict(softmax_test)
                scores_ood_doctor = monitor_doctor.predict(softmax_ood)

                aupr_doctor = eval_oms.get_average_precision(scores_test_doctor, scores_ood_doctor)
                auroc_doctor = eval_oms.get_auroc(scores_test_doctor, scores_ood_doctor)
                tnr95tpr_doctor = eval_oms.get_tnr_frac_tpr_oms(scores_test_doctor, scores_ood_doctor, frac=0.95)

                data = [model, layer, id_dataset, ood_dataset, str(additional_transform), str(None), "DOCTOR_beta", aupr_doctor, auroc_doctor, tnr95tpr_doctor]
                writer_pertu1.writerow(data)


                

                # Moniteur ReActMonitor MSP
                monitor_react = ReActMonitor(quantile_value=0.9, mode="msp")
                monitor_react.fit(feature_extractor, features_train[-1])

                scores_test_react = monitor_react.predict(features_test[-1])
                scores_ood_react = monitor_react.predict(features_ood[-1])

                aupr_react = eval_oms.get_average_precision(scores_test_react, scores_ood_react)
                auroc_react = eval_oms.get_auroc(scores_test_react, scores_ood_react)
                tnr95tpr_react = eval_oms.get_tnr_frac_tpr_oms(scores_test_react, scores_ood_react, frac=0.95)

                data = [model, layer, id_dataset, ood_dataset, str(additional_transform), str(None), "ReAct mode(MSP)", aupr_react, auroc_react, tnr95tpr_react]
                writer_pertu1.writerow(data)
                                
                # Moniteur ReActMonitor mode maxlogits
                monitor_react = ReActMonitor(quantile_value=clip, mode="maxlogits")
                monitor_react.fit(feature_extractor, features_train[-1])

                scores_test_react = monitor_react.predict(features_test[-1])
                scores_ood_react = monitor_react.predict(features_ood[-1])

                aupr_react = eval_oms.get_average_precision(scores_test_react, scores_ood_react)
                auroc_react = eval_oms.get_auroc(scores_test_react, scores_ood_react)
                tnr95tpr_react = eval_oms.get_tnr_frac_tpr_oms(scores_test_react, scores_ood_react, frac=0.95)

                
                data = [model, layer, id_dataset, ood_dataset, str(additional_transform), str(None), "ReAct mode(Maxlogits) ", aupr_react, auroc_react, tnr95tpr_react]
                writer_pertu1.writerow(data)


                # Moniteur ReActMonitor mode energy
                monitor = ReActMonitor(quantile_value=0.9, mode = "energy")
                monitor.fit(feature_extractor, features_train[-1])

                scores_test = monitor.predict(features_test[-1])
                scores_ood = monitor.predict(features_ood[-1])

                aupr = eval_oms.get_average_precision(scores_test, scores_ood)
                auroc = eval_oms.get_auroc(scores_test, scores_ood)
                tnr95tpr = eval_oms.get_tnr_frac_tpr_oms(scores_test, scores_ood, frac=0.95)

                data = [model, layer, id_dataset, ood_dataset, str(additional_transform), str(None), "ReAct mode(Energy)", aupr, auroc, tnr95tpr]
                writer_pertu1.writerow(data)


                # Moniteur ReActMonitor mode "ODIN"

                monitor_react_odin = ReActMonitor(quantile_value=0.9, mode = "ODIN")
                monitor_react_odin.fit(feature_extractor, features_train[-1])

                scores_test_react_odin = monitor_react_odin.predict(features_test[-1])
                scores_ood_react_odin = monitor_react_odin.predict(features_ood[-1])

                aupr_react_odin = eval_oms.get_average_precision(scores_test_react_odin, scores_ood_react_odin)
                auroc_react_odin = eval_oms.get_auroc(scores_test_react_odin, scores_ood_react_odin)
                tnr95tpr_react_odin = eval_oms.get_tnr_frac_tpr_oms(scores_test_react_odin, scores_ood_react_odin, frac=0.95)


                data = [model, layer, id_dataset, ood_dataset, str(additional_transform), str(None), "ReAct mode(ODIN)", aupr_react_odin, auroc_react_odin, tnr95tpr_react_odin]
                writer_pertu1.writerow(data)


                # Moniteur ReActMonitor mode "doctor alpha"

                monitor_react_doctor = ReActMonitor(quantile_value=0.9, mode = "doctor alpha")
                monitor_react_doctor.fit(feature_extractor, features_train[-1])

                scores_test_react_doctor = monitor_react_doctor.predict(features_test[-1])
                scores_ood_react_doctor = monitor_react_doctor.predict(features_ood[-1])

                aupr_react_doctor = eval_oms.get_average_precision(scores_test_react_doctor, scores_ood_react_doctor)
                auroc_react_doctor = eval_oms.get_auroc(scores_test_react_doctor, scores_ood_react_doctor)
                tnr95tpr_react_doctor = eval_oms.get_tnr_frac_tpr_oms(scores_test_react_doctor, scores_ood_react_doctor, frac=0.95)


                data = [model, layer, id_dataset, ood_dataset, str(additional_transform), str(None), "ReAct mode(DOCTOR)", aupr_react_doctor, auroc_react_doctor, tnr95tpr_react_doctor]
                writer_pertu1.writerow(data)

f_novelty1.close()
f_attack1.close()
f_pertu1.close()
