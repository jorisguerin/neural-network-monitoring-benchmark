from dataset import Dataset
from Monitors import *
from evaluator import Evaluator
from feature_extractor import FeatureExtractor

import os
import csv
import time
from sklearn.metrics import accuracy_score


EVAL_CONFIG_TEMPLATE = {
    'network': 'densenet',
    'network_layers': [98],
    'dataset': 'cifar10',
    'dataset_ood': 'cifar100',
    'monitor': None,
    'monitor_train_params': None,
    'monitor_infer_params': None,
    'perturbation': None,
    'adver_attack': None,
    'batch_size': 100,
    'TORCH_DEVICE': None,
    'evaluation_mode': 'oms',
    'evaluation_metrics': [],
    'is_train_timed': True,
    'is_infer_timed': True,
}


def evaluate_monitor(config):
    """
    """
    dataset_train = Dataset(
        config['dataset'], 
        "train", 
        config['network'], 
        batch_size=config['batch_size']
    )
    dataset_test = Dataset(
        config['dataset'], 
        "test", 
        config['network'], 
        batch_size=config['batch_size']
    )
    dataset_ood = Dataset(
        config['dataset_ood'], 
        "test", 
        config['network'], 
        config['perturbation'], 
        config['adver_attack'], 
        batch_size=config['batch_size']
    )

    feature_extractor = FeatureExtractor(config['network'], config['dataset'], config['network_layers'], config['TORCH_DEVICE'])

    (features_train, logits_train, softmax_train, 
     preds_train, labels_train) = feature_extractor.get_features(dataset_train)
    (features_test, logits_test, softmax_test,
     preds_test, labels_test) = feature_extractor.get_features(dataset_test)
    (features_ood, logits_ood, softmax_ood,
     preds_ood, labels_ood) = feature_extractor.get_features(dataset_ood)
    
    evaluator = Evaluator(config['evaluation_mode'], is_novelty=(config['dataset'] != config['dataset_ood']))
    evaluator.fit_ground_truth(labels_test, labels_ood, preds_test, preds_ood)

    monitor = config['monitor']
    results = {}

    if config['is_train_timed']:
        t0_train = time.time()
    if config['monitor_train_params']:
        monitor.fit(features_train[0], preds_train, labels_train)
    else:
        monitor.fit()
    if config['is_train_timed']:
        t1_train = time.time()

    if config['is_infer_timed']:
        t0_infer = time.time()
    match config['monitor_infer_params']:
        case 'features':
            scores_test = monitor.predict(features_test[0], preds_test)
            scores_ood  = monitor.predict(features_ood[0], preds_ood)
        case 'softmax':
            scores_test = monitor.predict(softmax_test)
            scores_ood  = monitor.predict(softmax_ood)
        case 'logits':
            scores_test = monitor.predict(logits_test)
            scores_ood  = monitor.predict(logits_ood)
        case _:
            scores_test = None
            scores_ood  = None
    if config['is_infer_timed']:
        t1_infer = time.time()

    if 'aupr_score' in config['evaluation_metrics']:
        results['aupr_score'] = evaluator.get_aupr_score(scores_test, scores_ood)
    if 'auroc_score' in config['evaluation_metrics']:
        results['auroc_score'] = evaluator.get_auroc_score(scores_test, scores_ood)
    if 'tnr_frac_tpr' in config['evaluation_metrics']:
        results['tnr_frac_tpr'] = evaluator.get_tnr_frac_tpr(scores_test, scores_ood)  
    if 'f1opt_scores' in config['evaluation_metrics']:
        prec, recall, f1 = evaluator.get_metrics_at_f1_opt(scores_test, scores_ood)
        results['f1'] = f1
        results['prec'] = prec
        results['recall'] = recall

    if config['is_train_timed']:
        train_timing = t1_train - t0_train
        results['train_time'] = train_timing
    if config['is_infer_timed']:
        infer_timing = t1_infer - t0_infer
        results['infer_time'] = infer_timing

    return results