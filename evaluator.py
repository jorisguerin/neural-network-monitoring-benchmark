import numpy as np
import matplotlib.pyplot as plt
from numba import jit

from sklearn.metrics import roc_auc_score, average_precision_score, PrecisionRecallDisplay, RocCurveDisplay
from sklearn.metrics import precision_score, recall_score, f1_score


class Evaluator:
    def __init__(self, setting, is_novelty):
        self.setting = setting
        self._check_accepted_setting()

        self.is_novelty = is_novelty

        self.monitor_y_pred = None  # The predictions of the monitor (whether we trust or not)
        self.monitor_y_true = None  # The true labels for the monitor
        self.model_y_pred = None    # The model predictions
        self.model_y_true = None    # The model true labels

    def fit_ground_truth(
            self, 
            model_labels_id=None, 
            model_labels_ood=None, 
            model_preds_id=None, 
            model_preds_ood=None
    ):
        """
        Sets what the perfect monitor should output.
        """
        if self.setting == "ood":
            self.monitor_y_true = np.array([0] * model_labels_id.shape[0] + [1] * model_labels_ood.shape[0])
        else:
            self.model_y_true = np.concatenate([model_labels_id, model_labels_ood])
            self.model_y_pred = np.concatenate([model_preds_id, model_preds_ood])
            self.monitor_y_true = self.model_y_true != self.model_y_pred
            if self.is_novelty:
                self.monitor_y_true[model_labels_id.shape[0]:] = np.ones(model_labels_ood.shape[0])

    def get_metrics_f1opt(
            self, 
            monitor_scores_id, 
            monitor_scores_ood
    ):
        """
        Computes the metrics for optimal f1score threshold when calculated:
        - precision
        - recall
        - f1 score
        """
        self.monitor_y_pred = np.concatenate([monitor_scores_id, monitor_scores_ood])

        if self.monitor_y_pred.dtype == "bool":
            recall = recall_score(self.monitor_y_true, self.monitor_y_pred)
            prec = precision_score(self.monitor_y_true, self.monitor_y_pred)
            f1 = f1_score(self.monitor_y_true, self.monitor_y_pred)
        else:
            thresh, f1, prec, recall, _, _, _ = get_optimal_threshold_f1(self.monitor_y_pred, self.monitor_y_true)
        return prec, recall, f1

    def get_metric_aupr(self, scores_id, scores_ood):
        """
        """
        self.monitor_y_pred = np.concatenate([scores_id, scores_ood])

        if self.monitor_y_pred.dtype == "bool":
            raise ValueError("Scores must be continuous values, not booleans")
        else:
            return average_precision_score(self.monitor_y_true, self.monitor_y_pred)

    def get_metric_auroc(self, scores_id, scores_ood):
        """
        """
        self.monitor_y_pred = np.concatenate([scores_id, scores_ood])

        if self.monitor_y_pred.dtype == "bool":
            raise ValueError("Scores must be continuous values, not booleans")
        else:
            return roc_auc_score(self.monitor_y_true, self.monitor_y_pred)

    def get_metric_tnr_frac_tpr(self, scores_id, scores_ood, frac=0.95):
        """
        """
        self.monitor_y_pred = np.concatenate([scores_id, scores_ood])

        if self.monitor_y_pred.dtype == "bool":
            raise ValueError("Scores must be continuous values, not booleans")
        else:
            if self.setting == "oms":
                scores_OK = self.monitor_y_pred[self.monitor_y_true == 1]
                scores_KO = self.monitor_y_pred[self.monitor_y_true == 0]
                scores_OK.sort()

                limit = scores_OK[int((1-frac)*len(scores_OK))]
                exclu = np.count_nonzero(scores_KO >= limit)
                total = scores_KO.shape[0]
                tnr = 1 - (exclu / total)
            else:
                scores_id.sort()

                limit = scores_id[int(1-frac)*len(scores_id)]
                exclu = np.count_nonzero(scores_ood < limit)
                total = scores_ood.shape[0]
                tnr = exclu / total
            return tnr

    def _neg_f1(self, threshold):
        y_pred = self.monitor_y_pred <= threshold
        f1 = f1_score(self.monitor_y_true, y_pred)

        return -f1

    def _check_accepted_setting(self):
        accepted_settings = ["ood", "oms"]
        if self.setting not in accepted_settings:
            raise ValueError("Accepted settings are: %s" % str(accepted_settings)[1:-1])


@jit(nopython=True, parallel=True)
def get_optimal_threshold_f1(scores, y_true):
    all_recalls = np.zeros(len(scores))
    all_precs = np.zeros(len(scores))
    all_f1s = np.zeros(len(scores))

    for i in range(len(scores)):
        is_tp = np.zeros(len(scores))
        is_tn = np.zeros(len(scores))
        is_fp = np.zeros(len(scores))
        is_fn = np.zeros(len(scores))
        y_pred = scores <= scores[i]
        for j in range(len(scores)):
            if y_pred[j] == True and y_true[j] == True:
                is_tp[j] = 1
            elif y_pred[j] == False and y_true[j] == False:
                is_tn[j] = 1
            elif y_pred[j] == True and y_true[j] == False:
                is_fp[j] = 1
            else:
                is_fn[j] = 1

        tot_tp = is_tp.sum()
        tot_tn = is_tn.sum()
        tot_fp = is_fp.sum()
        tot_fn = is_fn.sum()

        if tot_tp > 0:
            precision = tot_tp / (tot_tp + tot_fp)
            recall = tot_tp / (tot_tp + tot_fn)
            f1 = 2 * (precision * recall) / (precision + recall)

            all_f1s[i] = f1
            all_precs[i] = precision
            all_recalls[i] = recall

    argmax = np.argmax(all_f1s)
    thresh = scores[argmax]
    f1 = all_f1s[argmax]
    precision = all_precs[argmax]
    recall = all_recalls[argmax]

    return thresh, f1, precision, recall, all_precs, all_recalls, all_f1s


def plot_roc_curve_ood(scores_test, scores_ood):
    y = np.concatenate([scores_test, scores_ood])
    y_true = np.array([1] * scores_test.shape[0] + [0] * scores_ood.shape[0])

    RocCurveDisplay.from_predictions(y_true, y)


def plot_precision_recall_curve_ood(scores_test, scores_ood, labels_test, labels_ood,
                  preds_test, preds_ood):
    y = np.concatenate([scores_test, scores_ood])
    y_true = np.array([1] * scores_test.shape[0] + [0] * scores_ood.shape[0])

    PrecisionRecallDisplay.from_predictions(y_true, y)


def plot_roc_curve_oms(scores_test, scores_ood, labels_test, labels_ood,
                       preds_test, preds_ood):
    y = np.concatenate([scores_test, scores_ood])

    labs = np.concatenate([labels_test, labels_ood])
    preds = np.concatenate([preds_test, preds_ood])

    y_true = labs != preds

    RocCurveDisplay.from_predictions(y_true, y)


def plot_precision_recall_curve_oms(scores_test, scores_ood, labels_test, labels_ood,
                                    preds_test, preds_ood):
    y = np.concatenate([scores_test, scores_ood])

    labs = np.concatenate([labels_test, labels_ood])
    preds = np.concatenate([preds_test, preds_ood])

    y_true = labs != preds

    return PrecisionRecallDisplay.from_predictions(y_true, y)


def compute_tnr_frac_tpr(scores, labels, preds, frac=0.95):
    """
    General computation of the TNR with fixed TPR frac.
    """
    y_OK = labels != preds
    y_KO = labels == preds

    scores_OK = scores[y_OK]
    scores_KO = scores[y_KO]

    scores_OK.sort()
    limit = scores_OK[int((1 - frac) * len(scores_OK))]
    exclu = np.count_nonzero(scores_KO >= limit)
    total = scores_KO.shape[0]

    tnr = 1 - (exclu / total)
    return tnr


def compute_aupr(scores, labels, model_y_pred):
    """
    General computation of the Average-Precision score (AUPR)
    """
    y_true = labels != model_y_pred
    return average_precision_score(y_true, scores)


def compute_auroc(scores, labels, preds):
    """
    """
    y_true = labels != preds
    return roc_auc_score(y_true, scores)
