import numpy as np
import matplotlib.pyplot as plt
from numba import jit

from sklearn.metrics import roc_auc_score, average_precision_score, PrecisionRecallDisplay, RocCurveDisplay
from sklearn.metrics import precision_score, recall_score, f1_score

from scipy.optimize import minimize_scalar


class Evaluator:
    def __init__(self, setting, is_novelty):
        self.setting = setting
        self._check_accepted_setting()

        self.is_novelty = is_novelty

        self.scores = None
        self.y_true = None
        self.labels = None
        self.predictions = None

    def fit_ground_truth(self, labels_id=None, labels_ood=None, predictions_id=None, predictions_ood=None):

        if self.setting == "ood":
            self.y_true = np.array([0] * labels_id.shape[0] + [1] * labels_ood.shape[0])
        else:
            self.labels = np.concatenate([labels_id, labels_ood])
            self.predictions = np.concatenate([predictions_id, predictions_ood])

            self.y_true = self.labels != self.predictions
            if self.is_novelty:
                self.y_true[labels_id.shape[0]:] = np.ones(labels_ood.shape[0])

    def get_metrics_f1opt(self, scores_id, scores_ood):
        self.scores = np.concatenate([scores_id, scores_ood])

        if self.scores.dtype == "bool":
            precision = precision_score(self.y_true, self.scores)
            recall = recall_score(self.y_true, self.scores)
            f1 = f1_score(self.y_true, self.scores)

        else:
            thresh, f1, precision, recall, _, _, _ = get_optimal_threshold_f1(self.scores, self.y_true)
        return precision, recall, f1

    def get_average_precision(self, scores_id, scores_ood):
        self.scores = np.concatenate([scores_id, scores_ood])

        if self.scores.dtype == "bool":
            raise ValueError("Scores must be continuous values, not booleans")

        else:
            return average_precision_score(self.y_true, self.scores)

    def get_auroc(self, scores_id, scores_ood):
        self.scores = np.concatenate([scores_id, scores_ood])

        if self.scores.dtype == "bool":
            raise ValueError("Scores must be continuous values, not booleans")

        else:
            return roc_auc_score(self.y_true, self.scores)

    def get_tnr_frac_tpr_oms(self, scores_id, scores_ood, frac=0.95):
        self.scores = np.concatenate([scores_id, scores_ood])

        if self.scores.dtype == "bool":
            raise ValueError("Scores must be continuous values, not booleans")

        else:
            scores_correct = self.scores[self.y_true]
            scores_wrong = self.scores[self.y_true == 0]

            scores_correct.sort()

            limit = scores_correct[int((1 - frac) * len(scores_correct))]

            excluded = np.count_nonzero(scores_wrong >= limit)
            total = scores_wrong.shape[0]

            tnr = 1 - (excluded / total)

            return tnr

    def _neg_f1(self, threshold):
        y_pred = self.scores <= threshold
        f1 = f1_score(self.y_true, y_pred)

        return -f1

    def _check_accepted_setting(self):
        accepted_settings = ["ood", "oms"]
        if self.setting not in accepted_settings:
            raise ValueError("Accepted settings are: %s" % str(accepted_settings)[1:-1])


@jit(nopython=True, parallel=True)
def get_optimal_threshold_f1(scores, y_true):
    all_precisions = np.zeros(len(scores))
    all_recalls = np.zeros(len(scores))
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
            all_precisions[i] = precision
            all_recalls[i] = recall

    argmax = np.argmax(all_f1s)
    thresh = scores[argmax]
    f1 = all_f1s[argmax]
    precision = all_precisions[argmax]
    recall = all_recalls[argmax]

    return thresh, f1, precision, recall, all_precisions, all_recalls, all_f1s


def get_tnr_frac_tpr_ood(scores_test, scores_ood, frac=0.95):
    scores_test.sort()
    limit = scores_test[int((1 - frac) * len(scores_test))]

    excluded = np.count_nonzero(scores_ood < limit)
    total = scores_ood.shape[0]

    tnr = excluded / total

    return tnr, limit


def get_average_precision_ood(scores_test, scores_ood):
    y = np.concatenate([scores_test, scores_ood])
    y_true = np.array([1] * scores_test.shape[0] + [0] * scores_ood.shape[0])

    return average_precision_score(y_true, y)


def get_auroc_ood(scores_test, scores_ood):
    y = np.concatenate([scores_test, scores_ood])
    y_true = np.array([1] * scores_test.shape[0] + [0] * scores_ood.shape[0])

    return roc_auc_score(y_true, y)


def get_tnr_frac_tpr_oms(scores_test, scores_ood, labels_test, labels_ood,
                         preds_test, preds_ood, frac=0.95):

    y = np.concatenate([scores_test, scores_ood])

    labs = np.concatenate([labels_test, labels_ood])
    preds = np.concatenate([preds_test, preds_ood])

    y_true = labs == preds
    y_false = labs != preds

    scores_correct = y[y_true]
    scores_wrong = y[y_false]

    scores_correct.sort()
    limit = scores_correct[int((1 - frac) * len(scores_correct))]

    excluded = np.count_nonzero(scores_wrong < limit)
    total = scores_wrong.shape[0]

    tnr = excluded / total

    return tnr


def get_average_precision_oms(scores_test, scores_ood, labels_test, labels_ood,
                              preds_test, preds_ood):

    y = np.concatenate([scores_test, scores_ood])

    labs = np.concatenate([labels_test, labels_ood])
    preds = np.concatenate([preds_test, preds_ood])

    y_true = labs == preds

    return average_precision_score(y_true, y)


def get_auroc_oms(scores_test, scores_ood, labels_test, labels_ood,
                  preds_test, preds_ood):
    y = np.concatenate([scores_test, scores_ood])

    labs = np.concatenate([labels_test, labels_ood])
    preds = np.concatenate([preds_test, preds_ood])

    y_true = labs == preds

    return roc_auc_score(y_true, y)


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


def get_tnr_frac_tpr(scores, labels, preds, frac=0.95):

    y_true = labels != preds
    y_false = labels == preds

    scores_correct = scores[y_true]
    scores_wrong = scores[y_false]

    scores_correct.sort()
    limit = scores_correct[int((1 - frac) * len(scores_correct))]

    excluded = np.count_nonzero(scores_wrong >= limit)
    total = scores_wrong.shape[0]

    tnr = 1 - (excluded / total)

    return tnr


def get_average_precision(scores, labels, preds):

    y_true = labels != preds

    return average_precision_score(y_true, scores)


def get_auroc(scores, labels, preds):

    y_true = labels != preds

    return roc_auc_score(y_true, scores)
