'''
This file, contains a collection of functions designed to compute an optimal threshold for runtime monitor predictions. 
These thresholds are calculated based on a variety of metrics including F1 score, G-mean, Matthews Correlation Coefficient (MCC), and Cohen's Kappa. 

The goal of these functions is to maximize the performance of the prediction model by adjusting the threshold for classification.
This is done by comparing the prediction scores against the ground truth data.

In addition, this file also includes functions to evaluate various metrics such as True Negative Rate at 95% True Positive Rate (TNR@95TPR),
True Positive Rate at True Negative Rate (TPR@TNR), F1 score, G-mean, precision, recall, specificity, and accuracy. 
These functions provide a comprehensive evaluation of the model's performance.
'''

# Importing necessary libraries
import numpy as np 
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay
from sklearn.metrics import precision_score, recall_score, f1_score, fbeta_score, accuracy_score
from sklearn.metrics import confusion_matrix, matthews_corrcoef, cohen_kappa_score

# Functions to compute optimal threshold

def get_tnr_frac_tpr_oms_new(scores, y_true, frac=0.95):
    """
    Compute TNR (True Negative Rate) when TPR (True Positive Rate) reaches a score of frac (usually 0.95).

    Parameters:
    scores (numpy array): Array of scores, must be continuous values, not booleans.
    y_true (numpy array): Array of true binary labels.
    frac (float, optional): The fraction of TPR to reach. Default is 0.95.

    Returns:
    float: The computed TNR when TPR reaches the specified fraction.
    """
    
    # Check if scores are boolean
    if scores.dtype == "bool":
        raise ValueError("Scores must be continuous values, not booleans")
    # Get scores for positive and negative samples   
    scores_correct = scores[y_true]
    scores_wrong = scores[y_true == 0]
    # Sort scores and set limit to reach frac
    scores_correct.sort()
    limit = scores_correct[int((1 - frac) * len(scores_correct))]
    # Compute TNR
    excluded = np.count_nonzero(scores_wrong >= limit)
    total = scores_wrong.shape[0]
    tnr = 1 - (excluded / total)
    return tnr

def get_tpr_frac_tnr_oms_new(scores, y_true, frac=0.95):
    """
    Compute TPR (True Positive Rate) when TNR (True Negative Rate) reaches a score of frac (usually 0.95).

    Parameters:
    scores (numpy array): Array of scores, must be continuous values, not booleans.
    y_true (numpy array): Array of true binary labels.
    frac (float, optional): The fraction of TNR to reach. Default is 0.95.

    Returns:
    float: The computed TPR when TNR reaches the specified fraction.
    """
    
    # Check if scores are boolean
    if scores.dtype == "bool":
        raise ValueError("Scores must be continuous values, not booleans")
    # Get scores for positive and negative samples
    scores_posi = scores[y_true]
    scores_nega = scores[y_true == 0]
    # Sort scores and set limit to reach frac
    scores_nega.sort()
    limit = scores_nega[int(frac * len(scores_nega))]
    # Compute TPR
    true_posi = np.count_nonzero(scores_posi >= limit)
    total = scores_posi.shape[0]
    tpr = (true_posi / total)
    return tpr

def get_specificity_score(y_true, y_pred): 
    """Compute specificity score.

    Parameters:
    y_true (numpy array): Array of true binary labels.
    y_pred (numpy array): Array of predicted binary labels.

    Returns:
    float: The computed specificity score.
    """
    
    # not handle for now zero division error
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn+fp)

def get_optimal_threshold_f1_new(scores, y_true, beta=1, constraint=False, display=True):
    """
    Compute threshold that maximizes f1 score of the vector input corresponding to its ground truth.
    
    Parameters:
    scores (numpy array): Array of scores, must be continuous values, not booleans.
    y_true (numpy array): Array of true binary labels.
    beta (float, optional): The beta parameter for F-beta score. Default is 1, which corresponds to F1 score.
    constraint (bool, optional): A boolean indicating if there exists a constraint or not on Recall > Specificity. Default is False.
    display (bool, optional): A boolean indicating if the results should be printed. Default is True.

    Returns:
    float: The optimal threshold.
    float: The precision at the optimal threshold.
    float: The recall at the optimal threshold.
    float: The F1 score at the optimal threshold.

    Note:
    The thresholds chosen are taken from sklearn precision_recall_curve.
    """

    # Compute fp, tp, precision, recall, thresholds
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    precision, recall, thresholds = precision_recall_curve(y_true, scores)

    # Computing fbeta, handle case where tp = 0, leading to a null precision and null recall
    f1_scores = np.zeros_like(precision)
    nom = (1+beta**2)*recall*precision
    denom = (recall+beta**2*precision)
    np.divide(nom, denom, out=f1_scores, where=(denom != 0))

    # # Apply constraint if needed
    # if constraint:
    #     idx_constraint = np.where(tpr > (1-fpr))
    #     m = np.ones(f1_scores.size, dtype=bool)
    #     m[idx_constraint] = False
    #     f1_scores = np.ma.array(f1_scores, mask=m)

    # Get optimal threshold
    index_opt = np.argmax(f1_scores)
    threshold_opt = thresholds[index_opt]
    precision_opt = precision[index_opt]
    recall_opt = recall[index_opt]
    f1_opt = f1_scores[index_opt]
    
    # Display
    if display:
        print("\n ...Optimization set fitting")
        print('Optimal threshold: ', threshold_opt)
        print('Optimal F{} Score: '.format(beta), f1_opt)
        print(f"Recall score {recall_opt}, Precision score {precision_opt}")

    return threshold_opt, precision_opt, recall_opt, f1_opt

def get_optimal_threshold_Gmean(scores, y_true, constraint=False, display=True):
    """
    Compute threshold that maximizes Gmean(Recall, Specificity) score of the vector input corresponding to its ground truth.

    Parameters:
    scores (numpy array): Array of scores, must be continuous values, not booleans.
    y_true (numpy array): Array of true binary labels.
    constraint (bool, optional): A boolean indicating if there exists a constraint or not on Recall > Specificity. Default is False.
    display (bool, optional): A boolean indicating if the results should be printed. Default is True.

    Returns:
    float: The optimal threshold.
    float: The False Positive Rate (FPR) at the optimal threshold.
    float: The True Positive Rate (TPR) at the optimal threshold.
    float: The G-mean at the optimal threshold.

    Note:
    The thresholds chosen are taken from sklearn roc_curve.
    """
    
    # Compute fpr, tpr thresholds
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    # Compute Gmean
    gmean = np.sqrt(tpr * (1 - fpr))

    # # Apply constraint if needed
    # if constraint:
    #     idx_constraint = np.where(tpr > (1-fpr))
    #     m = np.ones(gmean.size, dtype=bool)
    #     m[idx_constraint] = False
    #     gmean = np.ma.array(gmean, mask=m)

    # Get optimal threshold
    index_opt = np.argmax(gmean)
    threshold_opt = thresholds[index_opt]
    fpr_opt = fpr[index_opt]
    tpr_opt = tpr[index_opt]
    gmean_opt = gmean[index_opt]
    
    # Display
    if display:
        print("\n ...Optimization set fitting")
        print('Optimal threshold: ', threshold_opt)
        print('Optimal G-mean: ', gmean_opt)
        print(f"FPR score: {fpr_opt}, TPR score: {tpr_opt}")

    return threshold_opt, fpr_opt, tpr_opt, gmean_opt

def get_optimal_threshold_YoudenJstat(scores, y_true, constraint=False, display=True):
    """
    Compute threshold that maximizes YoudenJ's Statistic of the vector input corresponding to its ground truth.
    
    Parameters:
    scores (numpy array): Array of scores, must be continuous values, not booleans.
    y_true (numpy array): Array of true binary labels.
    constraint (bool, optional): A boolean indicating if there exists a constraint or not on Recall > Specificity. Default is False.
    display (bool, optional): A boolean indicating if the results should be printed. Default is True.

    Returns:
    float: The optimal threshold.
    float: The False Positive Rate (FPR) at the optimal threshold.
    float: The True Positive Rate (TPR) at the optimal threshold.
    float: The Youden's J statistic at the optimal threshold.

    Note:
    The thresholds chosen are taken from sklearn roc_curve.
    """
    
    # Compute fpr, tpr thresholds
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    # Youden's J statistic calculation
    youdenJ = tpr - fpr
    
    # # Apply constraint if needed
    # if constraint:
    #     idx_constraint = np.where(tpr > (1-fpr))
    #     m = np.ones(youdenJ.size, dtype=bool)
    #     m[idx_constraint] = False
    #     youdenJ = np.ma.array(youdenJ, mask=m)

    # Get optimal threshold
    index_opt = np.argmax(youdenJ)
    threshold_opt = thresholds[index_opt]
    fpr_opt = fpr[index_opt]
    tpr_opt = tpr[index_opt]
    youdenJ_opt = youdenJ[index_opt]
    
    # Display
    if display:
        print("\n ...Optimization set fitting")
        print('Optimal threshold: ', threshold_opt)
        print('Optimal Youden J statistic: ', youdenJ_opt)
        print(f"FPR score: {fpr_opt}, TPR score: {tpr_opt}")

    return threshold_opt, fpr_opt, tpr_opt, youdenJ_opt

def get_optimal_threshold_MCC(scores, y_true, display=True):
    """
    Compute threshold that maximizes Matthews correlation coefficient (MCC) score of the vector input corresponding to its ground truth.

    Parameters:
    scores (numpy array): Array of scores, must be continuous values, not booleans.
    y_true (numpy array): Array of true binary labels.
    display (bool, optional): A boolean indicating if the results should be printed. Default is True.

    Returns:
    float: The optimal threshold.
    float: The MCC at the optimal threshold.
    """
    
    # Compute MCC
    all_mcc = np.ones(len(scores)) * (-1) # init all values at -1
    
    scores.sort()
    
    # Compute MCC for each threshold
    distinct_thresh_idx = np.r_[np.where(np.diff(scores))[0], scores.size-1]
    for i in distinct_thresh_idx:
        y_pred = scores <= scores[i]
        all_mcc[i] = matthews_corrcoef(y_true,y_pred)

    # Get optimal threshold
    argmax = np.argmax(all_mcc)
    threshold_opt = scores[argmax]
    mcc_opt = all_mcc[argmax]
    
    # Display
    if display:
        print("\n ...Optimization set fitting")
        print('Optimal threshold: ', threshold_opt)
        print('Optimal MCC: ', mcc_opt)

    return threshold_opt, mcc_opt

def get_optimal_threshold_kappa(scores, y_true, display=True):
    """
    Compute threshold that maximizes Cohen’s kappa statistic of the vector input corresponding to its ground truth.

    Parameters:
    scores (numpy array): Array of scores, must be continuous values, not booleans.
    y_true (numpy array): Array of true binary labels.
    display (bool, optional): A boolean indicating if the results should be printed. Default is True.

    Returns:
    float: The optimal threshold.
    float: The Cohen’s kappa statistic at the optimal threshold.

    """
    # Compute Kappa
    all_kappa = np.ones(len(scores)) * (-1) # init all values at -1
    
    scores.sort()
    
    # Compute Kappa for each threshold
    distinct_thresh_idx = np.r_[np.where(np.diff(scores))[0], scores.size-1]
    for i in distinct_thresh_idx:
        y_pred = scores <= scores[i]
        all_kappa[i] = cohen_kappa_score(y_true,y_pred)

    # Get optimal threshold
    argmax = np.argmax(all_kappa)
    threshold_opt = scores[argmax]
    kappa_opt = all_kappa[argmax]
    
    # Display
    if display:
        print("\n ...Optimization set fitting")
        print('Optimal threshold: ', threshold_opt)
        print('Optimal Kappa: ', kappa_opt)

    return threshold_opt, kappa_opt

def get_threshold_fracTNR(scores, y_true, frac=0.95, display=True):
    """
    Compute threshold which gives a TNR score at frac (usually 0.95) of the vector input corresponding to its ground truth.
    """
    if scores.dtype == "bool":
        raise ValueError("Scores must be continuous values, not booleans")

    scores_posi = scores[y_true == 1]
    scores_nega = scores[y_true == 0]

    scores_nega.sort()

    threshold = scores_nega[int(frac * len(scores_nega))]

    return threshold

def compute_metrics_thresholdOpt_evaluationset(y_true_evaluation, y_pred_evaluation, thresholdOpt, beta=1, display=True):
    """
    Compute different metrics useful for the evaluation of the experiences. 
    Evaluation metrics include Fbeta, precision, recall, accuracy, specificity, gmean, youdenJ stat.

    Parameters:
    y_true_evaluation (numpy array): Array of true binary labels in range {0, 1} or {-1, 1}. If labels are not binary, pos_label should be explicitly given.
    y_pred_evaluation (numpy array): Array of predicted binary labels in range {0, 1} or {-1, 1}.
    thresholdOpt (float): The optimal threshold.
    beta (float, optional): The beta parameter for F-beta score. Default is 1.
    display (bool, optional): A boolean indicating if the results should be printed. Default is True.

    Returns:
    float: The F-beta score.
    float: The precision score.
    float: The recall score.
    float: The accuracy score.
    float: The specificity score.
    float: The gmean score.
    float: The Youden's J statistic.

    """
    # Compute different metrics threshold-based
    fbeta = fbeta_score(y_true_evaluation, y_pred_evaluation, beta=beta)
    precision = precision_score(y_true_evaluation, y_pred_evaluation)
    recall = recall_score (y_true_evaluation, y_pred_evaluation)
    accuracy = accuracy_score(y_true_evaluation, y_pred_evaluation) # binary accuracy or jaccard score
    specificity = get_specificity_score(y_true_evaluation, y_pred_evaluation)
    gmean = np.sqrt(recall * specificity)
    youden = recall - (1-specificity) 

    # Display
    if display: 
        print("\n ...evaluation set evaluation")
        print('Optimal threshold = {} from training set is used to compute following metrics:'.format(np.round(thresholdOpt, 5)))

        print(f"F{beta}-score: {fbeta}")
        print(f"Recall score: {recall}")
        print(f"Precision score: {precision}")
        print(f"Accuracy score: {accuracy}")
        print(f"Specificity score: {specificity}")
        print('\n...Additional metrics:')
        print(f"gmean-score: {gmean}")
        print(f"youden-score: {youden}")
    return fbeta, precision, recall, accuracy, specificity, gmean, youden