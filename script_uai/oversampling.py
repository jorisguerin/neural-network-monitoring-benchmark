import numpy as np


# upsampling strategy to balance data in optimization set, by oversampling copies of data of minor class 
def upsampling_naive(features_optimization, logits_optimization, softmax_optimization, pred_optimization, lab_optimization, y_true_optimization): 
    """
    Implement a simple oversampling strategy on the threshold optimization set: multiply all data of the minor class (either Positive or Negative) N greater-or-equal-to-1 times so that we have a fairly balance ratio (between 0.4 and 0.6).    
    """
    # Determine minority class 
    minor_class = 0 if np.count_nonzero(y_true_optimization)/len(y_true_optimization) > 0.5 else 1 
    idx_minor = np.where(y_true_optimization==minor_class)
    assert len(idx_minor[0]) != 0 # case where no label of minor class? 
    
    upsample_coef = round(len(y_true_optimization[y_true_optimization==1-minor_class]) / len(y_true_optimization[y_true_optimization==minor_class]))
    
    # Create oversampled optimizationing data set for minority class
    features_optimization_ups, logits_optimization_ups, \
    softmax_optimization_ups, pred_optimization_ups, \
    lab_optimization_ups, y_true_optimization_ups = np.repeat(features_optimization[0][idx_minor], upsample_coef-1, axis=0), \
                                        np.repeat(logits_optimization[idx_minor], upsample_coef-1, axis=0), \
                                        np.repeat(softmax_optimization[idx_minor], upsample_coef-1, axis=0), \
                                        np.repeat(pred_optimization[idx_minor], upsample_coef-1, axis=0), \
                                        np.repeat(lab_optimization[idx_minor], upsample_coef-1, axis=0), \
                                        np.repeat(y_true_optimization[idx_minor], upsample_coef-1, axis=0) 
    
    # Reformat features into list of array(s)
    features_optimization_ups = [features_optimization_ups]

    # Append the oversampled minority class to threshold optimization data and related labels
    return [np.concatenate((features_optimization[0], features_optimization_ups[0]), axis=0)], \
            np.concatenate((logits_optimization, logits_optimization_ups), axis=0), \
            np.concatenate((softmax_optimization, softmax_optimization_ups), axis=0), \
            np.concatenate((pred_optimization, pred_optimization_ups), axis=0), \
            np.concatenate((lab_optimization, lab_optimization_ups), axis=0), \
            np.concatenate((y_true_optimization, y_true_optimization_ups), axis=0)   
            

def upsampling_naive_add_flag(features_optimization, logits_optimization, softmax_optimization, pred_optimization, lab_optimization, y_true_optimization, flag_type_optimization): 
    """
    Implement a simple oversampling strategy on the threshold optimization set: multiply all data of the minor class (either Positive or Negative) N greater-or-equal-to-1 times so that we have a fairly balance ratio (between 0.4 and 0.6).    
    """
    # Determine minority class 
    minor_class = 0 if np.count_nonzero(y_true_optimization)/len(y_true_optimization) > 0.5 else 1 
    idx_minor = np.where(y_true_optimization==minor_class)
    assert len(idx_minor[0]) != 0 # case where no label of minor class? 
    
    upsample_coef = round(len(y_true_optimization[y_true_optimization==1-minor_class]) / len(y_true_optimization[y_true_optimization==minor_class]))
    
    # Create oversampled optimizationing data set for minority class
    features_optimization_ups, logits_optimization_ups, \
    softmax_optimization_ups, pred_optimization_ups, \
    lab_optimization_ups, y_true_optimization_ups, flag_type_optimization_ups = np.repeat(features_optimization[0][idx_minor], upsample_coef-1, axis=0), \
                                        np.repeat(logits_optimization[idx_minor], upsample_coef-1, axis=0), \
                                        np.repeat(softmax_optimization[idx_minor], upsample_coef-1, axis=0), \
                                        np.repeat(pred_optimization[idx_minor], upsample_coef-1, axis=0), \
                                        np.repeat(lab_optimization[idx_minor], upsample_coef-1, axis=0), \
                                        np.repeat(y_true_optimization[idx_minor], upsample_coef-1, axis=0), \
                                        np.repeat(np.array(flag_type_optimization)[idx_minor], upsample_coef-1, axis=0)  
    
    # Reformat features into list of array(s)
    features_optimization_ups = [features_optimization_ups]

    # Append the oversampled minority class to threshold optimization data and related labels
    return [np.concatenate((features_optimization[0], features_optimization_ups[0]), axis=0)], \
            np.concatenate((logits_optimization, logits_optimization_ups), axis=0), \
            np.concatenate((softmax_optimization, softmax_optimization_ups), axis=0), \
            np.concatenate((pred_optimization, pred_optimization_ups), axis=0), \
            np.concatenate((lab_optimization, lab_optimization_ups), axis=0), \
            np.concatenate((y_true_optimization, y_true_optimization_ups), axis=0), \
            np.concatenate((flag_type_optimization, flag_type_optimization_ups), axis=0)
            
