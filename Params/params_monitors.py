import numpy as np

GMM_MIN_N_COMPONENTS_VALUES = 1
GMM_MAX_N_COMPONENTS_VALUES = 9

path_to_saved_monitors = "./Monitors/saves/"

accepted_n_components = ["auto_knee", "auto_bic", "auto_aic"]
accepted_t_covariance = ["full", "diag", "tied", "spherical", "auto_bic"]

otb_n_clust_values = np.arange(1, 11, 1)

gmm_n_components_values = np.arange(
    GMM_MIN_N_COMPONENTS_VALUES, 
    GMM_MAX_N_COMPONENTS_VALUES + 1, 1)
gmm_t_covariance_values = ["full", "diag", "tied", "spherical"]
