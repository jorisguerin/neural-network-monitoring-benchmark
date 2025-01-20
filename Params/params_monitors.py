import numpy as np

path_to_saved_monitors = "./Monitors/saves/"

accepted_n_comp = ["auto_knee", "auto_bic"]
accepted_constr = ["full", "diag", "tied", "spherical", "auto_bic"]

otb_n_clust_values = np.arange(1, 11, 1)

gmm_n_comp_values = np.arange(1, 11, 1)
gmm_constr_values = ["full", "diag", "tied", "spherical"]
