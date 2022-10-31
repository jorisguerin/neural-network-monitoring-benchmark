import numpy as np

save_monitors_path = "./Monitors/"

accepted_n_comp = ["auto_knee", "auto_bic"]
accepted_constraints = ["full", "diag", "tied", "spherical", "auto_bic"]

otb_n_clust_values = np.arange(1, 11, 1)

gmm_n_comp_values = np.arange(1, 11, 1)
gmm_constraints_values = ["full", "diag", "tied", "spherical"]
