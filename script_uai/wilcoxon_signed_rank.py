"""
This Python script contains functions for conducting the Wilcoxon signed-rank test and for plotting the results. 
The Wilcoxon signed-rank test is a non-parametric statistical hypothesis test used to compare two related samples.

Functions:
- create_subtitle: Adds a title to a set of subplots in a matplotlib figure.
- compute_average_ranks: Computes the average rank of scores in two lists.
- plot_compare_wilcoxon_per_optimization_method_multiplemonitors: Creates a complex plot comparing the results of different cases across multiple monitors using various metrics.
- plot_compare_wilcoxon_per_optimization_method_onemonitor: Similar to the previous function but for a single monitor.
Dependencies: matplotlib, pandas, numpy, seaborn, scipy
"""

# import libraries
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.gridspec import SubplotSpec
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np 
import seaborn as sns 


# Function to add a title to a set of subplots in a matplotlib figure
def create_subtitle(fig: plt.Figure, grid: SubplotSpec, title: str, y_posi=0.7, fontsize=20, fontweight=None):
    "Sign sets of subplots with title"
    row = fig.add_subplot(grid)
    # the '\n' is important
    row.set_title(f'\n{title}\n', fontsize=fontsize, y=y_posi, fontweight=fontweight)
    # hide subplot
    row.set_frame_on(False)
    row.axis('off')

# Function to compute the average rank of scores in two lists
def compute_average_ranks(list1, list2, pct=True):
    
    df = (pd.DataFrame({'list1':list1, 'list2':list2})
          .rename_axis('cv_fold')
          .melt(
              var_name='estimator',
              value_name='score',
              ignore_index=False,
          )
          .reset_index())
    
    avg_rank = df.groupby('cv_fold').score.rank(pct=pct).groupby(df.estimator).mean()

    return avg_rank[0], avg_rank[1]

# Function to create a complex plot comparing the results of different cases across multiple monitors using various metrics
def plot_compare_wilcoxon_per_optimization_method_multiplemonitors(result_A, result_B, monitor_concerned_list, opt_method_A, opt_method_B, 
                                            fig_title, alpha=0.05):

    metrics_compare = ['f1', 'gmean', 'recall', 'precision', 'specificity']
    metrics_compare_colname = ["f1-score_evaluation",  "gmean_evaluation", 
                               'recall-score_evaluation', 'precision-score_evaluation', 'specificity-score_evaluation']


    # Define RGB color 
    red = (255/255.,140/255.,0/255.) # Red/orange cells = the case corresponding to the row is worst than the column
    green = (100/255.,149/255.,237/255.) # Green/blue cells = the case corresponding to the row is better than the column
    
    # create subplots
    f, axes = plt.subplots(len(monitor_concerned_list), 4, figsize=(20, 15))
    plt.suptitle(fig_title, 
              fontsize=30, y= 1, fontweight='semibold')

    # Define case permutations and their real names
    case_permute = [0, 1, 3, 2] # case 3 linking to strategy ID+O while case 2 linking to strategy ID+T+O
    real_case_name = ["ID", 'ID+T', 'ID+O', 'ID+T+O']

    # Loop over each monitor and create a heatmap for each case
    for no_monitor, monitor_name in enumerate(monitor_concerned_list): 
        # Filter results for the current monitor
        result_to_compare_A = result_A[result_A["Monitor"]==monitor_name]
        result_to_compare_B = result_B[result_B["Monitor"]==monitor_name]
        # Filter results for the current optimization method
        result_to_compare_A_app = result_to_compare_A[result_to_compare_A['Optimal approach']==opt_method_A]
        result_to_compare_B_app = result_to_compare_B[result_to_compare_B['Optimal approach']==opt_method_B]

        # Loop over each case
        for i in range(4):
            # Initialize arrays to store statistics and p-values
            res_statistic = np.zeros((1, len(metrics_compare_colname)))
            res_pvalue = np.zeros((1, len(metrics_compare_colname)))
            flag =  np.zeros((1, len(metrics_compare_colname))) # -1 if less, +1 if greater, 0 if equal 

            # Loop over each metric
            p_value_list = []
            for no_metric, col in enumerate(metrics_compare_colname):
                # Get scores for the current case and metric
                score_A = result_to_compare_A_app[result_to_compare_A_app["Case"]==f'Case{case_permute[i]}'][col].tolist()
                score_B = result_to_compare_B_app[result_to_compare_B_app["Case"]==f'Case{case_permute[i]}'][col].tolist()

                # conduct the Wilcoxon-Signed Rank Test to compare if 2 scores come from same or other distributions
                res = wilcoxon(score_A, score_B, alternative='two-sided')
                
                # Store the test statistic and p-value
                res_statistic[0, no_metric] = res.statistic
                res_pvalue[0, no_metric] = res.pvalue    

                # If p-value is less than alpha, the distributions are different
                if res.pvalue <= alpha: #different distribution 
                        # Compute average ranks for both groups
                        avg_rank_groupA, avg_rank_groupB = compute_average_ranks(score_A, score_B)

                        # Set flag based on which group has a higher average rank
                        if avg_rank_groupA > avg_rank_groupB: 
                            flag[0, no_metric] = 1
                        else:
                            flag[0, no_metric] = -1

            # plotting
            ax_plot = sns.heatmap(res_pvalue, annot=True, fmt='.0e',annot_kws={'rotation': 45, 'size': 15}, 
                                  xticklabels=metrics_compare, yticklabels=[],
                                  cmap=ListedColormap(['white']), square=True, cbar=False,
                                  ax=axes[no_monitor, i])

            ax_plot = sns.heatmap(res_pvalue, mask=(flag!=1), cbar=False, 
                                  xticklabels=metrics_compare, yticklabels=[], cmap=ListedColormap(green), 
                                  ax=axes[no_monitor, i])
            ax_plot = sns.heatmap(res_pvalue, mask=(flag!=-1), cbar=False, linewidths=2, linecolor='gray',
                                  xticklabels=metrics_compare, yticklabels=[], cmap=ListedColormap(red), 
                                  ax=axes[no_monitor, i])
            ax_plot.set_title(real_case_name[i])
            ax_plot.set_xticklabels(metrics_compare, rotation = 45, ha = 'right')
            
            axes[no_monitor, i].title.set_fontsize('25')
            for item in ([axes[no_monitor, i].xaxis.label, axes[no_monitor, i].yaxis.label] +
                         axes[no_monitor, i].get_xticklabels() + axes[no_monitor, i].get_yticklabels()):
                item.set_fontsize('22')
                
    # Add title of each row (each monitor)               
    grid = plt.GridSpec(len(monitor_concerned_list), 4)
    for i in range(len(monitor_concerned_list)):
        create_subtitle(f, grid[i, ::], f'{monitor_concerned_list[i]}', fontsize=28)

    plt.tight_layout(pad=3.0)
    return f

# Function to create a complex plot comparing the results of different cases using one monitor-monitoring approach using various metrics
def plot_compare_wilcoxon_per_optimization_method_onemonitor(result_A, result_B, monitor_concerned, opt_method_A, opt_method_B, 
                                            fig_title, alpha=0.05):
    # Define metrics to compare
    metrics_compare = ['f1', 'gmean', 'recall', 'precision', 'specificity']
    metrics_compare_colname = ["f1-score_evaluation",  "gmean_evaluation", 
                               'recall-score_evaluation', 'precision-score_evaluation', 'specificity-score_evaluation']

    # Define RGB color 
    red = (255/255.,140/255.,0/255.) # Red cells = the case corresponding to the row is worst than the column
    green = (100/255.,149/255.,237/255.) # Green cells = the case corresponding to the row is better than the column
    
    # create subplots
    f, axes = plt.subplots(1, 4, figsize=(20, 15))

    # Define case permutations and their real names
    case_permute = [0, 1, 3, 2] # case 3 linking to strategy ID+O while case 2 linking to strategy ID+T+O
    real_case_name = ["ID", 'ID+T', 'ID+O', 'ID+T+O']

    # Filter results for the current monitor
    result_to_compare_A = result_A[result_A["Monitor"].isin(monitor_concerned)]
    result_to_compare_B = result_B[result_B["Monitor"].isin(monitor_concerned)]
    # Filter results for the current optimization method
    result_to_compare_A_app = result_to_compare_A[result_to_compare_A['Optimal approach']==opt_method_A]
    result_to_compare_B_app = result_to_compare_B[result_to_compare_B['Optimal approach']==opt_method_B]

    # Loop over each case
    for i in range(4):
        # Initialize arrays to store statistics and p-values
        res_statistic = np.zeros((1, len(metrics_compare_colname)))
        res_pvalue = np.zeros((1, len(metrics_compare_colname)))
        flag =  np.zeros((1, len(metrics_compare_colname))) # -1 if less, +1 if greater, 0 if equal 

        # Loop over each metric
        p_value_list = []
        for no_metric, col in enumerate(metrics_compare_colname):
            # Get scores for the current case and metric
            score_A = result_to_compare_A_app[result_to_compare_A_app["Case"]==f'Case{case_permute[i]}'][col].tolist()
            score_B = result_to_compare_B_app[result_to_compare_B_app["Case"]==f'Case{case_permute[i]}'][col].tolist()

            # conduct the Wilcoxon-Signed Rank Test to compare if 2 scores come from same or other distributions
            res = wilcoxon(score_A, score_B, alternative='two-sided')
            # Store the test statistic and p-value
            res_statistic[0, no_metric] = res.statistic
            res_pvalue[0, no_metric] = res.pvalue    

            # If p-value is less than alpha, the distributions are different
            if res.pvalue <= alpha: #different distribution 
                    avg_rank_groupA, avg_rank_groupB = compute_average_ranks(score_A, score_B)

                    if avg_rank_groupA > avg_rank_groupB: 
                        flag[0, no_metric] = 1
                    else:
                        flag[0, no_metric] = -1

        # plotting
        ax_plot = sns.heatmap(res_pvalue, annot=True, fmt='.0e',annot_kws={'rotation': 45, 'size': 15}, 
                              xticklabels=metrics_compare, yticklabels=[],
                              cmap=ListedColormap(['white']), square=True, cbar=False,
                              ax=axes[i])

        ax_plot = sns.heatmap(res_pvalue, mask=(flag!=1), cbar=False, 
                              xticklabels=metrics_compare, yticklabels=[], cmap=ListedColormap(green), 
                              ax=axes[i])
        ax_plot = sns.heatmap(res_pvalue, mask=(flag!=-1), cbar=False, linewidths=2, linecolor='gray',
                              xticklabels=metrics_compare, yticklabels=[], cmap=ListedColormap(red), 
                              ax=axes[i])
        ax_plot.set_title(real_case_name[i], fontsize=25)
        ax_plot.set_xticklabels(metrics_compare, rotation = 45, ha = 'right')
        
        for item in ([axes[i].xaxis.label, axes[i].yaxis.label] +
                     axes[i].get_xticklabels() + axes[i].get_yticklabels()):
            item.set_fontsize('22')

    plt.suptitle(fig_title, y=0.6, fontsize=28)
    plt.tight_layout(pad=3.0)
    return f
