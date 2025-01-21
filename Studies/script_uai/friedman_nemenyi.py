'''
This file defines functions to conduct the Friedman and Nemenyi post hoc tests. 
The results of the Nemenyi test are visualized using a critical difference diagram. 
These tests are commonly used in machine learning to compare the performance of different algorithms across multiple datasets.
'''

# import libraries
from scipy import stats
import scikit_posthocs as sp
import scipy.stats as ss
import scipy
from scipy.stats import friedmanchisquare
from scikit_posthocs import posthoc_nemenyi_friedman
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt


########### FUNCTIONS TO CONDUCT FRIEDMAN AND NEMENYI TESTS ###########

"""FUNCTIONS sign_array, _find_maximal_cliques, _bron_kerbosch, critical_difference_diagram TAKEN FROM https://github.com/maximtrp/scikit-posthocs (new version of library scikit-posthoc)
*** with some modifications to plot color differently """

from typing import Union, List, Tuple, Dict, Set
from pandas import DataFrame, Series
from matplotlib import colors
from matplotlib.axes import SubplotBase
from matplotlib.colorbar import ColorbarBase, Colorbar
from matplotlib.colors import ListedColormap
from matplotlib import pyplot

def sign_array(
        p_values: Union[List, np.ndarray],
        alpha: float = 0.05) -> np.ndarray:
    """Significance array.

    Converts an array with p values to a significance array where
    0 is False (not significant), 1 is True (significant),
    and -1 is for diagonal elements.
    """
    p_values = np.array(p_values)
    p_values[p_values > alpha] = 0
    p_values[(p_values < alpha) & (p_values > 0)] = 1
    np.fill_diagonal(p_values, 1)

    return p_values

def _find_maximal_cliques(adj_matrix: DataFrame) -> List[Set]:
    """Wrapper function over the recursive Bron-Kerbosch algorithm.

    Will be used to find points that are under the same crossbar in critical
    difference diagrams.
    """
    if (adj_matrix.index != adj_matrix.columns).any():
        raise ValueError("adj_matrix must be symmetric, indices do not match")
    if not adj_matrix.isin((0, 1)).values.all():
        raise ValueError("Input matrix must be binary")
    if adj_matrix.empty or not (adj_matrix.T == adj_matrix).values.all():
        raise ValueError("Input matrix must be non-empty and symmetric")

    result = []
    _bron_kerbosch(
        current_clique=set(),
        candidates=set(adj_matrix.index),
        visited=set(),
        adj_matrix=adj_matrix,
        result=result,
    )
    return result

def _bron_kerbosch(
        current_clique: Set,
        candidates: Set,
        visited: Set,
        adj_matrix: DataFrame,
        result: List[Set]) -> None:
    """Recursive algorithm to find the maximal fully connected subgraphs.
    """
    while candidates:
        v = candidates.pop()
        _bron_kerbosch(
            current_clique | {v},
            # Restrict candidate vertices to the neighbors of v
            {n for n in candidates if adj_matrix.loc[v, n]},
            # Restrict visited vertices to the neighbors of v
            {n for n in visited if adj_matrix.loc[v, n]},
            adj_matrix,
            result,
        )
        visited.add(v)

    # We do not need to report a clique if a children call aready did it.
    if not visited:
        # If this is not a terminal call, i.e. if any clique was reported.
        result.append(current_clique)
        
def critical_difference_diagram(
        ranks: Union[dict, Series],
        sig_matrix: DataFrame,
        *,
        ax: SubplotBase = None,
        label_fmt_left: str = '{label} ({rank:.3g})',
        label_fmt_right: str = '({rank:.3g}) {label}',
        label_props: dict = None,
        marker_props: dict = None,
        elbow_props: dict = None,
        crossbar_props: dict = None,
        text_h_margin: float = 0.01) -> Dict[str, list]:
    
    """Plot a Critical Difference diagram from ranks and post-hoc results.

    The diagram arranges the average ranks of multiple groups on the x axis
    in order to facilitate performance comparisons between them. The groups
    that could not be statistically deemed as different are linked by a
    horizontal crossbar [1]_, [2]_.

    ::

                      rank markers
         X axis ---------O----O-------------------O-O------------O---------
                         |----|                   | |            |
                         |    |                   |---crossbar---|
                clf1 ----|    |                   | |            |---- clf3
                clf2 ---------|                   | |----------------- clf4
                                                  |------------------- clf5
                    |____|
                text_h_margin

    In the drawing above, the two crossbars indicate that clf1 and clf2 cannot
    be statistically differentiated, the same occurring between clf3, clf4 and
    clf5. However, clf1 and clf2 are each significantly lower ranked than clf3,
    clf4 and clf5.

    """
    elbow_props = elbow_props or {}
    marker_props = {"zorder": 3, **(marker_props or {})}
    label_props = {"va": "center", **(label_props or {})}
    crossbar_props = {"color": "k", "zorder": 3,
                      "linewidth": 5, **(crossbar_props or {})}

    ax = ax or pyplot.gca()
    ax.yaxis.set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.xaxis.set_ticks_position('top')
    ax.spines['top'].set_position('zero')

    # lists of artists to be returned
    markers = []
    elbows = []
    labels = []
    crossbars = []

    # True if pairwise comparison is NOT significant
    adj_matrix = DataFrame(
        1 - sign_array(sig_matrix),
        index=sig_matrix.index,
        columns=sig_matrix.columns,
        dtype=bool,
    )
    
    ranks = Series(ranks)  # Standardize if ranks is dict
    points_left, points_right = np.array_split(ranks.sort_values(), 2)
    
    # Sets of points under the same crossbar
    crossbar_sets = _find_maximal_cliques(adj_matrix)

    # Sort by lowest rank and filter single-valued sets
    crossbar_sets = sorted(
        (x for x in crossbar_sets if len(x) > 1),
        key=lambda x: ranks[list(x)].min()
    )

    # Create stacking of crossbars: for each level, try to fit the crossbar,
    # so that it does not intersect with any other in the level. If it does not
    # fit in any level, create a new level for it.
    crossbar_levels: list[list[set]] = []
    for bar in crossbar_sets:
        for level, bars_in_level in enumerate(crossbar_levels):
            if not any(bool(bar & bar_in_lvl) for bar_in_lvl in bars_in_level):
                ypos = -level-1
                bars_in_level.append(bar)
                break
        else:
            ypos = -len(crossbar_levels) - 1
            crossbar_levels.append([bar])

        crossbars.append(ax.plot(
            # Adding a separate line between each pair enables showing a
            # marker over each elbow with crossbar_props={'marker': 'o'}.
            [ranks[i] for i in bar],
            [ypos] * len(bar),
            **crossbar_props,
        ))

    lowest_crossbar_ypos = -len(crossbar_levels)
    
    custom_color = ['red', 'orange', 'green', 'blue'] ########### ADDED BY USER
    if len(list(ranks.index)) != 4:
        custom_color = ['red', 'orange', 'green', 'blue', 'yellow', 'purple', 'pink', 'brown']
    label_to_color = dict(zip(list(ranks.index), custom_color)) ########### ADDED BY USER
    
    def plot_items(points, xpos, label_fmt, label_props):
        """Plot each marker + elbow + label."""
        ypos = lowest_crossbar_ypos - 0.5
        for i, (label, rank) in enumerate(points.items()):
            curr_color = label_to_color[label]

            elbow, *_ = ax.plot(
                [xpos, rank, rank],
                [ypos, ypos, 0],
                **{"color": curr_color, **elbow_props},
            )
            elbows.append(elbow)
            markers.append(
                ax.scatter(rank, 0, **{"color": curr_color, **marker_props})
            )
            labels.append(
                ax.text(
                    xpos,
                    ypos,
                    label_fmt.format(label=label, rank=rank),
                    **{"color": curr_color, **label_props},
                )
            )
            ypos -= 1

    plot_items(
        points_left,
        xpos=points_left.iloc[0] - text_h_margin,
        label_fmt=label_fmt_left,
        label_props={"ha": "right", **label_props},
    )
    plot_items(
        points_right[::-1],
        xpos=points_right.iloc[-1] + text_h_margin,
        label_fmt=label_fmt_right,
        label_props={"ha": "left", **label_props},
    )

    return {
        "markers": markers,
        "elbows": elbows,
        "labels": labels,
        "crossbars": crossbars,
    }


########### FUNCTIONS TO DISPLAY THE RESULT OF THE FRIEDMAN TEST AND THE POST HOC NEMENYI TEST ##########

def display_result_posthoc_nemenyi_friedman(dict_data, alpha=0.05, display=False):
    """
    This function performs the Friedman test and the post hoc Nemenyi test on the input data.
    If the p-value from the Friedman test is less than the significance level (alpha), 
    it performs the Nemenyi test and optionally displays the results.

    Parameters:
    dict_data (dict): The input data for the tests.
    alpha (float): The significance level for the tests. Default is 0.05.
    display (bool): Whether to print the results of the Nemenyi test. Default is False.

    Returns:
    DataFrame or None: The results of the Nemenyi test if the p-value from the Friedman test 
    is less than alpha, otherwise None.
    """

    # Step 1: Perform the Friedman test
    input_friedman = np.array([value_per_case for value_per_case in dict_data.values()])
    friedman_stat, p_value = friedmanchisquare(*input_friedman)
    # print("Friedman chi-squared statistic:", friedman_stat)
    # print("p-value:", p_value)

    # Step 2: Perform the post hoc Nemenyi test
    # First, reshape the data into a format suitable for the Nemenyi test
    data = (
      pd.DataFrame(dict_data)
      .rename_axis('cv_fold')
      .melt(
          var_name='estimator',
          value_name='score',
          ignore_index=False,
      )
      .reset_index()
    ) 

    # If the p-value from the Friedman test is less than alpha, perform the Nemenyi test
    if p_value < alpha:  # Adjust the significance level as needed
        
        nemenyi_results = posthoc_nemenyi_friedman(data,
                 melted=True,
                 block_col='cv_fold',
                 group_col='estimator',
                 y_col='score',
             )
        
        # If display is True, print the results of the Nemenyi test
        if display == True:
            # Compare mean of each group with every other group
            for i in range(len(nemenyi_results)):
                for j in range(i + 1, len(nemenyi_results)):
                    group1 = f"Case {i}"
                    group2 = f"Case {j}"
                    if nemenyi_results.iloc[i, j] < alpha:  # Adjust the significance level as needed
                        if np.mean(data[:, i]) > np.mean(data[:, j]):
                            print(f"...{group1} is significantly greater than {group2}.")
                        else:
                            print(f"...{group1} is significantly less than {group2}.")
                    else:
                        print(f"...There is no significant difference between {group1} and {group2}.")
        return nemenyi_results
    else: 
        return None


def plot_compare_metrics_nemenyi_diagram(results_df_all, monitor_concerned_list, fig_show=True, app='gmean', alpha=0.05, ood_type='all', save=True, path_image=''):
    """
    This function plots a critical difference diagram for the results of the Nemenyi post hoc test.
    It first extracts the data for each case, computes the average rank of the different cases, 
    performs the Friedman and Nemenyi tests, and then plots the diagram.

    Parameters:
    results_df_all (DataFrame): The input data for the tests.
    monitor_concerned_list (list): The list of monitors to include in the tests.
    fig_show (bool): Whether to display the plot. Default is True.
    app (str): The optimal approach to use. Default is 'gmean'.
    alpha (float): The significance level for the tests. Default is 0.05.
    ood_type (str): The type of threat data to use. Default is 'all'.
    save (bool): Whether to save the plot. Default is True.
    path_image (str): The path where to save the plot. Default is ''.

    Returns:
    None
    """

    # Define the metrics to compare and their corresponding column names
    metrics_compare = ['f1', 'gmean',
                       'recall', 'precision', 
                       'specificity']
    metrics_compare_colname = ["f1-score_evaluation",  "gmean_evaluation", 
                               'recall-score_evaluation', 'precision-score_evaluation', 
                               'specificity-score_evaluation']
    assert len(metrics_compare) == len(metrics_compare_colname)
    
    ood_type='all'
    
    # Define the names of the cases
    case_name = ["ID", 'ID+T', 'ID+T+O', 'ID+O']
    
    # Filter the results to include only the monitors and optimal approach specified
    result_to_compare = results_df_all[results_df_all["Monitor"].isin(monitor_concerned_list)]
    result_to_compare_app = result_to_compare[result_to_compare['Optimal approach']==app]

    # For each metric, perform the tests and plot the diagram
    for no_metric, col in enumerate(metrics_compare_colname):
                
        plt.figure(figsize=(7, 1.75), dpi=100)
        
        # Extract data of each case and store in a dict
        dict_data_compare = {}
        for i in range(4):
            dict_data_compare[case_name[i]] = result_to_compare_app[result_to_compare_app["Case"]==f'Case{i}'][col].tolist()
        
        data_df = (pd.DataFrame(dict_data_compare)
                  .rename_axis('cv_fold')
                  .melt(
                      var_name='estimator',
                      value_name='score',
                      ignore_index=False,
                  )
                  .reset_index())
        
        # Compute Average Rank of different cases
        avg_rank = data_df.groupby('cv_fold').score.rank(pct=True).groupby(data_df.estimator).mean()
    
        # Implement test friedman and nemenyi post hoc
        nemenyi_results = display_result_posthoc_nemenyi_friedman(dict_data_compare, display=False)
        if nemenyi_results is None: 
            raise Exception("No significant difference among cases!") # if Exception need to check
        
        # Plot critical difference diagram
        critical_difference_diagram(avg_rank, nemenyi_results)
        if fig_show:
            plt.title(f"CD diagram of average score ranks for "+ r"$\bf{" + metrics_compare[no_metric] + "}$" + f', opt metric : {app}')

        plt.tight_layout()
        if save:
            plt.savefig(path_image + f'CD diagram nemenyi_{col}_{monitor_concerned_list}_optimize{app}.png')
        plt.show()
        