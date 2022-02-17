from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu
from scipy.stats import wilcoxon

from provided_code.constants_class import ModelParameters


def make_dvh_metric_diff_plots(df_dvh_metrics: pd.DataFrame, constants: ModelParameters):
    """
    Generates box plots to visualize distribution of DVH point differences
    Args:
        df_dvh_metrics: Set of DVH metrics
        constants: Model constants
    """
    # Prep dvh metrics for analysis
    df_to_plot = df_dvh_metrics.unstack((0, 2, 3)).melt()
    df_to_plot.drop(columns=[None], axis=1, inplace=True)  # Drops prediction name
    # Merge the melted data
    df_to_plot.dropna(axis=0, inplace=True)
    df_to_plot.set_index('Metric', inplace=True)
    sns.reset_defaults()
    # Iterate through each type of DVH metric (e.g., D_mean, D_99)
    for m in df_to_plot.index.unique():
        data_to_plot = df_to_plot.loc[m].copy(deep=True)
        data_to_plot.replace(constants.structure_printing, inplace=True)
        data_to_plot.replace(constants.optimization_short_hands_dict, inplace=True)

        # Set plot titles
        if m in ['D_99', 'D_95']:
            title = 'Better $\longrightarrow$'
            data_to_plot.value *= -1  # Correct the from negative values that are used previously
            alternative_hyp = 'greater'
        else:
            title = '$\longleftarrow$ Better'
            alternative_hyp = 'less'

        # Prepare data (split on OAR and target criteria)
        if m in ['mean', 'D_0.1_cc']:  # OAR criteria
            limits = [-45, 12]
            plt.figure(figsize=(constants.line_width / 1.5, 4.5))
            structure_order = constants.rois_plotting_order['oars']
            plt.xticks(np.arange(-40, 11, 10))

        else:  # Target criteria
            plt.figure(figsize=(constants.line_width / 2.25, 2.5))
            limits = [-10.25, 8]
            structure_order = constants.rois_plotting_order['targets']
            plt.xticks(np.arange(-10, 6, 5))

        # Do mann-whitney u test to test difference between prediction and plans
        pred_values = data_to_plot[data_to_plot['Dose_type'] == 'Prediction']
        plan_values = data_to_plot[data_to_plot['Dose_type'] != 'Prediction']
        p_values = pd.DataFrame(plan_values.groupby(['Structure']).apply(
            lambda x: mannwhitneyu(
                x.value,
                pred_values[pred_values['Structure'] == x.iloc[0, 1]].value,
                alternative=alternative_hyp
            )[1]))  # [1] retrieves the p value from the mann-whitney u test
        # Prepare p value to print in figure
        p_values['p'] = p_values.applymap(lambda x: '{:.3f}'.format(x))
        p_values['equal'] = '$P=$'
        p_values = p_values['equal'].str.cat(p_values['p'])
        p_values = p_values.replace({'$P=$0.000': '$P<$0.001'})

        # Generate box plot
        number_of_structures = plan_values.Structure.unique().shape[0]
        y = np.arange(-0.5, number_of_structures - 0.5, 0.0001)
        ax = sns.boxplot(data=data_to_plot, x='value', y='Structure', showfliers=False, hue='Dose_type',
                         linewidth=1, hue_order=[*constants.optimization_short_hands_dict.values()],
                         order=structure_order, zorder=2, boxprops={'zorder': 2})
        ax.set_ylim((number_of_structures - 0.5, -0.5))
        ax.fill_betweenx(y, limits[0], limits[1], where=np.round(y).__mod__(2) == 1,
                         facecolor='grey', alpha=0.15, zorder=1)

        # Add p values to right axis
        ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
        ax2.set_yticks(ax.get_yticks())
        ax2.set_yticklabels(p_values.values.squeeze())
        ax2.set_ylim(ax.get_ylim())

        # Format figure and save
        ax.axvline(0, ls='--', color='black', linewidth=1.5, zorder=-1)
        ax.set_title(title)
        ax.set_xlim(limits)
        ax.set_xlabel(f'{constants.dvh_metric_axis_dict[m]} (Gy)')
        ax.set_ylabel(None)
        save_plot(constants, f'{m} error', 5, ax=ax)


def make_criteria_satisfaction_plots(names_to_rank: pd.Series, df_dvh_error_to_plot: pd.DataFrame,
                                     file_name_prefix: str, constants: ModelParameters):
    """
    Generate plots to summarize the clinical criteria satisfied
    Args:
        names_to_rank: mapping from prediction name to ranking
        df_dvh_error_to_plot:  criteria that should be plotted
        file_name_prefix: prefix for file to save plot in
        constants: model constants
    """
    # Convert model names to ranking
    number_of_ranks = len(names_to_rank)
    rank_to_names = pd.Series(index=names_to_rank.values, data=names_to_rank.index)

    # Prepare the dataframe to plot
    if file_name_prefix != 'all':
        df_to_plot = df_dvh_error_to_plot[df_dvh_error_to_plot['roi_class'] == file_name_prefix]
    else:
        df_to_plot = df_dvh_error_to_plot

    # Format data for criteria satisfaction
    df_to_plot.replace(dict(names_to_rank), inplace=True)
    ref_criteria = df_to_plot[df_to_plot.Dose_type == 'Reference']
    reference_satisfaction = ref_criteria.value.mean() * 100
    df_to_plot.replace(constants.optimization_short_hands_dict, inplace=True)
    df_to_plot.sort_values(by='Prediction', inplace=True)
    df_to_plot.loc[:, 'Prediction'] = df_to_plot['Prediction'].apply(str)

    # Prepare figure to plot
    plt.figure(figsize=(constants.line_width / 1.25, 3))
    plt.axhline(reference_satisfaction, ls='--', color='black', linewidth=1, zorder=-1)
    df_grouped = df_to_plot.groupby(['Prediction', 'Dose_type']).mean().reset_index()
    df_grouped.value *= 100
    ax = sns.pointplot(
        data=df_grouped, x="Prediction", y="value", hue="Dose_type",
        err_style="bars", ci=95, capsize=.1, dodge=0.75, scale=0.35, errwidth=1,
        hue_order=[*constants.optimization_short_hands_dict.values()], join=False,
        order=rank_to_names.index.astype(str).to_list(), n_boot=1
    )

    # Create grey bars for figure to separate prediction bins
    y = np.arange(-0.5, number_of_ranks - 0.5, 0.0001)
    ax.fill_between(y, 0, 100, where=np.round(y).__mod__(2) == 1,
                    facecolor='grey', alpha=0.15)

    # Format plot
    plt.ylim((22.5, 92.5))
    plt.xlim((-0.6, number_of_ranks - 0.4))  # Put some padding between edge bins
    plt.ylabel(f'Satisfied criteria (%)')
    plt.xlabel(f'Dose score rank')
    save_plot(constants, f'{file_name_prefix}-criteria',
              legend_cols=len(constants.optimization_short_hands_dict.values()))

    # Summarize information for paper related to plots
    satisfied_criteria = df_grouped.groupby(['Prediction', 'Dose_type']).mean().reset_index()
    satisfied_criteria = satisfied_criteria[satisfied_criteria['Dose_type'] != 'Reference']
    most_crit_sat = satisfied_criteria.loc[satisfied_criteria.drop('Dose_type', axis=1).groupby('Prediction').
        idxmax().value].groupby('Dose_type').size()
    print(f'The {file_name_prefix} criteria satisfaction ranged from '
          f'{satisfied_criteria.value.min():.3f} to {satisfied_criteria.value.max():.3f}')
    print(f'The optimization models that satisfied the most {file_name_prefix.upper()} '
          f'criteria were {most_crit_sat.index.to_list()} in {most_crit_sat.values} predictions')


def make_opt_error(cs: ModelParameters, df_dose_error: pd.DataFrame, plotted_error: str, xlim: List or None = None):
    """
    Generate box plot for distribution of error across each KBP model
    Args:
        cs: Model constants
        df_dose_error: The error of each KBP model over all patients
        plotted_error: Type of error being plotted (DVH or dose) that will label plot
        xlim: range on x-axis
    """

    # Prepare data to plot
    df_to_plot = df_dose_error.unstack(0).T.unstack(0)

    # Calculate p value
    score_p_values = df_to_plot.T.apply(
        lambda x: wilcoxon(x, df_to_plot.T['Prediction'], alternative='greater')[1]
        if x.name != 'Prediction' else 1, axis=0, result_type='expand')

    # Prepare p value for plot
    p_values = pd.DataFrame()
    p_values['p'] = score_p_values.apply(lambda x: '{:.3f}'.format(x))
    p_values['equal'] = '$P=$'
    p_values = p_values['equal'].str.cat(p_values['p'])
    p_values = p_values.replace({'$P=$0.000': '$P<$0.001'})

    # Rename dose ranking to bold labels
    y_data = df_dose_error.T.melt(col_level=0)
    y_data.replace(cs.optimization_short_hands_dict, inplace=True)

    # Plot data
    plt.figure(figsize=(cs.line_width / 1.25, 3))  # (0.5*cs.line_width, 2))
    ax = sns.boxplot(data=y_data, x='value', y='Dose_type', showfliers=False, linewidth=1,
                     width=0.65, order=cs.optimization_short_hands_dict.values())
    # Format and label plot
    plt.ylabel(None)
    plt.xlabel(f'{plotted_error} error (Gy)')
    plt.xlim(xlim)

    # Put p values on right axis
    ax2 = ax.twinx()  # instantiate a second axes that shares the same axes
    ax2.set_yticks(ax.get_yticks())
    ax2.set_yticklabels(p_values.values.squeeze())
    ax2.set_ylim(ax.get_ylim())

    # # Highlight the dose with significantly low error with red
    # score_p_values = score_p_values[cs.optimization_short_hands_dict.keys()]
    # for p_idx, ps in enumerate(score_p_values):
    #     if ps < cs.p_thresh:
    #         ax.get_yticklabels()[p_idx].set_color('red')
    #         ax.get_yticklabels()[p_idx].set_weight('bold')

    # Save plot
    save_plot(cs, f'{plotted_error} error')

    # Print out info for paper
    min_median = df_to_plot.median(axis=1).min().round(2)
    min_median_model = df_to_plot.median(axis=1).idxmin()
    print(f'Model {min_median_model} achieved the lowest median error of {min_median}')


def plot_weight_norms(df_objective_data: pd.DataFrame, cs: ModelParameters):
    """
    Generates a boxplot to visualize distribution of weights (not used in paper)
    Args:
        df_objective_data: weight and objective function data
        cs: model constants
    """

    # Prepare data to plot
    df_weights = df_objective_data['Weight']
    df_input_objectives = df_objective_data['Input objective']
    # Prepare strings for axis labels
    weight_column_name = 'Optimized value'
    norm_name = 'Normalization term'
    norm_name_dictionary = {'a_max': '$\max\{\\bf\\alpha$$\}$',
                            'a_sum': '$\\bf\\alpha\'e$',
                            'aCx_max': '$\max\{\\bf{\\alpha\odot C\hat{x}}$$\}$',
                            'aCx_sum': '$\\bf\\alpha\' C\hat{x}$',
                            }

    # Define colour order
    colors = sns.color_palette()[1:]
    sns.set_palette(colors)

    # Calculate aCx sum (for relative max norm = 1)
    aCx = df_input_objectives * df_weights
    aCx.reset_index(inplace=True)
    aCx_sum = aCx.groupby(['Patients', 'Dose_type']).sum()
    aCx_sum_to_plot = aCx_sum.stack().to_frame(norm_name_dictionary['aCx_sum'])
    # aCx max (for relative mean norm <= 1)
    aCx = df_input_objectives * df_weights
    aCx_max = aCx.groupby(['Patients', 'Dose_type']).max()
    aCx_max_to_plot = aCx_max.stack().to_frame(norm_name_dictionary['aCx_max'])
    # a sum (for absolute max norm = 1)
    a = df_weights.copy()
    a.reset_index(inplace=True)
    a_sum = a.groupby(['Patients', 'Dose_type']).sum()
    a_sum_to_plot = a_sum.stack().to_frame(norm_name_dictionary['a_sum'])
    # a_max (for absolute mean norm <= 1)
    a = df_weights
    a_max = a.groupby(['Patients', 'Dose_type']).max().select_dtypes('number')
    a_max_to_plot = a_max.stack().to_frame(norm_name_dictionary['a_max'])

    # Combine all weight norm data to plot
    norms_to_plot = aCx_max_to_plot.join([aCx_sum_to_plot, a_max_to_plot, a_sum_to_plot])
    norms_to_plot = norms_to_plot.T.unstack().reset_index()

    # Clean up strings for plot
    norms_to_plot.replace(cs.optimization_short_hands_dict, inplace=True)
    norms_to_plot.rename(columns={'level_3': norm_name, 0: weight_column_name}, inplace=True)

    # Plot data and save
    sns.boxplot(data=norms_to_plot, x=weight_column_name, y=norm_name, showfliers=False, hue='Dose_type',
                linewidth=1, hue_order=list(cs.optimization_short_hands_dict.values())[1:],
                order=norm_name_dictionary.values())
    plt.axvline(1, ls='--', color='black', linewidth=1, zorder=-1, label=None)
    plt.ylabel(None)
    save_plot(cs, f'normalization_term_plot', legend_cols=4)


def save_plot(cs: ModelParameters, dvh_error_label: str, legend_cols: [int, None] = None,
              ax: [plt.figure, None] = None):
    """
    Saves the plot in a standard format
    Args:
        cs: model constants
        dvh_error_label: Label for file to save
        legend_cols: The number of columns in the legend
        ax: A matplotlib figure that will be saved
    """

    # Prepare name for file to save
    hyphenated_label = dvh_error_label.replace(' ', '-')

    # Get legend for plot
    if legend_cols:
        if ax:
            ax.legend(ncol=legend_cols, frameon=False, bbox_to_anchor=(-1, -1), borderaxespad=0)
            legend = ax.get_legend()
        else:
            legend = plt.legend(ncol=legend_cols, frameon=False, bbox_to_anchor=(-1, -1), borderaxespad=0)
        # Save legend as separate pdf
        fig = legend.figure
        fig.canvas.draw()
        bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(f'{cs.results_dir}/{hyphenated_label}-legend.pdf', bbox_inches=bbox)
        legend.remove()

    # Save full plot
    plt.tight_layout(rect=(0.025, 0, 0.99, 1))
    plt.savefig(f'{cs.results_dir}/{hyphenated_label}-plot.pdf')
    plt.show()
