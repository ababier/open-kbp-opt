import os

from provided_code.analysis_data_prep import (
    consolidate_data_for_analysis,
    make_criteria_satisfaction_df,
    make_criteria_satisfaction_summary,
    make_meta_data_table,
    save_score_summary,
    summarize_objective_weights,
    summarize_scores,
    summarize_solve_time_distribution,
)
from provided_code.constants_class import ModelParameters
from provided_code.plotting import make_criteria_satisfaction_plots, make_dvh_metric_diff_plots, make_opt_error, plot_weight_norms

cs = ModelParameters(io_name="baseline")

if __name__ == "__main__":

    # Define model parameters and directories
    os.makedirs(cs.results_dir, exist_ok=True)

    # Get consolidated data that is used to generate results
    (
        df_dose_error,
        df_dvh_metrics,
        df_clinical_criteria,
        df_ref_dvh_metrics,
        df_ref_clinical_criteria,
        df_objective_data,
        df_solve_time,
    ) = consolidate_data_for_analysis(cs, overwrite_existing_results=False)

    # Calculate DVH metric errors
    df_dvh_signed_error = df_dvh_metrics.subtract(df_ref_dvh_metrics.Reference, axis=0)
    df_dvh_signed_error = df_dvh_signed_error.reorder_levels(df_ref_dvh_metrics.index.names)
    df_dvh_max_signed_error = df_dvh_signed_error.groupby(level=(0, 1)).max()
    df_dvh_mean_signed_error = df_dvh_signed_error.groupby(level=(0, 1)).mean()
    df_dvh_error = df_dvh_signed_error.abs()

    # Calculate score metrics
    df_dose_score = summarize_scores(cs, df_dose_error, name="Dose")
    df_dvh_score = summarize_scores(cs, df_dvh_error, name="DVH")

    # Make pandas series to map team name to rank
    name_to_dose_rank = df_dose_score["Prediction dose rank"]
    name_to_dvh_rank = df_dvh_score["Prediction dvh rank"]

    # Results 1.A - Dose score correlation
    save_score_summary(cs, df_dose_error, name="dose")  # Table 4
    # Results 1.B - Dose score differences for each kbp method
    make_opt_error(cs, df_dose_error, "Dose", xlim=[-0.25, 7.25])  # Figure 4

    # Results 2 - DVH metrics
    make_dvh_metric_diff_plots(df_dvh_signed_error, cs)  # Figure 5a-5e

    # Results 3 - Clinical criteria
    criteria_sat_df = make_criteria_satisfaction_df(cs, df_clinical_criteria, df_ref_clinical_criteria)
    criteria_summary = make_criteria_satisfaction_summary(cs, criteria_sat_df)  # Table 5
    make_criteria_satisfaction_plots(name_to_dose_rank, criteria_sat_df, "oars", cs)  # Figure 6a
    make_criteria_satisfaction_plots(name_to_dose_rank, criteria_sat_df, "targets", cs)  # Figure 6b
    make_criteria_satisfaction_plots(name_to_dose_rank, criteria_sat_df, "all", cs)  # Figure 6c

    # Results 4 - Meta data analysis
    weight_proportion = summarize_objective_weights(cs, df_objective_data)
    solve_time_dist = summarize_solve_time_distribution(df_solve_time)
    make_meta_data_table(cs, [weight_proportion, solve_time_dist])  # Table 7
    plot_weight_norms(df_objective_data, cs)  # Weights distribution (not in paper)
