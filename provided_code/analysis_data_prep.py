import os
from itertools import product as it_product
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

from provided_code.constants_class import ModelParameters
from provided_code.data_loader import DataLoader
from provided_code.dose_evaluation_class import EvaluateDose
from provided_code.general_functions import get_paths, get_predictions_to_optimize


def consolidate_data_for_analysis(
    cs: ModelParameters, overwrite_existing_results: bool = False
) -> [pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Consolidated data of all reference plans, dose predictions, and KBP plans. This may take about an hour to run, but
    only needs to be run once for a given set of experiments.

    Args:
        cs: A constants object.
        overwrite_existing_results: Flag that will force consolidating data, which will overwrite previous data that was
         consolidated in previous iterations.
    Returns:
        dose_error: Summary of dose error
        dvh_metrics: Summary of DVH metric performance (can be converted to DVH error later)
        clinical_criteria: Summary of clinical criteria performance
        ref_dvh_metrics: Summary of reference dose DVH metrics
        ref_clinical_criteria: Summary of reference dose clinical criteria performance
        objective_data: The data from the objective functions (e.g., weights, objective function values)
        solve_time: The time it took to solve models
    """

    # Run consolidate_data_for_analysis when new predictions or plans
    consolidate_data_paths = {
        "dose": f"{cs.results_data_dir}/dose_error_df.csv",
        "dvh": f"{cs.results_data_dir}/dvh_metric_df.csv",
        "clinical_criteria": f"{cs.results_data_dir}/clinical_criteria_df.csv",
        "ref_dvh": f"{cs.results_data_dir}/reference_metrics.csv",
        "ref_clinical_criteria": f"{cs.results_data_dir}/reference_criteria.csv",
        "weights": f"{cs.results_data_dir}/weights_df.csv",
        "solve_time": f"{cs.results_data_dir}/solve_time_df.csv",
    }

    # Check if consolidated data already exists
    no_consolidated_date = False
    for p in consolidate_data_paths.values():
        if not os.path.isfile(p):
            print(p)
            no_consolidated_date = True
            os.makedirs(cs.results_data_dir, exist_ok=True)  # Make dir for results

    # Consolidate data if it doesn't exist yet or force flag is True
    if no_consolidated_date or overwrite_existing_results:
        # Prepare strings for data that will be evaluated
        predictions_to_optimize, prediction_names = get_predictions_to_optimize(cs)
        patient_names = os.listdir(cs.reference_data_dir)
        hold_out_plan_paths = get_paths(cs.reference_data_dir, ext="")  # list of paths used for held out testing

        # Evaluate dose metrics
        patient_data_loader = DataLoader(hold_out_plan_paths, mode_name="evaluation")  # Set data loader
        dose_evaluator_sample = EvaluateDose(patient_data_loader)

        # Make reference dose DVH metrics and clinical criteria
        dose_evaluator_sample.make_metrics()
        dose_evaluator_sample.melt_dvh_metrics("Reference", "reference_dose_metric_df").to_csv(consolidate_data_paths["ref_dvh"])
        dose_evaluator_sample.melt_dvh_metrics("Reference", "reference_criteria_df").to_csv(consolidate_data_paths["ref_clinical_criteria"])

        # Initialize DataFrames for all scores and errors
        optimizer_names = os.listdir(cs.plans_dir)  # Get names of all optimizers
        dose_error_index_dict, dvh_metric_index_dict = make_error_and_metric_indices(patient_names, dose_evaluator_sample, optimizer_names)
        df_dose_error_indices = pd.MultiIndex.from_product(**dose_error_index_dict)
        df_dvh_error_indices = pd.MultiIndex.from_arrays(**dvh_metric_index_dict)

        # Make DataFrames
        df_dose_error = pd.DataFrame(columns=prediction_names, index=df_dose_error_indices)
        df_solve_time = pd.DataFrame(columns=prediction_names, index=df_dose_error_indices)
        df_dvh_metrics = pd.DataFrame(columns=prediction_names, index=df_dvh_error_indices)
        df_clinical_criteria = pd.DataFrame(columns=prediction_names, index=df_dvh_error_indices)
        weights_list = []
        weight_columns = []
        # Iterate through each prediction in the list of prediction_names
        for prediction in prediction_names:
            # Make a dataloader that loads predicted dose distributions
            prediction_paths = get_paths(f"{cs.prediction_dir}/{prediction}", ext="csv")
            prediction_dose_loader = DataLoader(prediction_paths, mode_name="predicted_dose")  # Set prediction loader
            # Evaluate predictions and plans with respect to ground truth
            dose_evaluator = EvaluateDose(patient_data_loader, prediction_dose_loader)
            populate_error_dfs(dose_evaluator, df_dose_error, df_dvh_metrics, df_clinical_criteria, prediction, "Prediction")

            # Make dataloader for plan dose distributions
            for opt_name in optimizer_names:
                print(opt_name)
                # Get the paths of all optimized plans for prediction
                cs.get_optimization_directories(prediction, opt_name)
                weights_list, weight_columns = populate_weights_df(cs, weights_list)
                populate_solve_time_df(cs, df_solve_time)
                # Make data loader to load plan doses
                plan_paths = get_paths(cs.plan_dose_from_pred_dir, ext="csv")  # List of all plan dose paths
                plan_dose_loader = DataLoader(plan_paths, mode_name="predicted_dose")  # Set plan dose loader
                plan_evaluator = EvaluateDose(patient_data_loader, plan_dose_loader)  # Make evaluation object
                # Ignore prediction name if no data exists, o/w populate DataFrames
                if not patient_data_loader.file_paths_list:
                    print("No patient information was given to calculate metrics")
                else:
                    # Evaluate prediction errors
                    populate_error_dfs(plan_evaluator, df_dose_error, df_dvh_metrics, df_clinical_criteria, prediction, opt_name)

        # Clean up weights
        weights_df = pd.DataFrame(weights_list, columns=weight_columns)
        weights_df.set_index(["Objective", "Structure", "Patients", "Dose_type", "Prediction"], inplace=True)
        weights_df = weights_df.unstack("Prediction")

        # Save dose and DVH error DataFrames
        df_dose_error.to_csv(consolidate_data_paths["dose"])
        df_dvh_metrics.to_csv(consolidate_data_paths["dvh"])
        df_clinical_criteria.to_csv(consolidate_data_paths["clinical_criteria"])
        weights_df.to_csv(consolidate_data_paths["weights"])
        df_solve_time.to_csv(consolidate_data_paths["solve_time"])

    # Loads the DataFrames that contain consolidated data
    df_dose_error = pd.read_csv(consolidate_data_paths["dose"], index_col=[0, 1])
    df_dvh_metrics = pd.read_csv(consolidate_data_paths["dvh"], index_col=[0, 1, 2, 3])
    df_clinical_criteria = pd.read_csv(consolidate_data_paths["clinical_criteria"], index_col=[0, 1, 2, 3])
    df_ref_dvh_metrics = pd.read_csv(consolidate_data_paths["ref_dvh"], index_col=[0, 1, 2, 3], squeeze=True)
    df_ref_dvh_metrics.index.set_names(df_dvh_metrics.index.names, inplace=True)
    df_ref_clinical_criteria = pd.read_csv(consolidate_data_paths["ref_clinical_criteria"], index_col=[0, 1, 2, 3], squeeze=True)
    df_ref_clinical_criteria.index.set_names(df_clinical_criteria.index.names, inplace=True)
    df_objective_data = pd.read_csv(consolidate_data_paths["weights"], index_col=[0, 1, 2, 3], header=[0, 1])
    df_solve_time = pd.read_csv(consolidate_data_paths["solve_time"], index_col=[0, 1]).drop("Prediction", axis=0, level=0)

    # Adjust DVH metric signs to reflect direction of "better"
    df_dvh_metrics.loc[:, :, ["D_95", "D_99"], :] *= -1
    df_clinical_criteria.loc[:, :, ["D_95", "D_99"], :] *= -1
    df_ref_dvh_metrics.loc[:, :, ["D_95", "D_99"], :] *= -1
    df_ref_clinical_criteria.loc[:, :, ["D_95", "D_99"], :] *= -1

    return df_dose_error, df_dvh_metrics, df_clinical_criteria, df_ref_dvh_metrics, df_ref_clinical_criteria, df_objective_data, df_solve_time


def make_error_and_metric_indices(patient_names: List[str], dose_evaluator_sample: EvaluateDose, optimizers: List[str]) -> [Dict, Dict]:
    """
    Initialize the data frame indices for the dose error and DVH metric DataFrames
    Args:
        patient_names: list of patient names/identifiers
        dose_evaluator_sample: A sample of the dose evaluator object that will be used during the processing stage
        optimizers: list of optimizer names
    Returns:
        dose_error_dict: Dictionaries with stored indices (dose type, patients) for dose error
        dvh_metric_dict: Dictionaries with stored indices (dose types, patients) for DVH metrics
    """

    iterables = [["Prediction", *optimizers], patient_names, dose_evaluator_sample.metric_difference_df.columns]
    iterables_with_tuple = list(it_product(*iterables))
    iterables_new = []
    for i in iterables_with_tuple:
        iterables_new.append((i[0], i[1], i[2][0], i[2][1]))

    dose_error_indices = [iterables[0], iterables[1]]
    dvh_metric_indices = list(zip(*iterables_new))

    # Set names
    dose_error_dict = {"iterables": dose_error_indices, "names": ["Dose_type", "Patients"]}
    dvh_metric_dict = {"arrays": dvh_metric_indices, "names": ["Dose_type", "Patients", "Metric", "Structure"]}

    return dose_error_dict, dvh_metric_dict


def populate_error_dfs(
    evaluator: EvaluateDose,
    df_dose_error: pd.DataFrame,
    df_dvh_metrics: pd.DataFrame,
    df_clinical_criteria: pd.DataFrame,
    prediction_name: str,
    dose_type: str,
):
    """
    Populates the DataFrames that summarize
    Args:
        evaluator: An EvaluateDose Object that will be summarized
        df_dose_error: The DataFrame that contains dose errors
        df_dvh_metrics: The DataFrame that contains DVH metrics
        df_clinical_criteria: THe DataFrame that contains clinical criteria performace
        prediction_name: The name of the prediction model
        dose_type: The type of dose (e.g., reference, prediction, optimization model)
    """
    # Evaluate prediction errors
    evaluator.make_metrics()

    # Save collection of dose errors
    dose_indices = evaluator.dose_score_vec.index
    df_dose_error.loc[(dose_type, dose_indices), prediction_name] = evaluator.dose_score_vec[dose_indices].values

    # Populate the DVH errors
    evaluated_dvh_metrics = evaluator.melt_dvh_metrics(dose_type)
    df_dvh_metrics.loc[evaluated_dvh_metrics.index, prediction_name] = evaluated_dvh_metrics.values

    # Populate clinical criteria metrics
    evaluated_clinical_criteria = evaluator.melt_dvh_metrics(dose_type, dose_metrics_att="new_criteria_metric_df")
    df_clinical_criteria.loc[evaluated_clinical_criteria.index, prediction_name] = evaluated_clinical_criteria.values


def populate_weights_df(cs: ModelParameters, weights_list) -> [List, List]:
    """
    Populated a list (weights_list) with data related to cost function (e.g., structure, objective function values)
    Args:
        cs: Constant object
        weights_list: List of weights that will be populated

    Returns:
        weights_list: List of populated weights
        weights_list_column_headers: Column headers for list
    """
    # Initialize information for plan weights
    plan_weights_paths = get_paths(cs.plan_weights_from_pred_dir, ext="csv")
    plan_weights_loader = DataLoader(plan_weights_paths, mode_name="plan_weights")
    weights_list_column_headers = []
    # Load weight info for each patient
    for batch_idx in range(plan_weights_loader.number_of_batches()):
        data_batch = plan_weights_loader.get_batch(batch_idx)
        pt_id = data_batch["patient_list"][0]
        plan_weights = data_batch["plan_weights"][0]
        # Separate objective function from structure
        roi_criteria_pairs = plan_weights.apply(lambda x: pd.Series(x["Objective"].split(" ", 1)), axis=1)
        plan_weights["Structure"] = roi_criteria_pairs[0]
        plan_weights["Objective"] = roi_criteria_pairs[1]
        # Adjust plan weights DataFrame with plan/patient data
        plan_weights["Patients"] = pt_id
        plan_weights["Dose_type"] = cs.opt_name
        plan_weights["Prediction"] = cs.prediction_name
        # Extend weight data to weight list
        weights_list.extend(plan_weights.values.tolist())
        weights_list_column_headers = plan_weights.columns.to_list()

    return weights_list, weights_list_column_headers


def populate_solve_time_df(cs: ModelParameters, df_solve_time: pd.DataFrame):
    """
    Populated a DataFrame (solve_time) with data related to solve time and plan (optimization) gap
    Args:
        cs: Constants object
        df_solve_time: DataFrame with solve time information
    """
    # Initialize plan gap/solve time information
    plan_gap_paths = get_paths(cs.plan_gap_from_pred_dir, ext="csv")
    plan_gap_loader = DataLoader(plan_gap_paths, mode_name="plan_gap")
    # Load solve time/gap for each patient
    for batch_idx in range(plan_gap_loader.number_of_batches()):
        data_batch = plan_gap_loader.get_batch(batch_idx)
        pt_id = data_batch["patient_list"][0]
        plan_gap = data_batch["plan_gap"][0]
        # Populate summary dataframe with time/gap info
        df_solve_time.loc[(cs.opt_name, pt_id), cs.prediction_name] = plan_gap["solve time"]


def summarize_scores(cs: ModelParameters, df_errors: pd.DataFrame, name: str, level=0) -> pd.DataFrame:
    """


    Args:
        cs: Model constants
        df_errors: DataFrame of errors that can be converted into a score by taking average for every prediction/opt
        name: Name of score that will be generated
        level: Level of df_errors that average is calculated over

    Returns:
        ranked_scores:
    """

    # Calculate scores
    score = round(df_errors.mean(axis=0, level=level), 3)
    score = score.loc[cs.optimization_short_hands_dict.keys()]
    rank = score.rank(axis=1)

    # Set order based on prediction rank
    sorted_scores = score.sort_values(by="Prediction", axis=1).columns
    # Rename index prior to concatenating the data
    score.index = score.index.map(lambda x: f"{x} {name.lower()} score")
    rank.index = rank.index.map(lambda x: f"{x} {name.lower()} rank")
    # Concat scores and rank, ordered based on prediction rank
    ranked_scores = pd.concat((score[sorted_scores], rank[sorted_scores])).T

    # Alternate between score and rank
    score_rank_column_order = []
    for idx, dose_type in enumerate(score.index):
        score_rank_column_order.extend(ranked_scores.columns[idx :: len(score.index)].to_list())
    ranked_scores = ranked_scores[score_rank_column_order]

    # Convert ranks to integers
    ranked_scores[score_rank_column_order[1::2]] = ranked_scores[score_rank_column_order[1::2]].astype(int)

    # Prep ranked scores for table
    ranked_scores_for_csv = ranked_scores.copy(deep=True)
    ranked_scores_for_csv[score_rank_column_order[2::2]] = ranked_scores_for_csv[score_rank_column_order[2::2]].subtract(
        ranked_scores_for_csv[f"Prediction {name.lower()} score"].values, axis=0
    )
    ranked_scores_for_csv.set_index(f"Prediction {name.lower()} rank", drop=False, inplace=True)
    ranked_scores_for_csv[score_rank_column_order[1::2]] = ranked_scores_for_csv[score_rank_column_order[1::2]].applymap(lambda x: f" ({str(x)})")
    ranked_scores_for_csv[score_rank_column_order[0::2]] = ranked_scores_for_csv[score_rank_column_order[0::2]].applymap(lambda x: f"{x:.2f}")
    ranked_scores_for_csv[score_rank_column_order[0::2]] = (
        ranked_scores_for_csv[score_rank_column_order[0::2]].values + ranked_scores_for_csv[score_rank_column_order[1::2]].values
    )
    ranked_scores_for_csv = ranked_scores_for_csv[score_rank_column_order[0::2]]
    # Save as csv
    ranked_scores_for_csv.to_csv(f"{cs.results_data_dir}/{name}.csv")

    return ranked_scores


def save_score_summary(cs: ModelParameters, df_errors: pd.DataFrame, name: str, level=0):
    """
    Save table that contains ank order correlations between prediction score and KBP pipeline score
    Args:
        cs: Model constants
        df_errors: DataFrame of errors that can be converted into a score by taking average for every prediction/opt
        name: Name of score that will be generated
        level: Level of df_errors that average is calculated over
    """

    # Calculate scores
    score = df_errors.mean(axis=0, level=level)
    score = score.loc[cs.optimization_short_hands_dict.keys()].T
    rank = score.rank(axis=0)  # Generate rank pandas series

    #
    # Test rank order
    spearman_test_results = rank.apply(lambda x: spearmanr(x, rank[f"Prediction"].values))
    spearman_test_results.rename(index={0: "Correlation", 1: "$P$-value"}, inplace=True)
    summary_df = spearman_test_results
    summary_df.drop("Prediction", inplace=True, axis=1)
    summary_df.rename(columns=cs.optimization_short_hands_dict, inplace=True)
    summary_df = summary_df.applymap(lambda x: f"{x:.2f}")
    summary_df.to_latex(f"{cs.results_data_dir}/{name}-correlation.tex", escape=False)


def make_criteria_satisfaction_df(cs: ModelParameters, df_clinical_criteria: pd.DataFrame, df_ref_clinical_criteria: pd.DataFrame) -> pd.DataFrame:
    """
    Make a DataFrame with all clinical criteria satisfaction
    Args:
        cs: Model constants
        df_clinical_criteria: Clinical criteria metrics from KBP generated dose (i.e., plans and predictions)
        df_ref_clinical_criteria: Clinical criteria metrics from reference dose

    Returns:
        df: Dataframe of clinical criteria satisfaction for each type of dose
    """
    # Prep clinical criteria goals (i.e., what number needs to be achieved to pass criteria)
    criteria_series = pd.Series(cs.plan_criteria_dict)
    # Concatenate reference dose (duplicated because it's the same for each set)
    reference_duplicates = np.array([df_ref_clinical_criteria.values] * len(df_clinical_criteria.columns)).T
    reference_duplicates_df = pd.DataFrame(reference_duplicates, index=df_ref_clinical_criteria.index, columns=df_clinical_criteria.columns)
    # Make df with all criteria
    df = df_clinical_criteria.append(reference_duplicates_df)
    # Calculate satisfaction
    df = df.unstack(0).reset_index(0).loc[criteria_series.index]
    df.dropna(axis=0, inplace=True)
    # Evaluate criteria satisfaction
    df.iloc[:, 1:] = df.iloc[:, 1:].le(criteria_series[df.index], axis=0).astype(float)
    # Format df which now contains 1s and 0s to indicate pass/fail
    df = df.reset_index().set_index(["Metric", "Structure", "Patients"])
    df = df.melt(ignore_index=False).reset_index()
    df.rename(columns={None: "Prediction"}, inplace=True)

    # Label rois by class
    roi_type_series = pd.Series(dict((i, k) for k, v in cs.rois.items() for i in v))
    df["roi_class"] = roi_type_series[df["Structure"]].values

    return df


def make_criteria_satisfaction_summary(cs: ModelParameters, df: pd.DataFrame) -> [pd.DataFrame]:
    """
    Summarizes the clinical criteria satisfaction of all models and the "best" pipeline generated
    Args:
        cs: Model constants
        df: A DataFrame containing criteria satisfaction across all criteria, dose types, and patients

    Returns:
        criteria_sat_summary: Percentage of criteria satisfied by each type of dose
    """

    # Determine pipeline that satisfied most criteria
    best_pipeline_index = df.groupby(["Prediction", "Dose_type"]).mean().idxmax()
    best_pipeline = df.set_index(["Prediction", "Dose_type"]).loc[best_pipeline_index]
    best_pipeline.reset_index(inplace=True)
    best_pipeline.Prediction = "Best"
    best_pipeline.Dose_type = "Best"
    df_with_best = pd.concat([df, best_pipeline])

    # Calculate proportion of criteria satisfied for each roi class
    crit_summary = df_with_best.groupby(["Structure", "Dose_type"]).mean()
    all_roi_criteria_sat = df_with_best.groupby(["Dose_type"]).mean()  # all criteria
    roi_class_criteria_sat = df_with_best.groupby(["roi_class", "Dose_type"]).mean()
    criteria_sat_summary = pd.concat(
        (
            crit_summary.unstack(1),
            roi_class_criteria_sat.unstack(1),
        ),
        axis=0,
        sort=False,
    )
    criteria_sat_summary = criteria_sat_summary.append(all_roi_criteria_sat.unstack(0).rename("AllCriteria"))
    criteria_sat_summary = criteria_sat_summary.droplevel(0, axis=1)[["Reference", *cs.optimization_short_hands_dict.keys(), "Best"]]

    # prep crits for table
    criteria_sat_summary = criteria_sat_summary.round(3) * 100
    max_row_values = criteria_sat_summary.max(axis=1).apply(lambda x: f"{x:.1f}")
    criteria_sat_summary = criteria_sat_summary.applymap(lambda x: f"{x:.1f}")
    criteria_sat_summary = criteria_sat_summary.apply(lambda x: x.replace(max_row_values[x.name], f"\textbf{{{max_row_values[x.name]}}}"), axis=1)
    criteria_sat_summary = criteria_sat_summary.loc[[*cs.all_rois, "oars", "targets", "AllCriteria"]]
    # Clean up strings and save
    criteria_sat_summary.rename(index=cs.structure_printing, columns=cs.optimization_short_hands_dict, inplace=True)
    criteria_sat_summary.to_latex(f"{cs.results_dir}/criteria.tex", escape=False)


def summarize_objective_weights(cs: ModelParameters, df_objective_data: pd.DataFrame) -> pd.DataFrame:
    """
    Summarizes the proportion of weight that each optimization model assigned to each class of structure
    Args:
        cs: Model constants
        df_objective_data: Objective weight data

    Returns:
        average_weights: Data frame the contains proportion of weights assigned
    """

    # Consolidate objective weight data based on roi types
    df_weights_stacked = df_objective_data["Weight"].stack().to_frame("Weight")
    df_weights_stacked.reset_index(inplace=True)
    all_objective_structures = df_weights_stacked.Structure.unique()
    df_weights_stacked.Structure.replace(cs.roi_series, inplace=True)
    df_weights_stacked.Structure.replace(all_objective_structures, "opt", inplace=True)

    # Calculate proportion of weight assigned to each roi type
    average_weights = df_weights_stacked.groupby(["Structure", "Dose_type"]).mean().unstack()
    average_weights = average_weights / average_weights.sum()
    average_weights = average_weights.droplevel(0, 1).round(3)
    average_weights = average_weights.applymap(lambda x: f"{x:.3f}")

    return average_weights


def summarize_solve_time_distribution(df_solve_time: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a dataframe to summarize distribution of solve times for each model
    Args:
        df_solve_time: Dataframe of all solve times for each model and patient

    Returns:
        df: Summary of solve time distribution
    """

    # Get points in distribution
    average_solve_time = df_solve_time.unstack().T.mean()
    first_quartile_solve_time = df_solve_time.unstack().T.quantile(0.25)
    third_quartile_solve_time = df_solve_time.unstack().T.quantile(0.75)

    # Merge data into single DataFrame
    df = pd.concat((average_solve_time, first_quartile_solve_time, third_quartile_solve_time), axis=1)
    df.columns = ("Mean solve time", "First quartile solve time", "Third quartile solve time")
    df = df.T.round().astype(int)
    df = df.applymap(lambda x: f"{x:.0f}")

    return df


def make_meta_data_table(cs: ModelParameters, meta_data_df: List[pd.DataFrame]):
    """
    Generate a table of meta data
    Args:
        cs: Model constants
        meta_data_df: List of DataFrames that contians meta data for each optimization model
    """

    # Merge into a single DataFrame
    meta_table = meta_data_df[0]
    for d in meta_data_df[1:]:
        meta_table = pd.concat((meta_table, d))

    # Make sure order is consistent
    opt_order = list(cs.optimization_short_hands_dict.keys())
    meta_table_columns = list(meta_table.columns)
    opt_order = [o_name for o_name in opt_order if o_name in meta_table_columns]
    meta_table = meta_table[opt_order]

    # Rename indices and columns
    meta_table.rename(index=cs.structure_printing, columns=cs.optimization_short_hands_dict, inplace=True)
    meta_table.to_latex(f"{cs.results_dir}/optimization_meta_data.tex")
