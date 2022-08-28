import os
import pathlib
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, load_npz

from provided_code.constants_class import ModelParameters


def load_file(file_name: str) -> Union[dict, pd.DataFrame, coo_matrix, np.array]:
    """Load a file in one of the formats provided in the OpenKBP dataset. This function is overloaded and should be broken up into several
    functions that return one type each.
    :param file_name: the name of the file to be loaded
    :return: the file loaded
    """

    file_extension = file_name.split(".")[-1]

    if file_extension == "csv":
        # Check if csv file contains plan weights, which saved in different style than other files
        if "/plan-weights/" in file_name:
            loaded_file = pd.read_csv(file_name, index_col=None, names=["Objective", "Weight", "Optimized objective", "Input objective"])
        elif "/plan-gap/" in file_name:
            loaded_file = pd.read_csv(file_name, index_col=0, header=None, squeeze=True)
        else:
            # Load the file as a csv
            loaded_file_df = pd.read_csv(file_name, index_col=0)
            # If the csv is voxel dimensions read it with numpy
            if "voxel_dimensions.csv" in file_name:
                loaded_file = np.loadtxt(file_name)
            elif "beamlet_indices.csv" in file_name:
                # Convert row index labels to a df column
                loaded_file = loaded_file_df.reset_index()  # Check if the data has any values
            elif loaded_file_df.isnull().values.any():
                # Then the data is a vector, which we assume is for a mask of ones
                loaded_file = np.array(loaded_file_df.index).squeeze()
            else:
                # Then the data is a matrix of indices and data points
                loaded_file = {"indices": np.array(loaded_file_df.index).squeeze(), "data": np.array(loaded_file_df["data"]).squeeze()}

    elif file_extension == "npz":
        loaded_file = load_npz(file_name)

    else:
        loaded_file = None

    return loaded_file


def get_paths(directory_path: str, ext: str = "") -> List[str]:
    """
    Get the paths of every file with a specified extension in a directory
    Args:
        directory_path: the path of the directory of interest
        ext: the extensions of the files of interest

    Returns: the path of all files of interest

    """
    # if dir_name doesn't exist return an empty array
    if not os.path.isdir(directory_path):
        return []
    # Otherwise dir_name exists and function returns contents name(s)
    else:
        all_image_paths = []
        # If no extension given, then get all files
        if ext == "":
            dir_list = os.listdir(directory_path)
            for iPath in dir_list:
                if "." != iPath[0]:  # Ignore hidden files
                    all_image_paths.append("{}/{}".format(directory_path, str(iPath)))
        else:
            # Get list of paths for files with the extension ext
            data_root = pathlib.Path(directory_path)
            for iPath in data_root.glob("*.{}".format(ext)):
                all_image_paths.append(str(iPath))

    return all_image_paths


def sparse_vector_function(x, indices=None):
    """
    Convert a tensor into a dictionary of the non zero values and their corresponding indices
    Args:
        x: the tensor or, if indices is not None, the values that belong at each index
        indices: the raveled indices of the tensor

    Returns: sparse vector in the form of a dictionary

    """

    non_zero_indices = np.nonzero(x.flatten())[-1]
    if indices is None:
        y = {"data": x[non_zero_indices], "indices": non_zero_indices}
    else:
        y = {"data": x[non_zero_indices], "indices": indices[non_zero_indices]}
    return y


def get_predictions_to_optimize(cs: ModelParameters, set_names_to_omit: Tuple[str] = ("set_20", "set_21")) -> (pd.Series, list):
    """
    Args:
        cs: Model parameter constants for project
        set_names_to_omit: names of predictions that should be omitted
    Returns:
        predictions_to_optimize: path of the directory containing prediction
        prediction_names: list of names corresponding to each prediction set
    """
    set_of_predictions = pd.Series(get_paths(cs.prediction_dir, ext=""))  # Get the names of each prediction
    if set_of_predictions.shape[0] == 0:
        raise ValueError("No predictions found. If data kept outside of project directory, change `parent_data_directory` in `ModelParameters`.")
    omit_predictions_mask = set_of_predictions.str.contains("|".join(set_names_to_omit))
    predictions_to_optimize = set_of_predictions[~omit_predictions_mask].sort_values()
    prediction_names = []
    for path in predictions_to_optimize:
        prediction_names.append(path.split("/")[-1])
    return predictions_to_optimize, prediction_names
