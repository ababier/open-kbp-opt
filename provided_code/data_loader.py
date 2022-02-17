from typing import List, Tuple

import numpy as np

from provided_code.general_functions import get_paths, load_file


class DataLoader:
    """Generates data for tensorflow"""

    def __init__(self,
                 file_paths_list: List[str],
                 batch_size: int = 2,
                 patient_shape: Tuple[int] = (128, 128, 128),
                 mode_name: str = 'training_model'
                 ):

        """
        Initialize the DataLoader class, which loads the data for OpenKBP
        Args:
            file_paths_list: list of the directories or single files where data for each patient is stored
            batch_size: the number of data points to lead in a single batch
            patient_shape: the shape of the patient data
            mode_name: the data loader mode (changes type of data loaded
        """
        # Set file_loader specific attributes
        self.rois = dict(oars=['Brainstem', 'SpinalCord', 'RightParotid', 'LeftParotid',
                               'Esophagus', 'Larynx', 'Mandible'],
                         targets=['PTV56', 'PTV63', 'PTV70'])
        self.batch_size = batch_size  # Number of patients to load in a single batch
        self.patient_shape = patient_shape  # Shape of the patient
        self.indices = np.arange(len(file_paths_list))  # Indices of file paths
        self.file_paths_list = file_paths_list  # List of file paths
        self.full_roi_list = sum(map(list, self.rois.values()), [])  # make a list of all rois
        self.num_rois = len(self.full_roi_list)
        self.patient_id_list = ['pt_{}'.format(k.split('/pt_')[1].split('/')[0].split('.csv')[0]) for k in
                                self.file_paths_list]  # the list of patient ids with information in this data loader

        # Set files to be loaded
        self.required_files = None
        self.mode_name = mode_name  # Defines the mode for which data must be loaded for
        self.set_mode(self.mode_name)  # Set load mode to prediction by default

    def get_batch(self, index: int or None = None, patient_list: List[str] or None = None) -> dict:
        """
        Loads one batch of data
        Args:
            index: the index of the batch to be loaded
            patient_list: the list of patients for which to load data for

        Returns:
            loaded_data:  a dictionary with the loaded data
        """

        if patient_list is None:
            # Generate batch based on the provided index
            indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        else:
            # Generate batch based on the request patients
            indices = self.patient_to_index(patient_list)

        # Make a list of files to be loaded
        file_paths_to_load = [self.file_paths_list[k] for k in indices]

        # Load the requested files as a tensors
        loaded_data = self.load_data(file_paths_to_load)
        return loaded_data

    def patient_to_index(self, patient_list) -> List[int]:
        """
        Converts a list of patient ids to their appropriate indices
        Args:
            patient_list: list of patient ids

        Returns: list of indices for the requested patients
        """
        return [self.patient_id_list.index(k) for k in patient_list]

    def set_mode(self, mode_name: str) -> None:
        """
        Selects the type of data that is loaded
        Args:
            mode_name: the name of the mode that the data loader is switching to
        """

        self.mode_name = mode_name

        if mode_name == 'training_model':
            # The mode that should be used when training or validing a model
            self.required_files = {'dose': (self.patient_shape + (1,)),  # The shape of dose tensor
                                   'ct': (self.patient_shape + (1,)),  # The shape of ct tensor
                                   'structure_masks': (self.patient_shape + (self.num_rois,)),
                                   # The shape of the structure mask tensor
                                   'possible_dose_mask': (self.patient_shape + (1,)),
                                   # Mask of where dose can be deposited
                                   'voxel_dimensions': (3,)
                                   # Physical dimensions (in mm) of voxels
                                   }
        elif mode_name == 'dose_prediction':
            # The mode that should be used when training or validing a model
            self.required_files = {'ct': (self.patient_shape + (1,)),  # The shape of ct tensor
                                   'structure_masks': (self.patient_shape + (self.num_rois,)),
                                   # The shape of the structure mask tensor
                                   'possible_dose_mask': (self.patient_shape + (1,)),
                                   # Mask of where dose can be deposited
                                   'voxel_dimensions': (3,)  # Physical dimensions (in mm) of voxels
                                   }
            self.batch_size = 1
            print('Warning: Batch size has been changed to 1 for dose prediction mode')

        elif mode_name == 'predicted_dose':
            # This mode loads a single feature (e.g., dose, masks for all structures)
            self.required_files = {mode_name: (self.patient_shape + (1,))}  # The shape of a dose tensor
            self.batch_size = 1
            print('Warning: Batch size has been changed to 1 for dose prediction mode')

        elif mode_name == 'plan_gap':
            # This mode loads a single feature (e.g., dose, masks for all structures)
            self.required_files = {'plan_gap': ()}  # The shape of a dose tensor
            self.batch_size = 1
            print('Warning: Batch size has been changed to 1 for dose prediction mode')

        elif mode_name == 'plan_weights':
            # This mode loads a single feature (e.g., dose, masks for all structures)
            self.required_files = {'plan_weights': ()}  # The shape of a dose tensor
            self.batch_size = 1
            print('Warning: Batch size has been changed to 1 for dose prediction mode')

        elif mode_name == 'evaluation':
            # The mode that should be used evaluate the quality of predictions
            self.required_files = {'dose': (self.patient_shape + (1,)),  # The shape of dose tensor
                                   'structure_masks': (self.patient_shape + (self.num_rois,)),
                                   'voxel_dimensions': (3,),  # Physical dimensions (in mm) of voxels
                                   'possible_dose_mask': (self.patient_shape + (1,)),
                                   }
            self.batch_size = 1
            print('Warning: Batch size has been changed to 1 for evaluation mode')

        elif mode_name == 'optimization':
            # The mode that should be used to run the optimization
            self.required_files = {'structure_masks': (self.patient_shape + (self.num_rois,)),
                                   'dij': (),
                                   'beamlet_indices': (),
                                   'voxel_dimensions': (3,)  # Physical dimensions (in mm) of voxels
                                   }
            self.batch_size = 1
            print('Warning: Batch size has been changed to 1 for evaluation mode')

        else:
            print('Mode does not exist. Please re-run with either \'training_model\', \'prediction\', '
                  '\'predicted_dose\', \'evaluation\', or \'optimization\'')

    def number_of_batches(self) -> int:
        """
        Calculates how many full batches can be made in an epoch

        Returns: the number of batches that can be loaded
        """

        return int(np.floor(len(self.file_paths_list) / self.batch_size))

    def load_data(self, file_paths_to_load: List[str]) -> dict:
        """
        Generates data containing batch_size samples X : (n_samples, *dim, n_channels)

        Args:
            file_paths_to_load: the paths of the files to be loaded:

        Returns :a dictionary of all the loaded files
        """

        # Initialize dictionary for loaded data and lists to track patient path and ids
        tf_data = {}.fromkeys(self.required_files)
        patient_list = []
        patient_path_list = []

        # Loop through each key in tf data to initialize the tensor with zeros
        for key in tf_data:
            # Make dictionary with appropriate data sizes for bath learning
            if self.required_files[key] == ():  # Files with no predefined shape loaded accordingly
                tf_data[key] = [[]] * self.batch_size
            else:
                tf_data[key] = np.zeros((self.batch_size, *self.required_files[key]))

        # Generate data
        for i, pat_path in enumerate(file_paths_to_load):
            # Get patient ID and location of processed data to load
            patient_path_list.append(pat_path)
            pat_id = pat_path.split('/')[-1].split('.')[0]
            patient_list.append(pat_id)
            # Make a dictionary of all the tensors
            loaded_data_dict = self.load_and_shape_data(pat_path)
            # Iterate through the dictionary add the loaded data to the "batch channel"
            for key in tf_data:
                if self.required_files[key] == ():  # Files with no predefined shape loaded accordingly
                    tf_data[key][i] = loaded_data_dict[key]
                else:
                    tf_data[key][i,] = loaded_data_dict[key]

        # Add two keys to the tf_data dictionary to track patient information
        tf_data['patient_list'] = patient_list
        tf_data['patient_path_list'] = patient_path_list

        return tf_data

    def load_and_shape_data(self, path_to_load:str) -> dict:
        """
        Reshapes data that is stored as vectors into matrices
        Args:
            path_to_load: the path of the data that needs to be loaded. If the path is a directory, all data in the
         directory will be loaded. If path is a file then only that file will be loaded.

        Returns: Loaded data with the appropriate shape
        """
        # Initialize the dictionary for the loaded files
        loaded_file = {}
        if '.csv' in path_to_load:
            loaded_file[self.mode_name] = load_file(path_to_load)
        else:
            files_to_load = get_paths(path_to_load, ext='')
            # Load files and get names without file extension or directory
            for f in files_to_load:
                f_name = f.split('/')[-1].split('.')[0]
                if f_name in self.required_files or f_name in self.full_roi_list:
                    loaded_file[f_name] = load_file(f)

        # Initialize matrices for features
        shaped_data = {}.fromkeys(self.required_files)
        for key in shaped_data:
            shaped_data[key] = np.zeros(self.required_files[key])

        # Populate matrices that were no initialized as []
        for key in shaped_data:
            if self.required_files[key] is ():
                # Load files without shape as they are stored
                shaped_data[key] = loaded_file[key]
            elif key == 'structure_masks':
                # Convert dictionary of masks into a tensor (necessary for tensorflow)
                for roi_idx, roi in enumerate(self.full_roi_list):
                    if roi in loaded_file.keys():
                        np.put(shaped_data[key], self.num_rois * loaded_file[roi] + roi_idx, int(1))
            elif key == 'possible_dose_mask':
                np.put(shaped_data[key], loaded_file[key], int(1))
            elif key == 'voxel_dimensions':
                shaped_data[key] = loaded_file[key]
            elif key == 'plan_weights':
                shaped_data[key] = loaded_file[key]
            elif loaded_file[key]['data'].size > 0:  # Files with shape
                np.put(shaped_data[key], loaded_file[key]['indices'], loaded_file[key]['data'])

        return shaped_data
