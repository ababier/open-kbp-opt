from typing import Union

import numpy as np
import pandas as pd
import scipy.spatial as spatial
from scipy import sparse

from provided_code.constants_class import ModelParameters
from provided_code.general_functions import load_file


class Patient:
    def __init__(
        self,
        cs: ModelParameters,
        identifier: str,
        patient_path: str,
        dose: np.array,
        structure_masks: np.array,
        dij: sparse.csr_matrix,
        voxel_dimensions: tuple,
        beamlet_indices: np.array,
        ring_sizes: tuple = (3, 6),
    ) -> None:
        """
        A patient object that contains the attributes of a patient job
        Args:
            cs (ModelParameters): constants in dataset
            identifier: Patient identifier
            patient_path: path where raw data is stored
            dose: The dose for a patient
            structure_masks: A mask for each present ROI
            dij: The dose influence matrix for the patient
            voxel_dimensions (object): The dimensions of any voxel in the patient
            beamlet_indices: The coordinates of the beamlets on a fluence map (rows x columns x angles
            ring_sizes: the size of PTV rings (in same units as voxel dimensions)
        """
        # Set patient attributes based on inputs
        self.cs = cs
        self.identifier = identifier
        self.patient_path = patient_path
        self.dose = dose
        self.structure_masks = structure_masks
        self.dij = dij
        self.voxel_dimensions = voxel_dimensions
        self.beamlet_xyz = beamlet_indices  # name change to convert to convention
        self.ring_sizes = np.array(ring_sizes)
        self.objective_df = None  # Dataframe with objective weights for inverse planning

        # Set extra parameters based on inputs
        cs.set_patient(self.identifier)
        self.number_of_beamlets = self.dij.shape[1]
        self.present_rois = np.any(self.structure_masks, axis=(0, 1, 2))
        # Prepare patient structures for optimization
        self._make_optimization_structs()
        self._sample_voxels()
        # Prepare sampled attributes
        self._get_voxels_of_interest()

    def _make_optimization_structs(self) -> None:
        """
        Generates the artificial optimization structures, which are designed to help with "dose shaping" in areas where there are no rois
        """
        # Prepare attributes to track optimization structures
        self.opt_structures = {}
        self.opt_structures_attributes = pd.DataFrame(columns=["dose_level", "expansion", "ring"])
        # Generate the optimization structures
        self._initialize_patient_cKDTree()  # Create cKD tree to increase efficiency
        self.make_rings()
        self.make_lim_post_neck()
        self.prune_optimization_structures()  # Cleans up structures and removes overlap

    def _initialize_patient_cKDTree(self) -> None:
        """
        Initializes the cKD tree using all patient voxels. This tree is an efficient tool for creating new structures
        based on existing structures.
        """
        # Make patient contour mask
        external_indices = np.argwhere(self.dij.sum(axis=1))[:, 0]  # Patient contour
        # Convert patient contour mask into xyz coordinates
        self.external_xyz = np.array(np.unravel_index(external_indices, self.cs.patient_shape)).T
        # Convert coordinates to mm based on voxel size
        external_xyz_in_mm = self.external_xyz * self.voxel_dimensions
        # Make the cKD tree
        self.external_cKDTree = spatial.cKDTree(external_xyz_in_mm)

    def make_rings(self) -> None:
        """
        Generate rings around the targets to promote high dose gradients/fall off around the targets.
        """
        # Iterate through each target
        for struct in self.cs.rois["targets"]:
            struct_index = self.cs.all_rois.index(struct)
            struct_xyz = np.argwhere(self.structure_masks[:, :, :, struct_index])
            if struct_xyz.size > 0:  # Only continue if the target exists
                struct_xyz_in_mm = struct_xyz * self.voxel_dimensions  # Convert xyz to spatial xyz values
                # Iterate through each ring size
                for ring_size in self.ring_sizes:
                    # Make the ring structure
                    ring_indices = self.make_ring_from_xyz(
                        struct_xyz_in_mm,
                        struct_xyz,
                        ring_size=ring_size,
                    )
                    # Save ring and its corresponding attributes
                    ring_indices = self.strip_voxels(ring_indices)  # Remove target voxels from ring
                    self.opt_structures[f"{struct}_{ring_size}mm_ring"] = ring_indices
                    self.opt_structures_attributes.loc[f"{struct}_{ring_size}mm_ring"] = [float(struct.strip("PTV")), ring_size, 1]

    def make_lim_post_neck(self) -> None:
        """
        Generate a limPostNeck structures
        """
        # Get structure indices for ROI that is used to create limPostNeck
        used_roi_index = self.cs.all_rois.index("SpinalCord")
        roi_xyz = np.argwhere(self.structure_masks[:, :, :, used_roi_index])
        # If no spinal cord is present use the Brainstem to generate a rough approximation
        if roi_xyz.size == 0:
            used_roi_index = self.cs.all_rois.index("Brainstem")
            roi_xyz = np.argwhere(self.structure_masks[:, :, :, used_roi_index])

        # Make a limPostNeck structure based on voxels in roi_xyz
        if roi_xyz.size > 0:
            # Set up ring structure from which LimPostNeck will be constructed
            roi_xyz_in_mm = roi_xyz * self.voxel_dimensions
            ring_indices = self.make_ring_from_xyz(roi_xyz_in_mm, roi_xyz, ring_size=5)
            ring_xyz = np.array(np.unravel_index(ring_indices, self.cs.patient_shape))
            max_xyz = np.max(ring_xyz, axis=-1)
            min_xyz = np.max(ring_xyz, axis=-1)
            # Iterate through of OAR slices and construct limPostNeck
            lim_post_neck = []
            for z_slice in np.unique(self.external_xyz.T[-1]):
                if z_slice in ring_xyz.T:
                    oar_slice = z_slice  # slice of the OAR
                elif z_slice > max_xyz[-1]:
                    oar_slice = max_xyz[-1]  # Highest slice
                else:
                    oar_slice = min_xyz[-1]  # Lowest slice
                # Next check to make sure ring exists inside patient space
                if np.max(ring_xyz[-1] == oar_slice):
                    # Identify points in ring using coordinates at oar_slice
                    max_x = np.max(ring_xyz[1][ring_xyz[-1] == oar_slice])  # highest x coordinate
                    min_x = np.min(ring_xyz[1][ring_xyz[-1] == oar_slice])  # lowest x coordinate
                    mean_y = np.mean(ring_xyz[0][ring_xyz[-1] == oar_slice])  # average y coordinate
                    # Get external slice
                    external_xy = self.external_xyz[self.external_xyz.T[-1] == z_slice].T  # Get external contour at z_slice
                    external_xy_indices = np.ravel_multi_index(external_xy, self.cs.patient_shape)
                    # Identify limPostNeck based on coordinate bounds
                    lim_post_neck_mask = np.all(
                        (
                            external_xy[1] > min_x,  # Above lowest x coordinate
                            external_xy[1] < max_x,  # Below highest x coordinate
                            external_xy[0] > mean_y,
                        ),  # Above highest y coordinate
                        axis=0,
                    )  # mask of limPostNeck based on external_xy_indices
                    lim_post_neck.extend(external_xy_indices[lim_post_neck_mask])

            # Clean up limPostNeck and save it with its attributes
            lim_post_neck = np.array(lim_post_neck).astype(int)
            lim_post_neck = self.strip_voxels(lim_post_neck)  # Remove target voxels from limPostNeck
            self.opt_structures["lim_post_neck"] = lim_post_neck
            self.opt_structures_attributes.loc["lim_post_neck"] = [0, 0, 0]  # Used as a place holder

    def strip_voxels(self, indices, voxels_to_strip=None) -> np.array:
        """
        String away (i.e., remove) the voxels in voxels_to_strip from indices
        Args:
            indices: A set of indices
            voxels_to_strip:  A set of indices that need to be removed from indices. If none are provided then
            the overlap with target voxels will be used
        Returns:
            new_indices: A set of indices with overlap removed between the two inputs
        """
        if voxels_to_strip is None:  # Get target voxels, which will be stripped from input
            first_target_index = self.cs.all_rois.index(self.cs.rois["targets"][0])
            voxels_to_strip = np.argwhere(self.structure_masks[:, :, :, first_target_index:].sum(axis=-1).flatten())
        # Strip the voxels_to_strip from indices
        new_indices = indices[np.in1d(indices, voxels_to_strip, invert=True)]

        return new_indices

    def _sample_voxels(self) -> None:
        """
        Get the set of voxels that are included in the ROIs and the optimization structures
        """
        # Get the mask of voxels that are assigned to an ROI
        voxels_of_interest_mask = self.structure_masks.any(axis=-1)
        all_roi_voxels = np.argwhere(voxels_of_interest_mask.flatten()).squeeze()
        # Get the mask of voxels that are assigned to an optimization structure
        opt_struct_voxels = []
        for opt_struct in self.opt_structures:
            opt_struct_voxels.extend(self.opt_structures[opt_struct])
        # Combine the roi and optimization structures
        self.sampled_voxels = np.concatenate((all_roi_voxels, opt_struct_voxels))
        self.sampled_voxels = np.unique(self.sampled_voxels).astype(int)  # only keep unique values as integers

    def prune_optimization_structures(self) -> None:
        """
        Removes overlap between the optimization structures
        """
        # Set a priority for the structures, voxels will be assigned to their highest priority structure
        priority = self.opt_structures_attributes.sort_values(by=["ring", "dose_level", "expansion"], ascending=[True, False, True])
        for idx, struct in enumerate(priority.index):  # Iterate through optimization structures
            voxels_to_strip = self.opt_structures[struct]
            for lp_struct in priority.index[idx + 1 :]:  # Iterate through lower priority structures
                self.opt_structures[lp_struct] = self.strip_voxels(self.opt_structures[lp_struct], voxels_to_strip)

    def sample_tensor(self, tensor_to_sample: Union[np.ndarray, sparse.csr_matrix]) -> np.array:
        """
        Sample a patient tensor, and keep the corresponding values only where self.sampled_voxels exist
        Args:
            tensor_to_sample: the tensor that will be sampled

        Returns:
            sampled_tensor: the set of values (ordered to match self.sampled_tensor) that were kept
        """
        if type(tensor_to_sample) is np.ndarray:
            flattened_tensor = tensor_to_sample.flatten()
            sampled_tensor = flattened_tensor[self.sampled_voxels]
        else:
            sampled_tensor = tensor_to_sample[self.sampled_voxels]
        return sampled_tensor

    def _get_voxels_of_interest(self) -> None:
        """
        Use sampled voxels to generate new sampled tensors (dose, dij, and structure indices) based on the sampled voxels
        """
        # Sample values (dose and dose influence matrix) for optimization
        self.sampled_dose = self.sample_tensor(self.dose)
        self.sampled_dij = self.sample_tensor(self.dij)
        # Sample structures for optimization
        orig_to_sampled_indices = pd.Series(data=range(len(self.sampled_voxels)), index=self.sampled_voxels)
        self.sampled_structure_masks = {}
        # Sample rois
        for roi_idx, roi_present in enumerate(self.present_rois):
            if roi_present:
                roi = self.cs.all_rois[roi_idx]
                roi_mask = self.structure_masks[:, :, :, roi_idx]
                roi_indices = np.argwhere(roi_mask.flatten()).squeeze()
                self.sampled_structure_masks[roi] = orig_to_sampled_indices[roi_indices].values
        # Sample optimization structures
        for opt_struct in self.opt_structures:
            opt_struct_indices = self.opt_structures[opt_struct]
            if opt_struct_indices.size > 0:
                self.sampled_structure_masks[opt_struct] = orig_to_sampled_indices[opt_struct_indices].values

    def make_ring_from_xyz(self, ring_roi_mm: np.array, ringed_roi: np.array, ring_size=2) -> np.array:
        """
        Make a ring around the given ROI mask.
        Args:
            ring_roi_mm:  the xyz coordinates of an ROI given in mm units
            ringed_roi: the xyz coordinates of an ROI without units (i.e., on an integer grid)
            ring_size: thr width of the ring that will be created around the roi
        Returns:
            only_ring_indices: The indices of the ring
        """
        # Get ring indices in reference to external points
        ring = self.external_cKDTree.query_ball_point(ring_roi_mm, ring_size)
        ring = np.array(np.unique(np.concatenate(ring, axis=0)), dtype=int)
        # Ravel all points to do comparisons
        ring_indices = np.ravel_multi_index(self.external_xyz[ring].T, self.cs.patient_shape)
        roi_indices = np.ravel_multi_index(ringed_roi.T, self.cs.patient_shape)
        # Map ring points (still includes roi) to main coordinate system
        only_ring_indices = ring_indices[np.in1d(ring_indices, roi_indices, invert=True)]

        return only_ring_indices

    def update_weights(self):
        self.objective_df: pd.DataFrame = load_file(self.get_weights_path())
        self.objective_df.set_index("Objective", inplace=True)

    def get_sampled_roi_dose(self, roi):
        return self.sampled_dose[self.sampled_structure_masks[roi]]

    def get_sampled_roi_dij(self, roi):
        return self.sampled_dij[self.sampled_structure_masks[roi]]

    def get_fluence_path(self):
        return f"{self.cs.plan_fluence_from_pred_dir}/{self.identifier}.csv"

    def get_dose_path(self):
        return f"{self.cs.plan_dose_from_pred_dir}/{self.identifier}.csv"

    def get_gap_path(self):
        return f"{self.cs.plan_gap_from_pred_dir}/{self.identifier}.csv"

    def get_weights_path(self):
        return f"{self.cs.plan_weights_from_pred_dir}/{self.identifier}.csv"
