import time
from typing import Union

import numpy as np
import pandas as pd
from ortools.linear_solver import pywraplp
from scipy import sparse

from provided_code.constants_class import ModelParameters
from provided_code.general_functions import sparse_vector_function
from provided_code.resources import Patient


def sparse_times_variable_vector(sparse_mat: sparse.csr_matrix):
    """
    Args:
        sparse_mat: a sparse matrix

    Returns:
        i: A row index
        sparse_mat.indices[start:end]: the column indices that have non-zero values
        sparse_mat.data[start:end]: the non-zero values across row i
    """
    for i in range(sparse_mat.shape[0]):  # Iterate through each row
        start = sparse_mat.indptr[i]  # First index in sparse matrix from row i
        end = sparse_mat.indptr[i + 1]  # Last index in sparse matrix from row i
        yield i, sparse_mat.indices[start:end], sparse_mat.data[start:end]


class PlanningModel:
    def __init__(self, patient: Patient, cs: ModelParameters, relative_or_absolute: str = 'relative',
                 mean_or_max: str = 'max', inverse_plan: bool = False) -> None:
        """
        Class that contains the optimization model. Use the methods to build a model using OR-tools.
        Args:
            patient: A patient object that contains all the information about the patient that we are optimizing
            a plan for, which is based on the dose attribute (should be a predicted dose)
            cs: A constants object.
            relative_or_absolute: Set whether to evaluate relative or absolute differences
            mean_or_max: Sets whether to minimize the max ar mean deviation
        """

        # Initialize class inputs
        self.patient = patient
        self.cs = cs
        self.relative_or_absolute = relative_or_absolute
        self.mean_or_max = mean_or_max
        self.inverse_plan = inverse_plan
        self.constraint_names = []
        self.input_objective_values = []

        # Initialize solver and necessary components
        self._solver = pywraplp.Solver.CreateSolver('GUROBI_LP')
        self._objective = self._solver.Objective()
        self.dummy_variable_counter = 0

        # c_vat objectives
        self.oar_c_vars = {0.975: 1, 0.9: 1, 0.75: 1, 0.5: 1, 0.25: 1}
        self.tar_c_vars = {1.05: 1, 1: -1}

        # Build the model
        self._set_variables()
        self._set_constraints()
        self._set_objective()
        self.solve_time = None

    def _set_variables(self) -> None:
        """
        Set all the variables for the problem and them directly to the optimization model object
        """

        # Initialize model variables
        self.w = {}  # Continuous beamlet intensities
        self.d = {}  # Continuous dose
        self.dose_constraint = {}

        # Iterate through feasible employees and jobs
        for beamlet_num in range(self.patient.number_of_beamlets):
            self.w[beamlet_num] = self._solver.NumVar(0, self.cs.max_beam, name=f'beamlet_{beamlet_num}')
        for voxel_num in range(self.patient.sampled_voxels.shape[0]):
            self.d[voxel_num] = self._solver.NumVar(0, self.cs.max_dose, name=f'dose_{voxel_num}')
            self.dose_constraint[voxel_num] = self._solver.Constraint(0, 0)

        # Set variables for objective function
        if self.mean_or_max == 'max' and not self.inverse_plan:
            self.obj_variable = self._solver.NumVar(-pywraplp.Solver_infinity(), pywraplp.Solver_infinity(),
                                                    'obj_variable')
            self._objective.SetCoefficient(self.obj_variable, 1)
        else:
            self.obj_variable = None

    def _set_constraints(self):
        """
        Set all constraints for the optimization model.
        Returns: Constraints for model
        """
        self._fluence_to_dose()
        self.add_spg_constraint()
        # Set constraints to define the dose objective

    def _fluence_to_dose(self):
        """
        Maps fluence intensity (decision variables)
        """

        for voxel, w_indices, dij_data in sparse_times_variable_vector(self.patient.sampled_dij):
            self.dose_constraint[voxel].SetCoefficient(self.d[voxel], -1)
            for dij_idx, beamlet_idx in enumerate(w_indices):
                self.dose_constraint[voxel].SetCoefficient(self.w[beamlet_idx], dij_data[dij_idx])

    def _set_objective(self):
        """
        Sets all the objectives for the model
        Returns:
        """
        for struct in self.patient.sampled_structure_masks:
            if struct not in self.cs.rois['targets']:
                self.mean_obj(struct)
                self.max_obj(struct)
                self.c_var_obj(struct, self.oar_c_vars, self.patient.get_sampled_roi_dose(struct).max())
            else:
                # self.mean_obj(struct)
                self.max_obj(struct)
                self.c_var_obj(struct, self.tar_c_vars, float(struct.split('PTV')[-1]))

        self._objective.SetMinimization()

    def solve(self, quick_test: bool = False):
        """
        Solves the optimization model and returns a list of matches.
        Args:
            quick_test: if true, the convergence tolerance is relaxed to generate a (low-quality) solution quickly.
            Should only be used to validate code runs as expected because the generated plans will be very suboptimal.
        """

        # Set model parameters (may need to be adjusted if a solver other than Gurobi is used)
        self._solver.EnableOutput()  # Makes verbose
        self._solver.SetSolverSpecificParametersAsString('Crossover 0')
        self._solver.SetSolverSpecificParametersAsString('Method 2')

        # Define how close to optimality solver will strive for (lower is more optimal)
        if quick_test:
            self._solver.SetSolverSpecificParametersAsString('BarConvTol 1')
        else:
            self._solver.SetSolverSpecificParametersAsString('BarConvTol 0.00001')

        # Start timer
        solve_start_time = time.time()
        self._solver.Solve()  # Solve model (this should take about 300 second on average)
        self.solve_time = time.time() - solve_start_time  # calculate solve time

    # Objectives
    def obj_norm(self, input_dose_obj: Union[float, int, np.ndarray], opt_dose_obj: pywraplp.Variable,
                 objective_name: str) -> None:
        """
        Calculate the difference between the optimized dose objective and its corresponding predicted dose objective,
        differences are added
        Args:
            input_dose_obj: the input dose objective value (constant)
            opt_dose_obj: the dose objective for the dose being optimized

        """
        # Calculate difference for input dose objective
        if self.relative_or_absolute == 'relative':  # evaluate relative difference
            if input_dose_obj <= 1e-5:  # If input objective is very small make the input a constraint to help stability
                self.constraint_names.remove(objective_name)
                self._solver.Constraint(0, input_dose_obj).SetCoefficient(opt_dose_obj, 1)
                return  # Nothing to add to objective in this case

        # Construct objective function
        if self.inverse_plan:  # inverse planning
            obj_weight = self.patient.objective_df.loc[objective_name, 'Weight']
            self._objective.SetCoefficient(opt_dose_obj, obj_weight)  # add to objective function
            return

        if self.mean_or_max == 'max':
            if self.relative_or_absolute == 'relative':  # evaluate lhs for relative max difference
                lhs = pywraplp.ProductCst(self.obj_variable, input_dose_obj)
                self._solver.Add(input_dose_obj + lhs >= opt_dose_obj, name=objective_name)  # does max(obj, 0)

            elif self.relative_or_absolute == 'absolute':  # evaluate lhs term for absolute max difference
                lhs = pywraplp.SumCst(self.obj_variable, input_dose_obj)
                self._solver.Add(lhs >= opt_dose_obj, name=objective_name)  # does max(obj, 0)

        elif self.mean_or_max == 'mean':
            sigma = self.add_dummy_variable(lb=0)  # variable that holds one sided dose objective difference
            psi = self.add_dummy_variable(lb=0)
            if self.relative_or_absolute == 'relative':
                lhs = pywraplp.ProductCst(sigma, input_dose_obj)
                psi_for_con = pywraplp.ProductCst(psi, input_dose_obj)
                self._solver.Add(input_dose_obj + lhs - psi_for_con == opt_dose_obj, name=objective_name)

            elif self.relative_or_absolute == 'absolute':
                lhs = pywraplp.SumCst(sigma, input_dose_obj)
                self._solver.Add(lhs - psi == opt_dose_obj, name=objective_name)  # does max(obj, 0)

            self._objective.SetCoefficient(sigma, 1)  # add to objective function
            self._objective.SetCoefficient(psi, -1e-5)  # add to objective function

        self.input_objective_values.append(input_dose_obj)

    def add_dummy_variable(self, lb: float = -pywraplp.Solver_infinity(),
                           ub: float = pywraplp.Solver_infinity()) -> pywraplp.Variable:
        """
        Creates a variable with a name that is equal to the number of dummy/auxiliary variables that have been created.
        Function exists to make it easier to add dummy variables with unique name.

        Args:
            lb: lower bound of variable
            ub: upper bound of variable

        Returns:
            var: variable for optimization model
        """
        var = self._solver.NumVar(lb, ub, str(self.dummy_variable_counter))
        self.dummy_variable_counter += 1
        return var

    def var_1_le_var_2(self, var_1: pywraplp.Variable, var_2: pywraplp.Variable) -> None:
        """
        Creates a constraint that forces var_1 <= var_2 (faster than self._solver.Add(var_1 < var_2)
        Args:
            var_1: variable 1
            var_2: variable 2
        """
        ct = self._solver.Constraint(0, pywraplp.Solver_infinity())
        ct.SetCoefficient(var_1, -1)
        ct.SetCoefficient(var_2, 1)

    def mean_obj(self, roi):

        # Name for objective
        objective_constraint_name = f'{roi} mean dose'
        self.constraint_names.append(objective_constraint_name)

        # Get sampled roi mask
        roi_mask = self.patient.sampled_structure_masks[roi]

        # Calculated the mean roi input dose
        mean_input_dose = self.patient.get_sampled_roi_dose(roi).mean()

        # Calculate the optimized mean dose
        mean_opt_dose_var = self.get_average_as_variable(self.d, objective_constraint_name, roi_mask)
        self.obj_norm(mean_input_dose, mean_opt_dose_var, objective_constraint_name)

    def get_average_as_variable(self, array_to_average: dict, objective_name: str,
                                mask_to_sum: np.ndarray = None) -> pywraplp.Variable:
        """
        Takes the mean of the variables stored in a dictionary.
        Args:
            array_to_average: array of variables that will be average
            objective_name: name of the dose objective that corresponds to this variavle
            mask_to_sum: The mask corresponding to voxels the average should be taken over. If none, the full array to
             sum will be used
        Returns:
            array_mean_var: a variable that is equal to the mean of the input array_to_average
        """
        if mask_to_sum is None:
            mask_to_sum = np.array(list(array_to_average.keys()))
        array_sum = pywraplp.SumArray(array_to_average[i] for i in mask_to_sum)
        array_mean = pywraplp.ProductCst(array_sum, 1 / mask_to_sum.shape[0])

        # Formulate the mean optimized dose as a constraint and corresponding objective
        array_mean_var = self._solver.NumVar(0, pywraplp.Solver_infinity(), name=f'{objective_name} obj')
        self._solver.Add(array_mean_var == array_mean)

        return array_mean_var

    def max_obj(self, roi: str):
        """
        Evaluates the max dose objective difference between the input dose and optimized dose
        Args:
            roi: the ROI over which the objective is calculated on
        """

        # Name for objective (used to label constraint that evaluates dose objective difference)
        objective_constraint_name = f'{roi} max dose'
        self.constraint_names.append(objective_constraint_name)

        # Get input constants
        roi_mask = self.patient.sampled_structure_masks[roi]  # Get sampled roi mask
        max_input_dose = self.patient.get_sampled_roi_dose(roi).max()  # Calculated the max roi input dose

        # Formulate the max optimized dose
        max_opt_dose_var = self._solver.NumVar(0, pywraplp.Solver_infinity(), name=f'{objective_constraint_name} obj')
        for voxel in roi_mask:
            self.var_1_le_var_2(self.d[voxel], max_opt_dose_var)

        self.obj_norm(max_input_dose, max_opt_dose_var, objective_constraint_name)

    def c_var_obj(self, roi: str, c_var_dict: dict, dose_multiple: Union[float, int, np.ndarray]):
        """
        Evaluate the C-VaR (conditional value at risk) dose object difference between input dose and optimized dose
        Args:
            roi: the ROI over which the object is calculated on
            c_var_dict: dictionary of relative penalties as a fraction (in keys) of dose_multiple, the values in the
             dictionary indicate if its a high (1) or low (-1) CVaR dose objective.
            dose_multiple: full penalty value
        """
        # Get sampled roi mask
        roi_mask = self.patient.sampled_structure_masks[roi]

        # Iterate through each c_var dose level
        for c_var_level in c_var_dict:
            dose_threshold = c_var_level * dose_multiple
            sign = c_var_dict[c_var_level]

            # Name for objective
            objective_constraint_name = f'{roi} c-var {sign} {c_var_level}'
            self.constraint_names.append(objective_constraint_name)

            # Calculate the c_var roi input dose
            c_var_input_dose = np.mean(np.maximum(0, sign * (self.patient.get_sampled_roi_dose(roi) - dose_threshold)))

            # Formulate the optimized c_var dose
            c_var_opt_voxel_dose = {}
            for roi_voxel, voxel in enumerate(roi_mask):
                c_var_opt_voxel_dose[roi_voxel] = self._solver.NumVar(0, pywraplp.Solver_infinity(),
                                                                      name=f'{roi}_{sign}_{c_var_level}_{voxel}')
                self._solver.Add(c_var_opt_voxel_dose[roi_voxel] >= sign * (self.d[voxel] - dose_threshold))

            c_var_opt_dose_var = self.get_average_as_variable(c_var_opt_voxel_dose, objective_constraint_name)
            self.obj_norm(c_var_input_dose, c_var_opt_dose_var, objective_constraint_name)

    def add_spg_constraint(self, spg_limit: int = 65) -> None:
        """
        Add the SPG constraint to limit the complexity (measure by sum of positive gradients) of the plan generated by
         optimization
        Args:
            spg_limit: the upper bound for SPG
        """

        # Initialize constant and spg constraint
        beam_angles_set = np.unique(self.patient.beamlet_coords.T[2])  # optimization beam angels
        spg = self._solver.Constraint(0, spg_limit)  # variable to store spg constraint

        # Initialize a variable to measure differences between beamlet and neighbour
        spg_beamlet_differences = {}  # dictionary for variables
        for beamlet_num in range(self.patient.number_of_beamlets):
            spg_beamlet_differences[beamlet_num] = self._solver.NumVar(0, self.cs.max_beam,
                                                                       name=f'beamlet_difference_{beamlet_num}')
        # Begin iterating though fluence maps at each angle
        spg_angle = {}  # dictionary to store spg angle variables
        for angle_idx, angle in enumerate(beam_angles_set):
            # Add spg of fluence at angle to total spg for plan
            spg_angle[angle] = self._solver.NumVar(0, pywraplp.Solver_infinity(), name=f'spg_{angle}')
            spg.SetCoefficient(spg_angle[angle], 1)
            # Get features of beamlets coming from current angle
            angle_beamlet_indices = np.argwhere(self.patient.beamlet_coords.T[2] == angle).squeeze()
            angle_beamlet_coords = self.patient.beamlet_coords[angle_beamlet_indices]  # coordinates on fluence map
            angle_beamlet_coords_rows = np.unique(
                angle_beamlet_coords.T[0])  # active rows in fluence map at current angle
            # Iterate the the rows at each angle
            for row_idx, row in enumerate(angle_beamlet_coords_rows):
                # Make constraint to evaluate spg across this row
                row_sum_of_gradients = self._solver.Constraint(0, pywraplp.Solver_infinity())
                row_sum_of_gradients.SetCoefficient(spg_angle[angle], 1)
                # Get features related to the column location of each beamlet in this row
                row_beamlet_coords_columns = np.where(
                    angle_beamlet_coords.T[0] == row)  # beamlet indices for active columns in row
                row_beamlet_indices = np.reshape(angle_beamlet_indices[row_beamlet_coords_columns], (-1,))
                column_positions_in_row = angle_beamlet_coords.T[1][
                    row_beamlet_coords_columns]  # active column in fluence map along row
                # Iterate through the columns of each row
                for col_idx, col in enumerate(column_positions_in_row):
                    beamlet_full_idx = row_beamlet_indices[col_idx]
                    neighbouring_column_position_mask = np.argwhere(column_positions_in_row == col + 1)  # sparse list
                    # Evaluate one sided differences between beamlet and neighbouring beamlet
                    if neighbouring_column_position_mask.size == 1:  # neighbouring beamlet is active
                        beamlet_neighbour_full_idx = row_beamlet_indices[neighbouring_column_position_mask[0]][0]
                        self._solver.Add(spg_beamlet_differences[beamlet_full_idx] >=
                                         self.w[beamlet_full_idx] -
                                         self.w[beamlet_neighbour_full_idx])
                    else:  # no active neighbouring beamlet, so spg is equal to beamlet intensity
                        self._solver.Add(spg_beamlet_differences[beamlet_full_idx] >=
                                         self.w[beamlet_full_idx])
                    # Add one sided differences between columns to row spg
                    row_sum_of_gradients.SetCoefficient(spg_beamlet_differences[beamlet_full_idx], -1)

    def save_fluence_and_dose(self) -> None:
        """
        Save the fluence, dose, and meta data that is generated by solving the optimization model
        """

        # Save the optimized fluence map
        w_opt = np.array([w.SolutionValue() for w in self.w.values()])  # optimized fluence map vector
        w_df = pd.DataFrame(w_opt, columns=['data'])  # convert intensities to save in consistent format
        w_df.to_csv(self.patient.get_fluence_path())  # save fluence map

        # Generate the full dose distribution (i.e., not sampled dose), and save it
        dose = self.patient.dij * w_opt  # optimized dose distribution
        sparse_dose = sparse_vector_function(dose)  # sparse dose for saving
        dose_df = pd.DataFrame(data=sparse_dose['data'].squeeze(),  # dose values
                               index=sparse_dose['indices'].squeeze(),  # voxel indices
                               columns=['data'])
        dose_df.to_csv(self.patient.get_dose_path())  # save dose distribution

        # Get objective function weights
        if self.inverse_plan:
            objective_df = self.patient.objective_df.copy()
            for c_idx, c_name in enumerate(self.constraint_names):
                objective_df.loc[c_name, 'optimized objective'] = self._solver.LookupVariable(
                    f'{c_name} obj').SolutionValue()  # get objective value
        else:
            objective_df = pd.DataFrame(index=self.constraint_names,
                                        columns=['weight', 'optimized objective', 'input objective'])
            # iterate through all constraints that evaluate objective differences
            for c_idx, c_name in enumerate(self.constraint_names):
                objective_df.loc[c_name, 'weight'] = self._solver.LookupConstraint(
                    c_name).dual_value()  # get objective weight (as dual variable)
                objective_df.loc[c_name, 'optimized objective'] = self._solver.LookupVariable(
                    f'{c_name} obj').SolutionValue()  # get objective value
                objective_df.loc[c_name, 'input objective'] = self.input_objective_values[c_idx]

        objective_df.to_csv(self.patient.get_weights_path(), header=False)  # save dose objective data

        # Get the gap (i.e., difference) between predicted and optimized dose
        if self.mean_or_max == 'mean':
            gap_value = self._objective.Value() / len(self.constraint_names)  # divided to convert total to mean
        else:
            gap_value = self._objective.Value()
        # Put misc. data into misc values pandas series
        misc_values = pd.Series(index=['gap value', 'solve time'], data=[gap_value, self.solve_time])
        misc_values.to_csv(self.patient.get_gap_path(), header=False)

    def compare_dm_to_inverse_planning(self) -> None:
        """
        Compares the dose mimicking plan to inverse planning plan
        """

        # Save the optimized fluence map
        w_opt = np.array([w.SolutionValue() for w in self.w.values()])  # optimized fluence map vector
        w_df = pd.DataFrame(w_opt, columns=['data'])  # convert intensities to save in consistent format

        # Generate the full dose distribution (i.e., not sampled dose), and save it
        dose = self.patient.dij * w_opt  # optimized dose distribution
        sparse_dose = sparse_vector_function(dose)  # sparse dose for saving
        dose_df = pd.DataFrame(data=sparse_dose['data'].squeeze(),  # dose values
                               index=sparse_dose['indices'].squeeze(),  # voxel indices
                               columns=['data'])

        # Get objective function weights
        objective_df = pd.DataFrame(index=self.constraint_names,
                                    columns=['weight', 'optimized objective', 'input objective'])
        for c_idx, c_name in enumerate(
                self.constraint_names):  # iterate through all constraints that evaluate objective differences
            # objective_df.loc[c_name, 'weight'] = self._solver.LookupConstraint(c_name).dual_value()  # get objective weight (as dual variable)
            objective_df.loc[c_name, 'optimized objective'] = self._solver.LookupVariable(
                f'{c_name} obj').SolutionValue()  # get objective value
            # objective_df.loc[c_name, 'input objective'] = self.input_objective_values[c_idx]

        # Get the gap (i.e., difference) between predicted and optimized dose
        if self.mean_or_max == 'mean':
            gap_value = self._objective.Value() / len(self.constraint_names)  # divided to convert total to mean
        else:
            gap_value = self._objective.Value()
        # Put misc. data into misc values pandas series
        misc_values = pd.Series(index=['gap value', 'solve time'], data=[gap_value, self.solve_time])

        a = objective_df
        b = self.patient.objective_df
        (a['optimized objective'] * b['Weight']).sum()
        (b['Optimized objective'] * b['Weight']).sum()
