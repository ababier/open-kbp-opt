from itertools import product as it_product

import tqdm
from scipy import sparse

from provided_code.constants_class import ModelParameters
from provided_code.data_loader import DataLoader
from provided_code.general_functions import get_paths, get_predictions_to_optimize
from provided_code.optimizer import PlanningModel
from provided_code.resources import Patient

if __name__ == '__main__':

    # Define project constants
    cs = ModelParameters()

    # Run extra inverse planning experiments
    inverse_planning_experiments = False

    # Prepare data loader for optimization
    testing_plan_paths = get_paths(cs.reference_data_dir, ext='')  # gets the path of each patient's directory
    data_loader = DataLoader(testing_plan_paths, mode_name='optimization')

    # Select the set of predictions to plan for
    predictions_to_optimize, _ = get_predictions_to_optimize(cs)
    predictions_to_optimize = predictions_to_optimize[0:13]
    # Iterate through each set of predictions
    for prediction_path in predictions_to_optimize:
        # Define hold out set
        hold_out_plan_paths = get_paths(prediction_path, ext='')  # list of paths used for held out validation
        prediction_name = prediction_path.split('/')[-1]
        # Predict dose for the held out set
        dose_loader = DataLoader(hold_out_plan_paths, mode_name='predicted_dose')
        # Prepare files
        for idx in tqdm.tqdm(range(dose_loader.number_of_batches())):
            print('Patient {} of {}'.format(idx + 1, dose_loader.number_of_batches()))
            # Get other patient info
            pat_data = data_loader.get_batch(idx)
            # Load prediction data
            all_predicted_data = dose_loader.get_batch(patient_list=pat_data['patient_list'])
            predicted_dose = all_predicted_data[dose_loader.mode_name]
            # Build a patient object with the predicted dose
            patient = Patient(cs,
                              pat_data['patient_list'][0],  # Patient ID
                              pat_data['patient_path_list'][0],  # Path where patient data is stored
                              predicted_dose.squeeze(),  # Dose for patient
                              pat_data['structure_masks'][0],  # Structure mask
                              sparse.csr_matrix(pat_data['dij'][0]),  # Full dose influence matrix
                              pat_data['voxel_dimensions'][0],  # Dimensions of a voxel (in units of mm)
                              pat_data['beamlet_indices'][0].values,  # Beamlet indices on fluence map
                              )

            cs.set_patient(patient.identifier)
            for rel_or_abs, max_or_mean in it_product(['relative', 'absolute'], ['mean', 'max']):
                cs.set_directories_for_new_io(prediction_name, opt_name=f'{rel_or_abs}_{max_or_mean}')
                # Optimize if dose not already present
                if not cs.check_patient():
                    model = PlanningModel(patient, cs, relative_or_absolute=rel_or_abs, mean_or_max=max_or_mean)
                    model.solve(quick_test=False)
                    model.save_fluence_and_dose()
                # Generate an inverse planning plan with the weights generated above (based on theory from paper)
                cs.set_directories_for_new_io(prediction_name, opt_name=f'inverse_{rel_or_abs}_{max_or_mean}')
                if inverse_planning_experiments and not cs.check_patient():
                    patient.update_weights()
                    inverse_model = PlanningModel(patient, cs, relative_or_absolute=rel_or_abs,
                                                  mean_or_max=max_or_mean, inverse_plan=True)
                    inverse_model.solve()
                    inverse_model.save_fluence_and_dose()

