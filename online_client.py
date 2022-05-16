# (C) Modulos AG (2019-2022). You may use and modify this code according
# to the Modulos AG Terms and Conditions. Please retain this header.

import argparse
import json
import os
import shutil
from typing import Any, Dict

from modulos_utils.metadata_handling import metadata_utils as meta_utils
from modulos_utils.solution_utils import utils as su
from modulos_utils.solution_utils import preprocessing


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
MODULES_DIR = os.path.join(BASE_DIR, "src/modules")
TMP_DIR = os.path.join(BASE_DIR, "tmp_data_dir")
METADATA_PATH = os.path.join(SRC_DIR, "metadata/input_metadata.bin")

# Simple Example, feel free to use your data here.
SAMPLE_DICT: Dict[str, Any] = {'customer_ID': '4701-AHWMW',
                               'gender': 'Male',
                               'has_dependents': 'No',
                               'has_device_protection': 'Yes',
                               'has_multiple_lines': 'No phone service',
                               'has_online_backup': 'No',
                               'has_online_security': 'No',
                               'has_paperless_billing': 'Yes',
                               'has_partner': 'Yes',
                               'has_phone_service': 'No',
                               'has_streaming_movies': 'Yes',
                               'has_streaming_ts': 'Yes',
                               'has_tech_support': 'Yes',
                               'monthly_charges': 54.55,
                               'senior_citizen': 0,
                               'tenure_months': 55,
                               'total_charges': 2978.3,
                               'type_of_contract': 'Month-to-month',
                               'type_of_internet_service': 'DSL',
                               'type_of_payment_method': 'Bank transfer (automatic)'}


def main(sample_dict: dict, tmp_dir: str, verbose: bool = False,
         keep_tmp: bool = False, ignore_sample_ids=False) -> dict:
    """Main function of the online client. Takes a sample as input and returns
    the predictions.

    Args:
        sample_dict (dict): Sample for which to predict.
        tmp_dir (str): Path to temporary data dir.
        verbose (bool = False): Whether to raise verbose error messages.
        keep_tmp (bool = False): Whether to keep the temporary data.
        ignore_sample_ids (bool = False): Whether to require unique sample_ids.

    Returns:
        dict: Dictionary containing prediction for this sample.
    """

    # Preprocess the data.
    input_node_names = list(meta_utils.MetadataDumper().load_all_nodes(
        METADATA_PATH).keys())
    prep_sample_dict = preprocessing.preprocess_dict(
        sample_dict, input_node_names, SRC_DIR, ignore_sample_ids)
    # Check whether required nodes are present. This raises an Exception if one
    # node is missing. Additional nodes are ignored.
    su.check_input_node_names(prep_sample_dict, METADATA_PATH, [tmp_dir],
                              verbose=verbose, keep_tmp=keep_tmp)

    # Create tmp data directory
    su.create_tmp_dir(tmp_dir, verbose=verbose, keep_tmp=keep_tmp)

    # Save sample dict for the sake of reproducibility.
    with open(os.path.join(tmp_dir, "sample.json"), "w") as f:
        json.dump(prep_sample_dict, f)

    # Create HDF5 version of the dataset. (TO BE REMOVED)
    prediction_input = su.prepare_sample_for_prediction(
        prep_sample_dict, BASE_DIR, [tmp_dir], verbose=verbose,
        keep_tmp=keep_tmp)
    sample_hdf5_path = os.path.join(tmp_dir, "sample.hdf5")
    su.convert_sample_to_hdf5(prediction_input, sample_hdf5_path, [tmp_dir],
                              verbose=verbose, keep_tmp=keep_tmp)

    # Run Feature Extractor.
    su.run_feature_extractor(sample_hdf5_path, tmp_dir, SRC_DIR, [tmp_dir],
                             verbose=verbose, keep_tmp=keep_tmp)

    # Run Model.
    su.run_model(SRC_DIR, tmp_dir, tmp_dir, [tmp_dir],
                 verbose=verbose, keep_tmp=keep_tmp)

    # Read Predictions (TO BE REMOVED)
    predictions_path = os.path.join(tmp_dir, "predictions.hdf5")
    predictions_raw = su.read_predictions(predictions_path, [tmp_dir],
                                          verbose=verbose, keep_tmp=keep_tmp)

    # Postprocessing the predictions.
    predictions = su.post_process(predictions_raw, SRC_DIR)

    # Remove the temporary data folder.
    if (not keep_tmp) and os.path.isdir(tmp_dir):
        shutil.rmtree(tmp_dir)

    return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run online client.")
    parser.add_argument("--verbose", action="store_true",
                        help="Use this flag to more verbose error messages.")
    parser.add_argument("--keep-tmp", action="store_true",
                        help="If this flag is used, the batch client does "
                        "not delete the temporary data.")
    parser.add_argument("--debug", action="store_true",
                        help="Use this flag to run the batch client in debug "
                        "mode. This is equivalent to activating both the "
                        "flags '--verbose' and '--keep-tmp'.")
    args = parser.parse_args()

    # Run Main Script.
    verbose_flag = args.verbose or args.debug
    keep_tmp_flag = args.keep_tmp or args.debug
    prediction = main(SAMPLE_DICT, TMP_DIR, verbose_flag, keep_tmp_flag)
    su.print_predictions(prediction)
