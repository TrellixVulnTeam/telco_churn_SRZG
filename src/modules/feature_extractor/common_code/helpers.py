# (C) Modulos AG (2019-2022). You may use and modify this code according
# to the Modulos AG Terms and Conditions. Please retain this header.
# THIS FILE IS DUPLICATED IN ALL TABLE PREP FEATURE EXTRACTORS. SO IF YOU
# CHANGE SOMETHING HERE, YOU HAVE TO COPY PASTE TO EVERY MEMBER OF THE TABLE
# PREP FAMILY!
"""This file contains helper functions for the table prep feature extractor
common code.
"""
import math
import numpy as np
from typing import Dict, Any, List
from abc import ABC, abstractmethod

from . import encode_categories as ec
from . import scale_numbers as sn
from modulos_utils.convert_dataset import dataset_return_object as d_obj
from modulos_utils.metadata_handling import metadata_properties as meta_prop
from modulos_utils.data_handling import data_utils

DictOfArrays = Dict[str, np.ndarray]
DictOfMetadata = Dict[str, meta_prop.AllProperties]


class TablePrepError(Exception):
    """Errors in the TablePrep Feature Extractor class.
    """
    pass


def update_transformation_results(
        tr_data: DictOfArrays, new_subnode_names: List,
        node_data: np.ndarray) -> None:
    """This is a helper function to update the transformed dictionaries with
    the result of the node transformation.

    Args:
        tr_data (DictOfArrays): Transformed data dict.
        new_subnode_names (List): Node names, that the node is
            transformed into.
        node_data (np.ndarray): Data of the transformed node.
    """
    if len(new_subnode_names) > 1:
        for new_node_index in range(len(new_subnode_names)):
            tr_data[new_subnode_names[new_node_index]] = \
                np.array([node_data[i][new_node_index]
                          for i in range(len(node_data))])
    else:
        tr_data[new_subnode_names[0]] = node_data


def check_data_types(input_object: Any) -> None:
    """Check the input data type and raise Exceptions if the types are not
    supported. We check that the input is a dictionary with strings as keys and
    only the following python native data types for the values: str, int,
    float, list. Furthermore we check that the number of samples is consistent
    across all nodes.
    """
    if not isinstance(input_object, dict):
        raise TablePrepError(
            "Input to the feature extractor must be a dictionary.")
    if input_object == {}:
        raise TablePrepError(
            "Input to the feature extractor cannot be empty.")
    n_samples = set()
    for key, value in input_object.items():
        if not isinstance(key, str):
            raise TablePrepError(
                "The keys of the feature extractor's input dictionary must "
                "be python strings.")
        if not (isinstance(value, np.ndarray)
                and (data_utils.is_modulos_numerical(value.dtype)
                     or data_utils.is_modulos_string(value.dtype))):
            raise TablePrepError(
                "The type of the values of the feature extractor's input "
                "dictionary must be numpy arrays of strings, floats or ints.")
        n_samples.add(len(value))
    if len(list(n_samples)) != 1:
        raise TablePrepError(
            "The number of samples in the feature extractor input must be the "
            "same for all nodes.")
    if list(n_samples)[0] == 0:
        raise TablePrepError(
            "The number of samples in the feature extractor input must be at "
            "least 1.")


def get_all_samples_from_generator(
        input_samples: d_obj.DatasetGenerator) -> dict:
    """Iterate over all samples of generator and create a dictionary containing
    all nodes and all samples for each node.

    Args:
        input_samples (d_obj.DatasetGenerator): A generator of
                dictionaries where the
                keys are the node names and the values are the batched node
                data as lists.

    Returns:
        dict: Dictionary containing the batch size with the key "batch_size"
            and the data (all samples) with the key "all_samples".
    """
    # THIS FUNCTION IS A COPY PASTE AND OCCURS IN ALL THE HELPERS OF THE TABLE
    # PREP FAMILY, THE T-TEST FEATURE SELECTION FAMILY AND
    # ImgIdentityCatIntEnc.

    # Note that we don't perform  checks to raise explicit exceptions in this
    # function because this whole function will be removed in BAS-603 and the
    # checks are performed after this functions call.
    batch_size = 0
    all_samples: DictOfArrays = {}
    for batch in input_samples:
        for key, values in batch.items():
            if key in all_samples:
                all_samples[key] = np.vstack((all_samples[key], values))
            else:
                all_samples[key] = values
                batch_size = len(values)
    return {"all_samples": all_samples, "batch_size": batch_size}


def get_generator_over_samples(all_samples: DictOfArrays,
                               batch_size: int) -> d_obj.DatasetGenerator:
    """Return a generator over batches of samples, given all the data.

    Args:
        all_samples (DictOfArrays): A dictionary with the node names as keys
            and the node data for all samples as values.
        batch_size (int): Batch size.

    Returns:
        Array: Generator over batches of samples.
    """
    # THIS FUNCTION IS A COPY PASTE AND OCCURS IN ALL THE HELPERS OF THE TABLE
    # PREP FAMILY, THE T-TEST FEATURE SELECTION FAMILY AND
    # ImgIdentityCatIntEnc.
    n_samples_total = len(list(all_samples.values())[0])
    n_iterations = math.ceil(n_samples_total / batch_size)
    for i in range(n_iterations):
        sample_dict = {}
        for node_name in all_samples.keys():
            sample_dict[node_name] = all_samples[node_name][
                i * batch_size: (i + 1) * batch_size]
        yield sample_dict


class ColumnTransformatorError(Exception):
    """Errors for ColumnTransformators.
    """
    # THIS FUNCTION IS A COPY PASTE AND OCCURS IN ALL THE HELPERS OF THE TABLE
    # PREP FAMILY AND THE T-TEST FEATURE SELECTION FAMILY.
    pass


class ColumnTransformator(ABC):
    # THIS CLASS IS A COPY PASTE AND OCCURS IN ALL THE HELPERS OF THE TABLE
    # PREP FAMILY AND THE T-TEST FEATURE SELECTION FAMILY.

    @abstractmethod
    def train_transformator(
            self, node_data: np.ndarray, metadata: DictOfMetadata) -> None:
        """Train transformator.

        Args:
            node_data (np.ndarray): Column data.
            metadata (DictOfMetadata): Metadata.
        """
        pass

    @abstractmethod
    def apply_trained_transformator(self, column: np.ndarray) -> np.ndarray:
        """Apply trained transformator.

        Args:
            column (np.ndarray): Data to transform.

        Returns:
            np.ndarray: Transformed data.
        """
        pass

    @abstractmethod
    def get_new_node_names(self) -> List:
        """Get node names of transformation result.

        Returns:
            List: List of strings.
        """
        pass


class ColumnTransformatorCategorical(ColumnTransformator):
    # THIS CLASS IS A COPY PASTE AND OCCURS IN ALL THE HELPERS OF THE TABLE
    # PREP FAMILY AND THE T-TEST FEATURE SELECTION FAMILY.

    def __init__(
            self,
            node_name: str,
            transformator_dictionary: Dict[str, ec.Encoder],
            transformation_type: ec.CategoryEncoderTypes) -> None:
        """Initialize object.

        Args:
            node_name (str): Node name of the column.
            transformator_dictionary (Dict[str, ec.Encoder]): Dictionary
                containing encoder to train.
            transformation_type (ec.CategoryEncoderTypes): Encoder type.
        """
        self._node_name: str = node_name
        self._transformation_type: ec.CategoryEncoderTypes = \
            transformation_type
        self._new_dimension: int = -1
        self._transformator_dictionary = transformator_dictionary

    def train_transformator(
            self,
            node_data: np.ndarray, metadata: DictOfMetadata) -> None:
        """Train transformator.

        Args:
            node_data (np.ndarray): Column data.
            metadata (DictOfMetadata): Metadata.
        """
        # Get unique values.
        unique_values = np.array(metadata[
            self._node_name].upload_unique_values.get())

        encoder = ec.CategoryEncoderPicker[self._transformation_type]()
        encoder.fit(unique_values)
        self._transformator_dictionary[self._node_name] = encoder

    def apply_trained_transformator(
            self,
            column: np.ndarray) -> np.ndarray:
        """Apply trained transformator.

        Args:
            column (np.ndarray): Data to transform.

        Returns:
            np.ndarray: Transformed data.
        """
        # Make sure the column consists of scalars, i.e. remove nested
        # dimensions.
        colunn_scalars = np.array(column).reshape(-1)
        transformed_column = \
            self._transformator_dictionary[self._node_name].transform(
                colunn_scalars)
        if len(transformed_column.shape) == 2:
            self._new_dimension = transformed_column.shape[1]
        else:
            self._new_dimension = 0
        return transformed_column

    def get_new_node_names(self) -> List:
        """Get node names of transformation result.

        Returns:
            List: List of strings.
        """
        if self._new_dimension == 0:
            return [self._node_name]
        elif self._new_dimension > 0:
            return [self._node_name + "_{}".format(i) for i in
                    range(self._new_dimension)]
        else:
            raise ColumnTransformatorError(
                "ColumnTransformator has not been applied yet!")


class ColumnTransformatorNumerical(ColumnTransformator):
    # THIS CLASS IS A COPY PASTE AND OCCURS IN ALL THE HELPERS OF THE TABLE
    # PREP FAMILY AND THE T-TEST FEATURE SELECTION FAMILY.

    def __init__(
            self,
            node_name: str,
            transformator_dictionary: Dict[str, sn.StandardScaler],
            transformation_type: sn.NumberScalingTypes) -> None:
        """Initialize object.

        Args:
            node_name (str): Node name of the column.
            transformator_dictionary: Dict[str, sn.StandardScaler]: Dictionary
                containing scaler object to train.
            transformation_type (ec.CategoryEncoderTypes): Encoder type.
        """
        self._node_name: str = node_name
        self._transformation_type: sn.NumberScalingTypes = \
            transformation_type
        self._transformator_dictionary = transformator_dictionary

    def train_transformator_online(
            self,
            node_data: np.ndarray, metadata: DictOfMetadata) -> None:
        """Train transformator in batches. (successively updating scaler
        weights).

        Args:
            node_data (np.ndarray): Column data.
            node_name (str): Node name.
            metadata (DictOfMetadata): Metadata.
        """
        if self._node_name not in self._transformator_dictionary:
            scaler = sn.NumberScalingPicker[self._transformation_type]()
            self._transformator_dictionary[self._node_name] = scaler
        self._transformator_dictionary[self._node_name].partial_fit(node_data)

    def train_transformator(
            self,
            node_data: np.ndarray, metadata: DictOfMetadata) -> None:
        """Train transformator in one run.

        Args:
            node_data (np.ndarray): Column data.
            metadata (DictOfMetadata): Metadata.
        """
        if self._node_name not in self._transformator_dictionary:
            scaler = sn.NumberScalingPicker[self._transformation_type]()
            self._transformator_dictionary[self._node_name] = scaler
        self._transformator_dictionary[self._node_name].fit(node_data)

    def apply_trained_transformator(self, column: np.ndarray) -> np.ndarray:
        """Apply trained transformator.

        Args:
            column (np.ndarray): Data to transform.

        Returns:
            np.ndarray: Transformed data.
        """
        return self._transformator_dictionary[self._node_name].transform(
            column)

    def get_new_node_names(self) -> List:
        """Get node names of transformation result.

        Returns:
            List: List of strings.
        """
        return [self._node_name]
