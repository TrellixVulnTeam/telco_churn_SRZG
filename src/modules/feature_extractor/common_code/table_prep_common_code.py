# (C) Modulos AG (2019-2022). You may use and modify this code according
# to the Modulos AG Terms and Conditions. Please retain this header.
# THIS FILE IS DUPLICATED IN ALL TABLE PREP FEATURE EXTRACTORS. SO IF YOU
# CHANGE SOMETHING HERE, YOU HAVE TO COPY PASTE TO EVERY MEMBER OF THE TABLE
# PREP FAMILY!
"""This Feature Extractor preprocesses tables. It applies scaling to numerical
columns and encodes categorical columns, such that they are numerical.
"""
import joblib
import os
import numpy as np
from typing import Dict, List, Optional
import copy

from modulos_utils.metadata_handling import metadata_transferer as meta_trans
from modulos_utils.metadata_handling import metadata_utils as meta_utils
from modulos_utils.metadata_handling import metadata_properties as meta_prop
from modulos_utils.convert_dataset import dataset_converter as dc
from modulos_utils.convert_dataset import dataset_return_object as d_obj
from modulos_utils.module_interfaces import feature_extractor as fe_interface
from . import encode_categories as ec
from . import scale_numbers as sn
from . import helpers as fe_helpers

DictOfArrays = Dict[str, np.ndarray]
DictOfMetadata = Dict[str, meta_prop.AllProperties]


class TablePrepFeatureExtractor(fe_interface.IFeatureExtractor):
    """Feature extractor that prepares tables.
    """
    def __init__(self) -> None:
        """Initialize class by declaring all member variables and setting some
        of them to a default value.
        """
        self._num_transformation: Optional[sn.NumberScalingTypes] = None
        self._cat_transformation: Optional[ec.CategoryEncoderTypes] = None
        self._node_list: Optional[List[str]] = None
        self._metadata: Optional[dict] = None
        self._encoders: Optional[dict] = None
        self._scalers: Optional[dict] = None
        self._transformed_metadata: Optional[dict] = None
        self._weights_loaded: bool = False

    @staticmethod
    def initialize_new(
            config_choice_path: str, num_transformation: sn.NumberScalingTypes,
            cat_transformation: ec.CategoryEncoderTypes
            ) -> fe_interface.IFeatureExtractor:
        """Initialize a new (untrained) feature extractor from a config choice
        file.

        Args:
            config_choice_path (str): Path to config choice file.
            num_transformation (sn.NumberScalingTypes): Which transformation
                to apply to numerical nodes.
            cat_transformation (ec.CategoryEncoderTypes): Which transformation
                to apply to categorical nodes.

        Returns:
            fe_interface.IFeatureExtractor: An initialized object of this
                class.
        """
        result_obj = TablePrepFeatureExtractor()
        result_obj._num_transformation = num_transformation
        result_obj._cat_transformation = cat_transformation
        return result_obj

    @staticmethod
    def initialize_from_weights(weights_folder: str) \
            -> fe_interface.IFeatureExtractor:
        """Load a trained feature extractor from weights. These weights are
        generalized weights meaning anything that is saved in the
        training phase and used in the prediction phase. (Therefore the term
        weights is not used in a strict sense as in the parameters of a neural
        network that are optimized during training.) The weights contain all
        the information necessary to reconstruct the Feature Extractor object.

        Args:
            weights_folder (str): Path to folder containing weights.

        Returns:
            fe_interface.IFeatureExtractor: An initialized object of this
                class.
        """
        # Check whether weights path exits.
        if not os.path.isdir(weights_folder):
            raise fe_helpers.TablePrepError(
                f"Directory {weights_folder} does not exist.")
        result_obj = TablePrepFeatureExtractor()
        metadata = meta_utils.MetadataDumper().load_all_nodes(
            os.path.join(weights_folder, "input_metadata.bin")
        )
        result_obj._metadata = metadata
        result_obj._node_list = list(metadata.keys())
        result_obj._encoders = joblib.load(
            os.path.join(weights_folder, "encoders.bin")
            )
        result_obj._scalers = joblib.load(
            os.path.join(weights_folder, "scalers.bin"))
        result_obj._num_transformation = joblib.load(
            os.path.join(weights_folder, "num_transformation.bin")
        )
        result_obj._cat_transformation = joblib.load(
            os.path.join(weights_folder, "cat_transformation.bin")
        )
        result_obj._weights_loaded = True
        return result_obj

    def save_weights(self, weights_folder: str) -> None:
        """Save feature extractor weights.

        Args:
            weights_folder (str): Path to folder where weights should be saved.
        """
        if self._metadata is None or self._scalers is None or \
                self._encoders is None:
            raise fe_helpers.TablePrepError(
                "Generalized weights of this feature extractor cannot be "
                "saved because the feature extractor has not been trained "
                "yet.")
        if not os.path.isdir(weights_folder):
            os.makedirs(weights_folder)
        meta_utils.MetadataDumper().write_all_nodes(
            self._metadata, os.path.join(weights_folder, "input_metadata.bin")
        )
        joblib.dump(self._scalers, os.path.join(weights_folder, "scalers.bin"))
        joblib.dump(self._encoders, os.path.join(weights_folder,
                                                 "encoders.bin"))
        joblib.dump(self._num_transformation,
                    os.path.join(weights_folder, "num_transformation.bin"))
        joblib.dump(self._cat_transformation,
                    os.path.join(weights_folder, "cat_transformation.bin"))

    def fit(self, input_data: DictOfArrays, metadata: DictOfMetadata) \
            -> "TablePrepFeatureExtractor":
        """Train a feature extractor with a list of samples as input
        and output.

        Args:
            input_data (DictOfArrays): A dictionary with node names
                as keys and node data for all samples as values.
            metadata (DictOfMetadata): Dictionary containing metadata.

        Returns:
            IFeatureExtractor: The class itself.
        """
        if self._num_transformation is None:
            raise fe_helpers.TablePrepError(
                "TablePrepFeatureExtractor object was not constructed "
                "properly. The member variable '_num_transformation' was not "
                "initialized.")
        if self._cat_transformation is None:
            raise fe_helpers.TablePrepError(
                "TablePrepFeatureExtractor object was not constructed "
                "properly. The member variable '_cat_transformation' was not "
                "initialized.")
        # Save metadata and sorted node list in member variables.
        self._metadata = copy.deepcopy(metadata)
        self._node_list = list(self._metadata.keys())

        # Initialize encoders and scalers with empty dictionaries.
        self._encoders = {}
        self._scalers = {}

        # Loop over all nodes and train encoders and scalers.
        for node_name in self._node_list:
            node_metadata = self._metadata[node_name]
            node_data = input_data[node_name]
            column_transformator: fe_helpers.ColumnTransformator
            if node_metadata.is_categorical():
                column_transformator = \
                    fe_helpers.ColumnTransformatorCategorical(
                        node_name, self._encoders, self._cat_transformation)
            elif node_metadata.is_numerical() and not \
                    node_metadata.is_categorical():
                column_transformator = \
                    fe_helpers.ColumnTransformatorNumerical(
                        node_name, self._scalers, self._num_transformation)
            else:
                raise fe_helpers.TablePrepError(
                    "This Feature extractor only accepts columns that are "
                    "either numerical or categorical.")
            column_transformator.train_transformator(node_data, metadata)
        return self

    def fit_generator(
            self, input_data: d_obj.DatasetGenerator,
            metadata: DictOfMetadata) \
            -> "TablePrepFeatureExtractor":
        """Train a feature extractor with a generator over batches as input.

        Args:
            input_data (d_obj.DatasetGenerator): A generator of dictionaries
                that returns
                batches of data. The keys of the dictionaries are the node
                names.
            metadata (DictOfMetadata): Dictionary containing metadata.

        Returns:
            The class itself.
        """
        # Hack: If input is a generator, iterate over all batches to get all
        # samples. The FE should train with a generator and not load everything
        # into RAM: BAS-603
        all_samples = fe_helpers.get_all_samples_from_generator(
            input_data)["all_samples"]
        return self.fit(all_samples, metadata)

    def fit_transform(
            self, input_data: DictOfArrays, metadata: DictOfMetadata) \
            -> DictOfArrays:
        """Train and apply a feature extractor with a list of samples as input
        and output.

        Args:
            input_data (DictOfArrays): A dictionary with node names
                as keys and node data for all samples as values.
            metadata (DictOfMetadata): Dictionary containing metadata.

        Returns:
            DictOfArrays: Transformed samples, all at once in a dictionary of
                lists.
        """
        self.fit(input_data, metadata)
        return self.transform(input_data)

    def fit_transform_generator(
            self, input_data: d_obj.DatasetGenerator,
            metadata: DictOfMetadata) \
            -> d_obj.DatasetGenerator:
        """Train and apply a feature extractor with a list of samples as input
        and output.

        Args:
            input_data (d_obj.DatasetGenerator): A generator of dictionaries
                that returns
                batches of data. The keys of the dictionaries are the node
                names.
            metadata (DictOfMetadata): Dictionary containing metadata.

        Returns:
            d_obj.DatasetGenerator: Transformed samples, batched as a
            generator.
        """
        # Hack: Iterate over all batches to get all samples. (Get rid of this
        # hack in BAS-603)
        unfolded_generator = \
            fe_helpers.get_all_samples_from_generator(input_data)
        all_samples = unfolded_generator["all_samples"]
        batch_size = unfolded_generator["batch_size"]

        transformed_data = self.fit_transform(all_samples,
                                              metadata)
        transformed_data[dc.SAMPLE_ID_KEYWORD] = \
            all_samples[dc.SAMPLE_ID_KEYWORD]

        # Convert the output back to a generator with the same batch size to
        # mock the generator case (hack to be removed in BAS-603)
        return fe_helpers.get_generator_over_samples(transformed_data,
                                                     batch_size)

    def transform(self, input_data: DictOfArrays, check_input: bool = False) \
            -> DictOfArrays:
        """Apply a trained feature extractor with a list of samples as input
        and output.

        Args:
            input_data (DictOfArrays): A dictionary with node names
                as keys and node data for all samples as values.
            check_input (bool): Whether or not to check whether input data
                type/format.

        Returns:
            DictOfArrays: Transformed samples, all at once in a dictionary of
                lists.
        """
        if self._num_transformation is None \
                or self._cat_transformation is None \
                or self._node_list is None or self._metadata is None \
                or self._encoders is None or self._scalers is None:
            raise fe_helpers.TablePrepError(
                "TablePrepFeatureExtractor object has not been trained yet.")
        self._compute_transformed_metadata()
        # If input check flag is True, we perform checks and raise user
        # friendly exceptions, if the input type is wrong.
        if check_input:
            fe_helpers.check_data_types(input_data)

        # Loop over all nodes and transform that according to their data type.
        transformed_data: DictOfArrays = {}
        transformed_metadata: DictOfMetadata = {}
        for node_name in self._node_list:
            node_data = input_data[node_name]
            node_metadata = self._metadata[node_name]
            column_transformator: fe_helpers.ColumnTransformator
            if node_metadata.is_categorical():
                column_transformator = \
                    fe_helpers.ColumnTransformatorCategorical(
                        node_name, self._encoders, self._cat_transformation)
            elif node_metadata.is_numerical() and not \
                    node_metadata.is_categorical():
                column_transformator = \
                    fe_helpers.ColumnTransformatorNumerical(
                        node_name, self._scalers, self._num_transformation)
            else:
                raise fe_helpers.TablePrepError(
                    "This Feature extractor only accepts columns that are "
                    "either numerical or categorical.")

            # Apply transformator.
            new_col = column_transformator.apply_trained_transformator(
                node_data)
            new_subnode_names = column_transformator.get_new_node_names()

            fe_helpers.update_transformation_results(
                transformed_data, new_subnode_names, new_col
                )

        # Save metadata in member variable.
        if not self._transformed_metadata:
            self._transformed_metadata = transformed_metadata

        return transformed_data

    def _compute_transformed_metadata(self) -> None:
        """Compute transformed metadata and save it in member variable.

        Raises:
            fe_helpers.TablePrepError: Error if object has not been
                initialized properly.
        """
        if self._node_list is None or self._scalers is None \
                or self._encoders is None or self._metadata is None:
            raise fe_helpers.TablePrepError(
                "TablePrepFeatureExtractor object has not been trained yet.")
        if self._transformed_metadata is not None:
            return None
        self._transformed_metadata = {}
        for n in self._node_list:
            if n in self._encoders and isinstance(self._encoders[n],
                                                  ec.OneHotEncoder):
                new_node_names = [n + f"_{i}" for i in range(
                    self._encoders[n].get_n_unique_categories())]
            else:
                new_node_names = [n]
            new_node_meta = meta_trans.NodeTransferer.from_obj(
                    self._metadata[n]).get_obj()
            new_node_meta.node_type.set("num")
            for new_node_name in new_node_names:
                self._transformed_metadata[new_node_name] = new_node_meta
        return None

    def _get_transformed_generator(
            self, input_gen: d_obj.DatasetGenerator,
            check_input: bool) -> d_obj.DatasetGenerator:
        """Iterate over a generator and transform each batch.

        Args:
            input_gen (d_obj.DatasetGenerator): Input generator.
            check_input (bool): Whether or not to perform check on the input.

        Raises:
            fe_helpers.TablePrepError: Error if sample ids are missing.

        Returns:
            d_obj.DatasetGenerator: Output generator.
        """
        for batch in input_gen:
            if dc.SAMPLE_ID_KEYWORD not in batch:
                raise fe_helpers.TablePrepError(
                    "Input generator must be a generator over dictionaries, "
                    "where every dictionary must contain the sample ids with "
                    f"the key {dc.SAMPLE_ID_KEYWORD}.")
            transformed_batch = self.transform(batch, check_input=check_input)
            transformed_batch[dc.SAMPLE_ID_KEYWORD] = \
                batch[dc.SAMPLE_ID_KEYWORD]
            yield transformed_batch

    def transform_generator(
            self, input_data: d_obj.DatasetGenerator,
            check_input: bool = False) -> d_obj.DatasetGenerator:
        """Apply a trained feature extractor with a list of samples as input
        and output.

        Args:
            input_data (d_obj.DatasetGenerator): A generator of dictionaries
                where the keys
                are the node names and the values are the batched node data as
                lists.
            metadata (DictOfMetadata): Dictionary containing metadata.

        Returns:
            d_obj.DatasetGenerator: Transformed samples, batched as a
            generator.
        """
        self._compute_transformed_metadata()
        return self._get_transformed_generator(
            input_data, check_input=check_input)

    def get_transformed_metadata(self) -> DictOfMetadata:
        """Get transformed metadata after training the Feature Extractor.

        Returns:
            DictOfMetadata: Transformed metadata.

        Raises:
            fe_helpers.TablePrepError: Error, if transformed metadata has
                not been computed yet.
        """
        if not self._transformed_metadata:
            raise fe_helpers.TablePrepError(
                "Metadata of transformed data can not be retrieved because "
                "this feature extractor has not been run yet."
                )
        return self._transformed_metadata
