import typing

from federatedml.protobuf.model_migrate.converter.binning_model_converter import FeatureBinningConverter
from federatedml.protobuf.model_migrate.converter.converter_base import ProtoConverterBase
from federatedml.protobuf.model_migrate.converter.feature_selection_model_converter import \
    HeteroFeatureSelectionConverter
from federatedml.protobuf.model_migrate.converter.pearson_model_converter import HeteroPearsonConverter
from federatedml.protobuf.model_migrate.converter.tree_model_converter import HeteroSBTConverter
<<<<<<< HEAD
=======
from federatedml.protobuf.model_migrate.converter.data_transform_converter import DataTransformConverter
>>>>>>> ce6f26b3e3e52263ff41e0f32c2c88a53b00895e


def converter_factory(module_name: str) -> typing.Optional[ProtoConverterBase]:
    if module_name == 'HeteroSecureBoost':
        return HeteroSBTConverter()
    elif module_name == 'HeteroFastSecureBoost':
        return HeteroSBTConverter()
    elif module_name == 'HeteroPearson':
        return HeteroPearsonConverter()
    elif module_name == 'HeteroFeatureBinning':
        return FeatureBinningConverter()
    elif module_name == 'HeteroFeatureSelection':
        return HeteroFeatureSelectionConverter()
<<<<<<< HEAD
=======
    elif module_name == "DataTransform":
        return DataTransformConverter()
>>>>>>> ce6f26b3e3e52263ff41e0f32c2c88a53b00895e
    else:
        return None
