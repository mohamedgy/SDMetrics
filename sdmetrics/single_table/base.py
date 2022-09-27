"""Base Single Table metric class."""

import copy
from operator import attrgetter

import pandas as pd

from sdmetrics.base import BaseMetric
from sdmetrics.errors import IncomputableMetricError


class SingleTableMetric(BaseMetric):
    """Base class for metrics that apply to single tables.

    Input to these family of metrics are two ``pandas.DataFrame`` instances
    and a ``dict`` representations of the corresponding ``Table`` metadata.

    Attributes:
        name (str):
            Name to use when reports about this metric are printed.
        goal (sdmetrics.goal.Goal):
            The goal of this metric.
        min_value (Union[float, tuple[float]]):
            Minimum value or values that this metric can take.
        max_value (Union[float, tuple[float]]):
            Maximum value or values that this metric can take.
    """

    name = None
    goal = None
    min_value = None
    max_value = None

    _DTYPES_TO_TYPES = {
        'i': {
            'type': 'numerical',
            'subtype': 'integer',
        },
        'f': {
            'type': 'numerical',
            'subtype': 'float',
        },
        'O': {
            'type': 'categorical',
        },
        'b': {
            'type': 'boolean',
        },
        'M': {
            'type': 'datetime',
        }
    }

    @classmethod
    def _select_fields(cls, metadata, types):
        """Select fields from metadata that match the specified types.

        Args:
            metadata (dict):
                The table metadata.
            types (str or tuple):
                The desired data types.

        Returns:
            list:
                All fields that match the specified types.

        Raises:
            IncompatibleMetricError:
                If no matching fields are found, the metric is unable to be computed.
        """
        fields = []
        if isinstance(types, str):
            types = (types, )

        for field_name, field_meta in metadata['fields'].items():
            field_type = field_meta['type']
            field_subtype = field_meta.get('subtype')
            if any(t in types for t in (field_type, (field_type, ), (field_type, field_subtype))):
                fields.append(field_name)

        if len(fields) == 0:
            raise IncomputableMetricError(f'Cannot find fields of types {types}')

        return fields

    @classmethod
    def _validate_inputs(cls, real_data, synthetic_data, metadata=None):
        """Validate the inputs and return the validated data and metadata.

        If a metadata is passed, the data is validated against it.

        If no metadata is passed, one is built based on the ``real_data`` values.

        Args:
            real_data (pandas.DataFrame):
                The real data.
            synthetic_data(pandas.DataFrame):
                The synthetic data.
            metadata (dict or Metadata or None):
                The metadata, if any.

        Returns:
            (pandas.DataFrame, pandas.DataFrame, dict):
                The validated data and metadata.
        """
        real_data = real_data.copy()
        synthetic_data = synthetic_data.copy()
        if metadata is not None:
            metadata = copy.deepcopy(metadata)

        if set(real_data.columns) != set(synthetic_data.columns):
            raise ValueError('`real_data` and `synthetic_data` must have the same columns')

        if metadata is not None:
            if not isinstance(metadata, dict):
                metadata = metadata.to_dict()

            fields = metadata['fields']
            for column in real_data.columns:
                if column not in fields:
                    raise ValueError(f'Column {column} not found in metadata')

            for field, field_meta in fields.items():
                if field not in real_data.columns:
                    raise ValueError(f'Field {field} not found in data')
                if (
                    field_meta['type'] == 'datetime' and
                    'format' in field_meta and
                    real_data[field].dtype == 'O'
                ):
                    real_data[field] = pd.to_datetime(
                        real_data[field], format=field_meta['format'])
                    synthetic_data[field] = pd.to_datetime(
                        synthetic_data[field], format=field_meta['format'])

            return real_data, synthetic_data, metadata

        dtype_kinds = real_data.dtypes.apply(attrgetter('kind'))
        return real_data, synthetic_data, {
            'fields': dtype_kinds.apply(cls._DTYPES_TO_TYPES.get).to_dict(),
        }

    @classmethod
    def compute(cls, real_data, synthetic_data, metadata=None):
        """Compute this metric.

        Real data and synthetic data must be passed as ``pandas.DataFrame`` instances
        and ``metadata`` as a ``Table`` metadata ``dict`` representation.

        If no ``metadata`` is given, one will be built from the values observed
        in the ``real_data``.

        Args:
            real_data (pandas.DataFrame):
                The values from the real dataset, passed as a pandas.DataFrame.
            synthetic_data (pandas.DataFrame):
                The values from the synthetic dataset, passed as a pandas.DataFrame.
            metadata (dict):
                Table metadata dict. If not passed, it is build based on the
                real_data fields and dtypes.

        Returns:
            Union[float, tuple[float]]:
                Metric output.
        """
        raise NotImplementedError()
