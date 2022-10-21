from sdmetrics.single_table import LogisticDetection, SVCDetection
import unittest
from unittest.mock import patch

class TestPrimaryKeyDetectionMetrics(self):
    """Test the ``compute_breakdown`` method with a multi-line value.

    Expect that the match is made correctly."""
    # Setup
    real_data = pd.DataFrame({
            'ID_1': [1, 2, 1, 3, 4],
            'col2': ['a', 'b', 'c', 'd', 'b'],
            'ID_2' : ['aa', 'bb', 'cc', 'dd', 'bb']
        })
    synthetic_data = pd.DataFrame({
        'ID_1': [1, 3, 4, 2, 2],
        'col2': ['a', 'b', 'c', 'd', 'e'],
        'ID_2': ['aa', 'bb', 'cc', 'dd', 'ee']
    })
    metadata = {
        'fields': {
            'ID_1': {'type': 'numerical', 'subtype': 'int'},
            'col2': {'type': 'categorical'},
            'ID_2': {'type': 'categorical'}
        },
        'primary_key' : {['ID_1','ID_2']}
    }
    metric = LogisticDetection()

    # Run
    score = metric.compute(real_data, synthetic_data, metadata)

    # Assert
    assert score == 0.5

