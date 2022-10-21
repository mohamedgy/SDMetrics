import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer

from sdmetrics.single_table.detection import (LogisticDetection,SVCDetection)

METRICS = [
    LogisticDetection,
    SVCDetection
]