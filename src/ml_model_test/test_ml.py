"""
pytests for ml model
"""
import importlib.util
import logging
import os
import numpy as np


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
LOGGER = logging.getLogger()

SPEC = importlib.util.spec_from_file_location(
    "common", os.path.abspath(
        __file__ + "/../../") + '/common/ml_pipeline.py')
ML_PIPELINE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ML_PIPELINE)


def test_rf_overfit(dummy_feats_and_labels, rf_config):
    """
    test rf overfit
    """
    feats, labels = dummy_feats_and_labels
    pipe, used_columns = ML_PIPELINE.get_inference_pipeline(rf_config)
    pipe.fit(feats[used_columns], labels)
    pred = pipe.predict(feats)
    assert np.array_equal(
        labels, pred), 'SK pipeline should fit perfectly with small data.'
