"""
pytests for ml model
"""
import numpy as np
import os
import logging
import importlib.util

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()
logger.info(os.listdir())

spec = importlib.util.spec_from_file_location("common", os.path.abspath(__file__ + "/../../") + '/common/ml_pipeline.py')
ml_pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ml_pipeline)

# Check if model can overfit perfectly
def test_rf_overfit(dummy_feats_and_labels,rf_config):
    feats, labels = dummy_feats_and_labels
    pipe, used_columns = ml_pipeline.get_inference_pipeline(rf_config)
    pipe.fit(feats[used_columns], labels)
    pred = pipe.predict(feats)
    assert np.array_equal(labels, pred), 'SK pipeline should fit perfectly with small data.'
