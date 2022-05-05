"""
pytests for ml model
"""
import numpy as np
import pandas as pd
from train_ml_model.run import get_inference_pipeline

# Check if model can overfit perfectly
def test_rf_overfit(dummy_feats_and_labels,rf_config):
    feats, labels = dummy_feats_and_labels
    pipe, used_columns = get_inference_pipeline(rf_config)
    pipe.fit(pd.DataFrame(feats, columns=used_columns), labels)
    pred = pipe.predict(feats)
    assert np.array_equal(labels, pred), 'SK pipeline should fit perfectly with small data.'
