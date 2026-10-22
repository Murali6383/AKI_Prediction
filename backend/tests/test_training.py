from backend.app.ml.train import synthetic_dataset
from backend.app.ml.features import make_features
def test_demo_data_feature_contract():
 d=synthetic_dataset(100,11);x=make_features(d)
 assert "target_24h" not in x and "patient_id" not in x and len(x.columns)==23
