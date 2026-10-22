import pandas as pd
from backend.app.ml.features import make_features,build_next24h_labels
def test_outcomes_not_features():
 d=make_features(pd.DataFrame([{"age":50,"creatinine":1.2,"baseline_creatinine":1.0,"aki_label":1}]))
 assert "aki_label" not in d.columns and abs(d.iloc[0].creatinine_delta-.2)<1e-6
def test_label_protocol_excludes_inadequate_followup():
 d=pd.DataFrame([{"patient_id":"a","timestamp":"2026-01-01T00:00Z","aki_onset_timestamp":"2026-01-01T12:00Z","followup_until":"2026-01-03T00:00Z"},{"patient_id":"b","timestamp":"2026-01-01T00:00Z","aki_onset_timestamp":None,"followup_until":"2026-01-03T00:00Z"},{"patient_id":"c","timestamp":"2026-01-01T00:00Z","aki_onset_timestamp":None,"followup_until":"2026-01-01T12:00Z"}])
 assert build_next24h_labels(d).set_index("patient_id").target_24h.to_dict()=={"a":1,"b":0}
