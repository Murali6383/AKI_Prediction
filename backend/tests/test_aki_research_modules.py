import pandas as pd
from backend.app.ml.features import build_next24h_labels, build_next24_48h_labels, add_past_only_temporal_features, make_features

def test_next24_labels_positive_negative_unknown_and_prevalent():
    base=pd.Timestamp("2026-01-01T00:00:00Z")
    df=pd.DataFrame({
        "patient_id":["p1","p2","p3","p4"],
        "timestamp":[base]*4,
        "aki_onset_timestamp":[base+pd.Timedelta(hours=12),pd.NaT,pd.NaT,base-pd.Timedelta(hours=1)],
        "followup_until":[base+pd.Timedelta(hours=24),base+pd.Timedelta(hours=30),base+pd.Timedelta(hours=10),base+pd.Timedelta(hours=30)]
    })
    out=build_next24h_labels(df).set_index("patient_id")
    assert out.loc["p1","target_24h"] == 1
    assert out.loc["p2","target_24h"] == 0
    assert "p3" not in out.index
    assert "p4" not in out.index

def test_24_to_48_hour_target_is_separate_horizon():
    t=pd.Timestamp("2026-01-01T00:00:00Z")
    df=pd.DataFrame({"patient_id":["a","b"],"timestamp":[t,t],"aki_onset_timestamp":[t+pd.Timedelta(hours=30),pd.NaT],"followup_until":[t+pd.Timedelta(hours=50),t+pd.Timedelta(hours=49)]})
    out=build_next24_48h_labels(df).set_index("patient_id")
    assert out.loc["a","target_24_48h"] == 1
    assert out.loc["b","target_24_48h"] == 0

def test_temporal_features_never_use_future_rows():
    t=pd.Timestamp("2026-01-01T00:00:00Z")
    df=pd.DataFrame({"patient_id":["p","p"],"timestamp":[t,t+pd.Timedelta(hours=12)],"creatinine":[1.0,9.0],"urine_output_ml_6h":[500,100],"heart_rate":[70,150],"systolic_bp":[120,80]})
    out=add_past_only_temporal_features(df)
    assert out.iloc[0]["creatinine_slope_24h"] != out.iloc[0]["creatinine_slope_24h"]
    assert out.iloc[0]["heart_rate_mean_24h"] == 70
    assert out.iloc[1]["heart_rate_mean_24h"] == 110

def test_features_have_fixed_columns_and_no_outcomes():
    x=make_features(pd.DataFrame([{"age":50,"sex":"female","creatinine":1.4,"baseline_creatinine":1.0,"target_24h":1,"aki_onset_timestamp":"2026-01-01"}]))
    assert "target_24h" not in x.columns
    assert abs(x.loc[0,"creatinine_delta"] - 0.4) < 1e-8
