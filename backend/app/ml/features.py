"""Leakage-conscious feature engineering and horizon labels for AKI research."""
from __future__ import annotations
import numpy as np
import pandas as pd

FEATURE_COLUMNS = ["age", "sex", "creatinine", "baseline_creatinine", "bun", "egfr", "heart_rate", "systolic_bp", "diastolic_bp", "sodium", "potassium", "bicarbonate", "urine_output_ml_6h", "creatinine_delta", "creatinine_ratio", "comorbidity_count", "creatinine_delta_6h", "creatinine_delta_24h", "creatinine_slope_24h", "urine_output_mean_24h", "heart_rate_mean_24h", "systolic_bp_min_24h", "observation_count_24h"]
OUTCOME_COLUMNS = {"aki_label", "aki_onset_timestamp", "aki_outcome", "outcome", "label", "target_24h", "target_24_48h", "followup_until", "death_timestamp", "future_creatinine", "aki_stage", "discharge_timestamp"}
NUMERIC_BASE = ["age", "creatinine", "baseline_creatinine", "bun", "egfr", "heart_rate", "systolic_bp", "diastolic_bp", "sodium", "potassium", "bicarbonate", "urine_output_ml_6h"]

def _comorbidity_count(v):
    if isinstance(v, (list, tuple, set)): return len(v)
    if v is None or (isinstance(v, float) and np.isnan(v)): return 0
    s = str(v).strip()
    return 0 if not s or s.lower() in {"none", "nan", "[]"} else len([p for p in s.replace(",", ";").split(";") if p.strip() and p.strip().lower() != "none"])

def make_features(frame: pd.DataFrame) -> pd.DataFrame:
    x = frame.copy()
    for c in NUMERIC_BASE:
        if c not in x: x[c] = np.nan
        x[c] = pd.to_numeric(x[c], errors="coerce")
    if "sex" not in x: x["sex"] = "unknown"
    x["sex"] = x["sex"].fillna("unknown").astype(str).str.lower()
    if "comorbidities" not in x: x["comorbidities"] = [[] for _ in range(len(x))]
    x["comorbidity_count"] = x["comorbidities"].apply(_comorbidity_count)
    x["creatinine_delta"] = x.creatinine - x.baseline_creatinine
    x["creatinine_ratio"] = x.creatinine / x.baseline_creatinine.where(x.baseline_creatinine > 0)
    defaults = {"creatinine_delta_6h": np.nan, "creatinine_delta_24h": np.nan, "creatinine_slope_24h": np.nan, "urine_output_mean_24h": np.nan, "heart_rate_mean_24h": np.nan, "systolic_bp_min_24h": np.nan, "observation_count_24h": 0}
    for c, d in defaults.items():
        if c not in x: x[c] = d
        x[c] = pd.to_numeric(x[c], errors="coerce")
    return x[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)

def add_past_only_temporal_features(frame: pd.DataFrame, patient_col="patient_id", time_col="timestamp") -> pd.DataFrame:
    """Add 6h/24h trends using only same-patient rows at or before each timestamp."""
    if not {patient_col, time_col}.issubset(frame.columns): raise ValueError(f"Requires {patient_col} and {time_col}")
    x = frame.copy(); x[time_col] = pd.to_datetime(x[time_col], errors="coerce", utc=True)
    x = x.loc[x[patient_col].notna() & x[time_col].notna()].copy().sort_values([patient_col, time_col], kind="stable")
    for c in ["creatinine", "urine_output_ml_6h", "heart_rate", "systolic_bp"]:
        if c not in x: x[c] = np.nan
        x[c] = pd.to_numeric(x[c], errors="coerce")
    for c in ["creatinine_delta_6h", "creatinine_delta_24h", "creatinine_slope_24h", "urine_output_mean_24h", "heart_rate_mean_24h", "systolic_bp_min_24h", "observation_count_24h"]: x[c] = np.nan
    for _, g in x.groupby(patient_col, sort=False):
        for idx, row in g.iterrows():
            t = row[time_col]; hist = g.loc[(g[time_col] <= t) & (g[time_col] >= t - pd.Timedelta(hours=24))]
            for h in (6, 24):
                earlier = g.loc[g[time_col] <= t - pd.Timedelta(hours=h), "creatinine"].dropna()
                x.loc[idx, f"creatinine_delta_{h}h"] = float(row.creatinine - earlier.iloc[-1]) if pd.notna(row.creatinine) and len(earlier) else np.nan
            valid = hist.loc[hist.creatinine.notna(), [time_col, "creatinine"]]
            if len(valid) >= 2:
                hrs = (valid[time_col] - valid[time_col].iloc[0]).dt.total_seconds().to_numpy() / 3600
                x.loc[idx, "creatinine_slope_24h"] = float(np.polyfit(hrs, valid.creatinine.to_numpy(), 1)[0]) if np.ptp(hrs) else np.nan
            x.loc[idx, "urine_output_mean_24h"] = hist.urine_output_ml_6h.mean()
            x.loc[idx, "heart_rate_mean_24h"] = hist.heart_rate.mean()
            x.loc[idx, "systolic_bp_min_24h"] = hist.systolic_bp.min()
            x.loc[idx, "observation_count_24h"] = len(hist)
    return x

def build_next24h_labels(frame: pd.DataFrame) -> pd.DataFrame:
    """Positive: onset in (t,t+24h]; negative only with full 24h follow-up; exclude prevalent/unknown."""
    x = frame.copy(); req = {"patient_id", "timestamp", "aki_onset_timestamp", "followup_until"}
    if not req.issubset(x.columns): raise ValueError(f"Label protocol requires columns: {sorted(req)}")
    for c in ["timestamp", "aki_onset_timestamp", "followup_until", "death_timestamp"]:
        if c in x: x[c] = pd.to_datetime(x[c], errors="coerce", utc=True)
    horizon = x.timestamp + pd.Timedelta(hours=24); onset = x.aki_onset_timestamp
    positive = onset.gt(x.timestamp) & onset.le(horizon)
    negative = onset.isna() & x.followup_until.ge(horizon)
    if "death_timestamp" in x: negative &= ~(x.death_timestamp.notna() & x.death_timestamp.lt(horizon))
    prevalent = onset.notna() & onset.le(x.timestamp)
    x["target_24h"] = pd.NA; x.loc[positive, "target_24h"] = 1; x.loc[negative, "target_24h"] = 0
    x = x.loc[~prevalent & x.target_24h.notna() & x.timestamp.notna()].copy()
    return x.assign(target_24h=lambda d: d.target_24h.astype(int))

def build_next24_48h_labels(frame: pd.DataFrame) -> pd.DataFrame:
    """Separate positive horizon: onset in (t+24h,t+48h], negative needs 48h follow-up."""
    x = frame.copy(); req = {"patient_id", "timestamp", "aki_onset_timestamp", "followup_until"}
    if not req.issubset(x.columns): raise ValueError(f"Label protocol requires columns: {sorted(req)}")
    for c in ["timestamp", "aki_onset_timestamp", "followup_until"]: x[c] = pd.to_datetime(x[c], errors="coerce", utc=True)
    h24, h48 = x.timestamp + pd.Timedelta(hours=24), x.timestamp + pd.Timedelta(hours=48); onset = x.aki_onset_timestamp
    positive = onset.gt(h24) & onset.le(h48); negative = onset.isna() & x.followup_until.ge(h48)
    if "death_timestamp" in x: negative &= ~(x.death_timestamp.notna() & x.death_timestamp.lt(h48))
    prevalent = onset.notna() & onset.le(x.timestamp)
    x["target_24_48h"] = pd.NA; x.loc[positive, "target_24_48h"] = 1; x.loc[negative, "target_24_48h"] = 0
    x = x.loc[~prevalent & x.target_24_48h.notna() & x.timestamp.notna()].copy()
    return x.assign(target_24_48h=lambda d: d.target_24_48h.astype(int))
