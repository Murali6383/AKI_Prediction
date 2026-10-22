# AKI research pipeline

## Modules

- `label_generation.py`: timestamp-based 24-hour and separate 24–48-hour labels.
- `features.py`: fixed feature schema and past-only rolling features.
- `train.py`: patient-level train/validation/test split; Logistic Regression, LightGBM, XGBoost, CatBoost when installed; validation AUPRC/AUROC model selection; validation-only F1 threshold tuning; one final test evaluation.
- `train_gru.py`: optional GRU benchmark for genuinely timestamped repeated observations. It is an experiment, not the default API model.
- `evaluation.py`: discrimination, threshold metrics, Brier score, ROC/PR curves and calibration bins.

## Label CSV protocol

Input columns must include `patient_id`, `timestamp`, `aki_onset_timestamp`, and `followup_until`. `death_timestamp` is optional. `aki_onset_timestamp` must be derived using a documented, clinically reviewed AKI definition (e.g. KDIGO criteria); do not infer it from the future prediction target itself. Negative examples require complete follow-up through the end of the relevant horizon. Rows with AKI already present at prediction time are excluded. Unknown/censored rows are not silently treated as negative.

```powershell
python -m backend.app.ml.label_generation --csv data\observations_with_outcomes.csv --out data\aki_labeled.csv --horizon 24h
python -m backend.app.ml.train --csv data\aki_labeled.csv
```

For the separate 24–48h horizon, generate that target and train with `--target target_24_48h` after making sure the CSV contains it. For both labels:

```powershell
python -m backend.app.ml.label_generation --csv data\observations_with_outcomes.csv --out data\aki_labels_both.csv --horizon both
python -m backend.app.ml.train --csv data\aki_labels_both.csv --target target_24_48h
```

GRU (requires an appropriate PyTorch installation):

```powershell
python -m backend.app.ml.train_gru --csv data\aki_labeled.csv --target target_24h
```

The training code uses patient-level splits. Temporal features only use each patient's observations at or before that timestamp. Do not put post-outcome variables, discharge outcomes, AKI stages measured after the prediction time, or other future-derived columns into predictors. Validate data provenance, patient identifiers, duplicate admissions, site/time generalizability, missingness, and target derivation before reporting research metrics.

`--demo` creates synthetic rows for software testing only. Demo metrics must never be reported as clinical performance. This project is a final-year research prototype and is not intended for diagnosis, treatment, or clinical deployment.
