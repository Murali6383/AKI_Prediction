"""Create time-horizon labels from clinically reviewed AKI onset/follow-up timestamps."""
import argparse
from pathlib import Path
import pandas as pd
from .features import build_next24h_labels, build_next24_48h_labels

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv",required=True,help="CSV with patient_id,timestamp,aki_onset_timestamp,followup_until; optional death_timestamp")
    p.add_argument("--out",default="data/aki_labeled.csv")
    p.add_argument("--horizon",choices=["24h","24_48h","both"],default="both")
    a=p.parse_args(); df=pd.read_csv(a.csv)
    out=build_next24h_labels(df) if a.horizon=="24h" else build_next24_48h_labels(df) if a.horizon=="24_48h" else None
    if a.horizon=="both":
        a24=build_next24h_labels(df); a48=build_next24_48h_labels(df)
        out=a24.merge(a48[["patient_id","timestamp","target_24_48h"]],on=["patient_id","timestamp"],how="outer",suffixes=("","_other"))
    path=Path(a.out); path.parent.mkdir(parents=True,exist_ok=True); out.to_csv(path,index=False)
    print(f"Wrote {len(out):,} labeled rows to {path}")
    for c in ["target_24h","target_24_48h"]:
        if c in out: print(f"{c}: {out[c].value_counts(dropna=False).to_dict()}")
    print("Review label validity, KDIGO onset derivation, censoring, and patient IDs before training.")
if __name__=="__main__": main()
