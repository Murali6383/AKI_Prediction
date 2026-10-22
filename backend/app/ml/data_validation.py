import pandas as pd
def inspect_dataframe(df):
    issues=[]
    if "timestamp" in df:
        parsed=pd.to_datetime(df.timestamp,errors="coerce",utc=True)
        bad=int(parsed.isna().sum())
        if bad: issues.append(f"{bad} rows have missing or invalid timestamps.")
        if "patient_id" in df:
            pairs=pd.DataFrame({"patient_id":df.patient_id,"timestamp":parsed})
            n=int(pairs.duplicated().sum())
            if n: issues.append(f"{n} duplicate patient/timestamp pairs.")
    else: issues.append("No timestamp column detected; temporal validation is unavailable.")
    return {"rows":int(len(df)),"columns":[str(c) for c in df.columns],"missingness_percent":{str(c):round(float(df[c].isna().mean()*100),2) for c in df.columns},"duplicate_rows":int(df.duplicated().sum()),"timestamp_issues":issues,"patient_id_available":"patient_id" in df}
