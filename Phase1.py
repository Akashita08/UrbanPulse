import pandas as pd
import numpy as np
import requests
from pathlib import Path
Path("data/raw").mkdir(parents=True, exist_ok=True)
Path("data/processed").mkdir(parents=True, exist_ok=True)
url = "https://data.cityofnewyork.us/resource/erm2-nwe9.json"
columns = [ "unique_key",
    "created_date",
    "closed_date",
    "agency",
    "agency_name",
    "complaint_type",
    "descriptor",
    "status",
    "borough",
    "incident_zip",
    "latitude",
    "longitude",
    "location_type",
    "open_data_channel_type",
    "city"
]
data_list=[]
for month in range(1,13):
    start = f"2024-{month:02d}-01T00:00:00"
    if month==12:
        end = "2025-01-01T00:00:00"
    else :
        end = f"2024-{month+1:02d}-01T00:00:00"

    params={"$select": ",".join(columns),"$where": f"created_date>='{start}' AND created_date<'{end}'","$order": "created_date ASC","$limit":3000}
    response= requests.get(url,params=params,timeout=60)
    response.raise_for_status()
    monthly_data=response.json()
    data_list.extend(monthly_data)
    print(f"Month{month}:{len(monthly_data)} records downloaded")
df= pd.DataFrame(data_list)
df.to_csv("data/raw/UrbanPulse_raw.csv",index=False)
print("\n Dataset Downloaded Sucessfully")

print("\nDataset Information")
print(df.info())
print("\nDataset Description")
print(df.describe(include="all"))
print("\nDataset shape")
print(df.shape)
print("\nColumn names")
print(df.columns)
print("\nMissing values")
print(df.isnull().sum())

cleaned_df= df.copy()
print("Duplicates",cleaned_df.duplicated().sum())
cleaned_df.drop_duplicates(subset="unique_key",inplace=True)
print(cleaned_df.isnull().sum())
cleaned_df.dropna(subset=["unique_key","created_date","complaint_type"],inplace=True)
cleaned_df["descriptor"]=cleaned_df["descriptor"].fillna("Unknown")
cleaned_df["borough"] = cleaned_df["borough"].fillna("Unknown")
cleaned_df["agency"] = cleaned_df["agency"].fillna("Unknown")
cleaned_df["status"]=cleaned_df["status"].fillna("Unknown")

cleaned_df["closed_date"] = pd.to_datetime(cleaned_df["closed_date"],errors="coerce",utc=True)
cleaned_df["created_date"]=pd.to_datetime(cleaned_df["created_date"],errors="coerce",utc=True)
cleaned_df["latitude"] = pd.to_numeric(cleaned_df["latitude"],errors="coerce")
cleaned_df["longitude"] = pd.to_numeric(cleaned_df["longitude"],errors="coerce")
categorical_columns=["agency","agency_name","complaint_type","descriptor","status","borough","location_type","open_data_channel_type","city"]
for col in categorical_columns:
    cleaned_df[col]= cleaned_df[col].astype("string").str.strip()

cleaned_df.loc[~cleaned_df["latitude"].between(-90,90),"latitude"]=np.nan
cleaned_df.loc[~cleaned_df["longitude"].between(-180,180),"longitude"]=np.nan

cleaned_df["resolution_hours"]=(cleaned_df["closed_date"]-cleaned_df["created_date"]).dt.total_seconds()/3600

cleaned_df.loc[cleaned_df["resolution_hours"]<0,"resolution_hours"]=np.nan

cleaned_df.to_csv("data/processed/UrbanPulse_cleaned.csv",index=False)
print("Cleaned Shape:",cleaned_df.shape)
print("\nRemaining Missing Values")
print(cleaned_df.isnull().sum())
print("\nFinal dataset info:")
print(cleaned_df.info())