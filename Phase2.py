import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

Path("outputs/Phase2").mkdir(parents=True,exist_ok=True)
df=pd.read_csv("data/processed/UrbanPulse_cleaned.csv")
df["created_date"]=pd.to_datetime(df["created_date"],errors="coerce")
df["closed_date"]=pd.to_datetime(df["closed_date"],errors="coerce")

print("Dataset Loaded Successfully")
print("Dataset Shape:",df.shape)
print("\nFirst five records")
print(df.head())
print("\nDataset Infromation")
df.info()
print("\nDataset Description")
print(df.describe(include="all"))
print("\nmissing Values")
print(df.isnull().sum())
print("\nUnique values")
categorical_columns=["agency","agency_name","complaint_type","descriptor","status","borough","location_type","open_data_channel_type","city"]
for col in categorical_columns:
    if col in df.columns:
        print(f"{col}:{df[col].nunique()} unique values")

numerical_columns=[ "latitude","longitude","resolution_hours"]
print(df[numerical_columns].describe())
print("\n Complaint Category Frequency")
print(df["complaint_type"].value_counts().head(10))
print("\nBorough Frequency")
print(df["borough"].value_counts())
print("\nComplaint Status Frequency")
print(df["status"].value_counts())

top_complaints= df["complaint_type"].value_counts().head(10)
plt.figure(figsize=(12,6))
sns.barplot( x = top_complaints.values,y=top_complaints.index)
plt.title("Top 10 most reported complaints categories")
plt.xlabel("Number of Complaints")
plt.ylabel("Complaint Category")
plt.tight_layout()
plt.savefig("outputs/Phase2/top_complaints.png")
plt.show()

borough_counts= df["borough"].value_counts()
plt.figure(figsize=(10,6))
sns.barplot(x=borough_counts.index,y=borough_counts.values)
plt.title("Distribution of complaints across Boroughs")
plt.xlabel("Borough")
plt.ylabel("Nuumebr of Complaints")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("outputs/Phase2/borough_distribution.png")
plt.show()


df["month"]=df["created_date"].dt.month
df["day_of_week"]=df["created_date"].dt.day_name()
df["hour"] = df["created_date"].dt.hour
monthly_counts=df["month"].value_counts().sort_index()

plt.figure(figsize=(12,6))
sns.lineplot(x=monthly_counts.index,y=monthly_counts.values,marker="o")
plt.title("Monthly Distribution of Sampled Complaints")
plt.xlabel("Month")
plt.ylabel("Number oof Sampled Records")
plt.xticks(range(1,13))
plt.tight_layout()
plt.savefig("outputs/Phase2/monthly_distribution.png")
plt.show()

weekday_order=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
weekday_counts=df["day_of_week"].value_counts().reindex(weekday_order)
plt.figure(figsize=(11,6))
sns.barplot(x=weekday_counts.index,y=weekday_counts.values)
plt.title("Complaints Distribution by Day of a week")
plt.xlabel("Day")
plt.ylabel("Number of records")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("outputs/Phase2/weekday_distribution.png")
plt.show()

resolution_df=df.dropna(subset=["resolution_hours"]).copy()
print("\nResoltion Time Stat") 
print(resolution_df["resolution_hours"].describe())
print("\nAverage Resolution Time")
print(resolution_df["resolution_hours"].mean())
print("\nMedian Resolutiion Time")
print(resolution_df["resolution_hours"].median())

upper_limit=resolution_df["resolution_hours"].quantile(0.99)
resolution_plot=resolution_df[resolution_df["resolution_hours"]<=upper_limit]
plt.figure(figsize=(12,6))
sns.histplot(resolution_plot["resolution_hours"],bins=30,kde=True)
plt.title("Distribution of Complaint Resolution Time")
plt.xlabel("Resolution Time")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("outputs/Phase2/resolution_histogram.png")
plt.show()

top_categories=df["complaint_type"].value_counts().head(8).index
category_resolution=resolution_df[resolution_df["complaint_type"].isin(top_categories)]
plt.figure(figsize=(14,7))
sns.boxplot(data=category_resolution,x="complaint_type",y="resolution_hours")
plt.title("Resolution time across top complaint categories")
plt.xlabel("Complaint Category")
plt.ylabel("resolution Time")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("outputs/Phase2/resolution_boxplot.png")
plt.show()

complaint_summary=df["complaint_type"].value_counts().reset_index()
complaint_summary.columns=["Complaint Category ","Count"]
borough_summary=df["borough"].value_counts().reset_index()
borough_summary.columns=["Borough","Count"]
resolution_summary=resolution_df.groupby("complaint_type")["resolution_hours"].agg(["count","mean","median","min","max"]).sort_values(by="mean",ascending=False) 
complaint_summary.to_csv("outputs/Phase2/complaint_summary.csv",index=False)
borough_summary.to_csv("outputs/Phase2/borough_summary.csv",index=False)
resolution_summary.to_csv("outputs/Phase2/resolution_summary.csv",index=False)
print("\nTop Complaint Categories:")
print(complaint_summary.head(10))

print("\nBorough Summary:")
print(borough_summary)
print("\nResolution Summary:")
print(resolution_summary.head(10))