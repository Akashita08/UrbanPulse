import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

Path("outputs/Phase3").mkdir(parents=True,exist_ok=True)
df=pd.read_csv("data/processed/UrbanPulse_cleaned.csv")
df["created_date"]=pd.to_datetime(df["created_date"],errors="coerce")
df["closed_date"]=pd.to_datetime(df["closed_date"],errors="coerce")
df["resolution_hours"]=(df["closed_date"]-df["created_date"]).dt.total_seconds()/3600
df.loc[df["resolution_hours"]<0,"resolution_hours"]= np.nan
print("Dataset Loaded")
print("Dataset Shape:",df.shape)

df["month"]= df["created_date"].dt.month
df["hour"]=df["created_date"].dt.hour
df["day_of_week"]=df["created_date"].dt.dayofweek
print("\nFeature Extraction")
print(df[["created_date","month","hour","day_of_week"]].head())


resolution_data=df["resolution_hours"].dropna()
print("\nDescriptive Stat")
print("Mean",resolution_data.mean())
print("Median",resolution_data.median())
print("Mode",resolution_data.mode().iloc[0])
print("Variance",resolution_data.var())
print("Standard Deviation",resolution_data.std())
print("Minimum",resolution_data.min())
print("Maximum",resolution_data.max())
print("Range:",resolution_data.max()-resolution_data.min())


Q1=resolution_data.quantile(0.25)
Q2=resolution_data.quantile(0.50)
Q3=resolution_data.quantile(0.75)
IQR=Q3-Q1
print("\nQuartiles and IQR")
print("Q1",Q1)
print("Q2",Q2)
print("Q3",Q3)
print("IQR",IQR)
percentiles= resolution_data.quantile([0.10,0.25,0.50,0.75,0.90,0.95])
print("\nPercntile Analysis")
print(percentiles)

numerical_columns=["resolution_hours","latitude","longitude","hour","month","day_of_week"]
correlation_df=df[numerical_columns].copy()
correlation_matrix=correlation_df.corr(method="pearson")

print("\nCorrelation Matrix")
print(correlation_matrix)
plt.figure(figsize=(11,8))
sns.heatmap(correlation_matrix,annot=True,cmap="coolwarm",fmt=".2f",linewidths=0.5)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("outputs/Phase3/correlation_heatmap.png")
plt.show()

covariance_matrix= correlation_df.cov()
print("\nCovariance Matrix")
print(covariance_matrix)
plt.figure(figsize=(11,8))
sns.heatmap(covariance_matrix,annot=True,cmap="YlGnBu",fmt=".2f",linewidths=0.5)
plt.title("Covariance Matrix")
plt.tight_layout()
plt.savefig("outputs/Phase3/covariance_heatmap.png")
plt.show()

scatter_df=df.dropna(subset=["resolution_hours","hour"]).copy()
scatter_df=scatter_df[scatter_df["resolution_hours"]<=scatter_df["resolution_hours"].quantile(0.99)]
plt.figure(figsize=(11,6))
sns.scatterplot(data=scatter_df,x="hour",y="resolution_hours",alpha=0.5)
plt.title("Relationship between Complaint Time and Resolution Time")
plt.xlabel("Complaint creation hour")
plt.ylabel("Resolution Time")
plt.tight_layout()
plt.savefig("outputs/Phase3/resolution_scatter.png")
plt.show()

scaler = MinMaxScaler()
df["normalized_resolution"]=np.nan
valid_resolution=df["resolution_hours"].notna()
df.loc[valid_resolution,"normalized_resolution"]=(scaler.fit_transform(df.loc[valid_resolution,["resolution_hours"]]).flatten())
print("\nNormalization Done")
print(df[["resolution_hours","normalized_resolution"]].head(10))
print("\nNormalized Data Stat")
print(df["normalized_resolution"].describe())

normalization_sample=df.dropna(subset=["resolution_hours","normalized_resolution"]).copy()
normalization_sample=normalization_sample.sample(n=min(2000,len(normalization_sample)),random_state=42)
plt.figure(figsize=(12,6))
plt.scatter( normalization_sample["resolution_hours"],normalization_sample["normalized_resolution"],alpha=0.5)
plt.title("Original vs Normalized Resolution time")
plt.xlabel("Original Resolution Time")
plt.ylabel("Normalized Resolution Time")
plt.tight_layout()
plt.savefig("outputs/Phase3/normalization_plt.png")
plt.show()


bins=[0,24,72,168,np.inf]
labels=["Within 24 Hours","1-3 Days","3-7 Days","More than 7 days"]
df["resolution_category"]=pd.cut(df["resolution_hours"],bins=bins,labels=labels,right=True)
print("\nBinning Done")
print(df[["resolution_hours","resolution_category"]].head(15))

bin_counts=df["resolution_category"].value_counts(sort=False)
print("\nResolution Category Distribution")
print(bin_counts)

plt.figure(figsize=(11,6))
sns.barplot(x=bin_counts.index,y=bin_counts.values)
plt.title("Distribution of Complains by resolution Category")
plt.xlabel("Resolution Category")
plt.ylabel("Number of Complaints")
plt.tight_layout()
plt.savefig("outputs/Phase3/binning_distribution.png")
plt.show()


category_statistics=df.groupby("complaint_type",observed=True)["resolution_hours"].agg(["count","mean","median","std"]).sort_values(by="mean",ascending=False)
print("\nCategory wise resolution stat")
print(category_statistics.head(10))


df.to_csv("data/processed/UrbanPulse_Phase3.csv",index=False)
correlation_matrix.to_csv("outputs/Phase3/correlation_matrix.csv")
covariance_matrix.to_csv("outputs/Phase3/covariance_matrix.csv")
category_statistics.to_csv("outputs/Phase3/category_statistics.csv")
bin_counts.to_csv("outputs/Phase3/resolution_categories.csv")
print("\nFinal dataset Shape",df.shape)