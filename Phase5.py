import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import folium
from folium.plugins import HeatMap,MarkerCluster
from pathlib import Path


Path("outputs/Phase5").mkdir(parents=True,exist_ok=True)
df=pd.read_csv("data/processed/UrbanPulse_Phase3.csv")
df["created_date"]=pd.to_datetime(df["created_date"],errors="coerce")
df["month"]=df["created_date"].dt.month
df["hour"]=df["created_date"].dt.hour
df["day_of_week"]=df["created_date"].dt.day_name()
df["resolution_hours"]=pd.to_numeric(df["resolution_hours"],errors="coerce")
df=df.replace([np.inf, -np.inf],np.nan)
print("Datset Shape:",df.shape)


top_complaints=df["complaint_type"].value_counts().head(10).reset_index()
top_complaints.columns=["Complaint Category","Count"]
fig=px.bar(top_complaints,x="Count",y="Complaint Category",orientation="h",title="Top 10 Most Reported Complaint Categories",color="Count", hover_data=["Count"])
fig.update_layout(yaxis={"categoryorder": "total ascending"},template ="plotly_white")
fig.write_html("outputs/Phase5/top_complaints.html")
fig.show()


borough_counts=df["borough"].value_counts().reset_index()
borough_counts.columns=["Borough","Count"]
fig=px.pie(borough_counts,names="Borough",values="Count",title="Complaint Distribution Across Boroughs",hole=0.4)
fig.update_traces(textposition="inside",textinfo="percent+label")
fig.update_layout(template="plotly_white")
fig.write_html("outputs/Phase5/borough_distribution.html")
fig.show()


monthly_counts=(df.groupby("month").size().reset_index(name="Count").sort_values("month"))
fig=px.line(monthly_counts,x="month",y="Count",markers=True,title="Monthly Distribution Of sampled Complaints",labels={"month":"Month","Count":"Number Of Sampled Records"})
fig.update_layout(xaxis=dict(tickmode="linear",dtick=1),template="plotly_white")
fig.write_html("outputs/Phase5/monthly_trnd.html")
fig.show()


hourly_counts=(df["hour"].value_counts().sort_index().reset_index())
hourly_counts.columns=["Hour","Count"]
fig=px.bar(hourly_counts,x="Hour",y="Count",title="Complaint Distribution By hour",color="Count")
fig.update_layout(xaxis=dict(tickmode="linear",dtick=1),template="plotly_white")
fig.write_html("outputs/Phase5/hourly_distribution.html")
fig.show()


weekday_order=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
temporal_df=df.dropna(subset=["day_of_week","hour"]).copy()
temporal_heatmap=temporal_df.pivot_table(index="day_of_week",columns="hour",values="unique_key",aggfunc="count",fill_value=0)
temporal_heatmap=temporal_heatmap.reindex(weekday_order)
plt.figure(figsize=(15,7))
sns.heatmap(temporal_heatmap,cmap="YlOrRd",linewidths=0.3)
plt.title("Complaint Patterns by Day and Hour")
plt.xlabel("Hour Of day")
plt.ylabel("Day of Week")
plt.tight_layout()
plt.savefig("outputs/Phase5/temporal_heatmap.png")
plt.show()

resolution_df=df.dropna(subset=["resolution_hours","complaint_type"]).copy()
resolution_df=resolution_df[resolution_df["resolution_hours"]>=0]
category_resolution=(resolution_df.groupby("complaint_type")["resolution_hours"].median().sort_values(ascending=False).head(10).reset_index())
category_resolution.columns=["Complaint Category","Median Resolution Hours"]
fig=px.bar(category_resolution,x="Median Resolution Hours",y="Complaint Category",orientation="h",color="Median Resolution Hours",title="Top 10 Categories by Median Resolution Hours")
fig.update_layout(yaxis={"categoryorder":"total ascending"},template="plotly_white")
fig.write_html("outputs/Phase5/category_resolution.html")
fig.show()


geo_df=df.dropna(subset=["latitude","longitude"]).copy()
geo_df["latitude"]=pd.to_numeric(geo_df["latitude"],errors="coerce")
geo_df["longitude"]=pd.to_numeric(geo_df["longitude"],errors="coerce")
geo_df=geo_df.dropna(subset=["latitude","longitude"])
geo_df=geo_df[geo_df["latitude"].between(-90,90) & geo_df["longitude"].between(-180,180)]
print("\nGeographical Records:",len(geo_df))


map_center=[geo_df["latitude"].median(),geo_df["longitude"].median()]
complaint_map=folium.Map(location=map_center,zoom_start=10,tiles="OpenStreetMap")
heat_data=geo_df[["latitude","longitude"]].values.tolist()
HeatMap(heat_data,radius=10,blur=15,min_opacity=0.3).add_to(complaint_map)
complaint_map.save("outputs/Phase5/complaint_heatmap.html")


marker_map=folium.Map(location=map_center,zoom_start=10,tiles="OpenStreetMap")
marker_cluster=MarkerCluster().add_to(marker_map)
marker_sample=geo_df.sample(n=min(2000,len(geo_df)),random_state=42)
for _, row in marker_sample.iterrows():
    popup_text=(f"Complaint:{row['complaint_type']}<br>" f"Borough:{row['borough']}<br>" f"Agency:{row['agency']}<br>" f"Status :{row['status']}")
    folium.Marker(location=[row["latitude"],row["longitude"]],popup=folium.Popup(popup_text,max_width=300),tooltip=row["complaint_type"]).add_to(marker_cluster)
marker_map.save("outputs/Phase5/complaint_markers.html")
