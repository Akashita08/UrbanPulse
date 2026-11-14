import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium
import joblib
from folium.plugins import HeatMap,MarkerCluster
from streamlit_folium import st_folium
from pathlib import Path


st.set_page_config(page_title="UrbanPulse | Smart City Intelligence",page_icon="🏙️",layout="wide")
st.title("🏙️UrbanPulse")
st.subheader("Probelm Intelligence System")
st.markdown("A platform to analyse civic complaints and identify pattern ")
st.divider()



@st.cache_data
def load_data():
    df=pd.read_csv("data/processed/UrbanPulse_Phase3.csv")
    df["created_date"]=pd.to_datetime(df["created_date"],errors="coerce",utc=True).dt.tz_convert(None)
    df["resolution_hours"]=pd.to_numeric(df["resolution_hours"],errors="coerce")
    df["latitude"]=pd.to_numeric(df["latitude"],errors="coerce")
    df["longitude"]=pd.to_numeric(df["longitude"],errors="coerce")
    df=df.replace([np.inf, -np.inf],np.nan)
    df["month"]=df["created_date"].dt.month
    df["hour"]=df["created_date"].dt.hour
    df["day_of_week"]=df["created_date"].dt.dayofweek
    df["day_name"]=df["created_date"].dt.day_name()
    categorical_columns=["complaint_type","agency","borough","location_type","open_data_channel_type","status"]
    for col in categorical_columns:
        df[col]=df[col].fillna("Unknown").astype(str)
    return df

df=load_data()
st.success("Dataset Loaded")
st.caption(f"Dataset Contains{len(df):,} records")


st.sidebar.title("Dashboard Filters")
st.sidebar.markdown("Customization")
min_date=df["created_date"].min().date()
max_date=df["created_date"].max().date()
date_range=st.sidebar.date_input("Select Range",value=(min_date,max_date),min_value=min_date,max_value=max_date)
complaint_options=sorted(df["complaint_type"].unique())
selected_complaints=st.sidebar.multiselect("Complaint Categories",complaint_options,default=complaint_options)
borough_options=sorted(df["borough"].unique())
selected_boroughs=st.sidebar.multiselect("Boroughs",borough_options,default=borough_options)
status_options=sorted(df["status"].unique())
selected_status=st.sidebar.multiselect("Complaint Status",status_options,default=status_options)



if len(date_range)!=2:
    st.warning("Please select valid range")
    st.stop()

start_date,end_date=date_range
filtered_df=df[(df["created_date"]>=pd.Timestamp(start_date))&(df["created_date"]<pd.Timestamp(end_date)+pd.Timedelta(days=1))&(df["complaint_type"].isin(selected_complaints))& (df["borough"].isin(selected_boroughs)) & (df["status"].isin(selected_status))].copy()
if filtered_df.empty:
    st.warning("No records found")
    st.stop()


st.header("Dashboard Overview")
total_complaints=len(filtered_df)
closed_complaints=(filtered_df["status"].str.lower()=="closed").sum()
avg_resolution=filtered_df["resolution_hours"].mean()
median_resolution=filtered_df["resolution_hours"].median()
col1,col2,col3,col4=st.columns(4)
col1.metric("Total Complaints",f"{total_complaints:,}")
col2.metric("Closed Complaints",f"{closed_complaints:,}")
col3.metric("Average Resolution",f"{avg_resolution:.2f} hrs" if pd.notna(avg_resolution) else "N/A")
col4.metric("Median Resolution",f"{median_resolution:.2f} hrs" if pd.notna(median_resolution) else "N/A")
st.divider()


tab1,tab2,tab3,tab4,tab5=st.tabs(["Overview","Complaint Analysis","Temporal Analysis","Geographical Analysis","Prediction"])

with tab1:
    st.header("Complaint Overview")
    top_complaints=(filtered_df["complaint_type"].value_counts().head(10).reset_index())
    top_complaints.columns=["Complaint Category","Count"]
    fig1=px.bar(top_complaints,x="Count",y="Complaint Category",orientation="h",title="Top 10 Most Reported Complaint Categories",color="Count", template="plotly_white")
    fig1.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig1,use_container_width=True)
    borough_counts=(filtered_df["borough"].value_counts().reset_index())
    borough_counts.columns=["Borough","Count"]
    fig2=px.pie(borough_counts,names="Borough",values="Count",title="Complaint Distribution Across Boroughs",hole=0.4)
    st.plotly_chart(fig2,use_container_width=True)


with tab2:
    st.header("Detailed Complaint Analysis")
    complaint_summary=(filtered_df.groupby("complaint_type").size().reset_index(name="Count").sort_values("Count",ascending=False))
    fig3=px.treemap(complaint_summary,path=["complaint_type"],values="Count",title="Complaint Category Treemap")
    st.plotly_chart(fig3,use_container_width=True)
    st.subheader("Complaint Stat")
    st.dataframe(complaint_summary,use_container_width=True)


with tab3:
    st.header("Temporal Complaint Analysis")
    monthly_counts=(filtered_df.groupby("month").size().reset_index(name="Count").sort_values("month"))
    fig4=px.line(monthly_counts,x="month",y="Count",markers=True,title="Monthly Distribution Of sampled Complaints",labels={"month":"Month","Count":"Number Of Sampled Records"})
    fig4.update_layout(xaxis=dict(tickmode="linear",dtick=1))
    st.plotly_chart(fig4,use_container_width=True)
    hourly_counts=(filtered_df["hour"].value_counts().sort_index().reset_index())
    hourly_counts.columns=["Hour","Count"]
    fig5=px.bar(hourly_counts,x="Hour",y="Count",title="Complaint Distribution By hour")
    st.plotly_chart(fig5,use_container_width=True)
    weekday_order=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    weekday_counts=(filtered_df["day_name"].value_counts().reindex(weekday_order).fillna(0).reset_index())
    weekday_counts.columns=["Day","Count"]
    fig6=px.bar(weekday_counts,x="Day",y="Count",title="Complaints by Day of week")
    st.plotly_chart(fig6,use_container_width=True)
    temporal_heatmap=filtered_df.pivot_table(index="day_name",columns="hour",values="unique_key",aggfunc="count",fill_value=0)
    temporal_heatmap=temporal_heatmap.reindex(weekday_order)
    fig7=go.Figure(data=go.Heatmap(z=temporal_heatmap.values,x=temporal_heatmap.columns,y=temporal_heatmap.index,colorscale="YlOrRd"))
    fig7.update_layout(title="Complaint Pattern by day and hour",xaxis_title="Hour",yaxis_title="Day of Week")
    st.plotly_chart(fig7,use_container_width=True)



with tab4:
    st.header("Geographical Complaint Analysis")
    geo_df=filtered_df.dropna(subset=["latitude","longitude"]).copy()
    geo_df=geo_df[geo_df["latitude"].between(-90,90) & geo_df["longitude"].between(-180,180)]
    if geo_df.empty:
        st.warning("No valid records Available")
    else :
        st.write(f"Geographical Records:{len(geo_df):,}")
        map_center=[geo_df["latitude"].median(),geo_df["longitude"].median()]
        map_type=st.radio("Select Map Type",["Complaint Heatmap","Complaint Markers"],horizontal=True)
        if map_type=="Complaint Heatmap":
            complaint_map=folium.Map(location=map_center,zoom_start=10,tiles="OpenStreetMap")
            heat_data=geo_df[["latitude","longitude"]].values.tolist()
            HeatMap(heat_data,radius=10,blur=15,min_opacity=0.3).add_to(complaint_map)
        else:
            complaint_map=folium.Map(location=map_center,zoom_start=10,tiles="OpenStreetMap")
            marker_cluster=MarkerCluster().add_to(complaint_map)
            marker_sample=geo_df.sample(n=min(2000,len(geo_df)),random_state=42)
            for _, row in marker_sample.iterrows():
                popup_text=(f"Complaint:{row['complaint_type']}<br>" f"Borough:{row['borough']}<br>" f"Agency:{row['agency']}<br>" f"Status :{row['status']}")
                folium.Marker(location=[row["latitude"],row["longitude"]],popup=folium.Popup(popup_text,max_width=300),tooltip=row["complaint_type"]).add_to(marker_cluster)
        st_folium(complaint_map,width=1100,height=600,returned_objects=[])


with tab5:
    st.header("Complaint Resolution Prediction")
    model_path=Path("outputs/Phase4/UrbanPulse_best_model.pkl")
    if not model_path.exists():
        st.warning("Trained Moddel not Found")
    else :
        model= joblib.load(model_path)
        st.write("Enter Complain details to estimate the expected resolution time")
        with st.form("prediction_form"):
            col1,col2=st.columns(2)
            with col1:
                complaint_type=st.selectbox("Complaint Category",sorted(df["complaint_type"].unique()))
                agency = st.selectbox("Agency",sorted(df["agency"].unique()))
                borough=st.selectbox("Borough",sorted(df["borough"].unique()))
                location_type=st.selectbox("Location Type",sorted(df["location_type"].unique()))
                channel=st.selectbox("Reperting Channel",sorted(df["open_data_channel_type"].unique()))
            with  col2:
                month = st.selectbox("Month",list(range(1,13)))
                hour=st.slider("Complaint Creation Hour",0,23,12)
                day_options={"Monday":0,"Tuesday":1,"Wednesday":2,"Thursday":3,"Friday":4,"Saturday":5,"Sunday":6}
                day_name=st.selectbox("Day of Week",list(day_options.keys()))
                latitude=st.number_input("Latitude",value=float(df["latitude"].median()),format="%.6f")
                longitude=st.number_input("Longitude",value=float(df["longitude"].median()),format="%.6f")
            predict_button=st.form_submit_button("Predict Resolution time")

        if predict_button:
            input_data=pd.DataFrame([{"latitude":latitude,"longitude":longitude,"month":month,"hour":hour,"day_of_week":day_options[day_name],"complaint_type":complaint_type,"agency":agency,"borough":borough,"location_type":location_type,"open_data_channel_type":channel}])
            prediction = model.predict(input_data)[0]
            st.success("Prediction Done")
            st.metric("Estimated Resolution Time",f"{prediction:.2f} hours")
            st.info(f"Approximately {prediction/24:.2f} days")
            st.caption("This is model based estimate derived from calculation on data available")


    st.divider()
    st.header("Download Filtered Dataset")
    csv_data=filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(label="Download Filtered Data as CSV",data=csv_data,file_name="UrbanPulse_filtered_data.csv",mime="text/csv")


    st.divider()
    st.markdown( """
    <div style="text-align:center;">
        <h4>UrbanPulse – Smart City Problem Intelligence System</h4>
    </div>
    """,unsafe_allow_html=True)
