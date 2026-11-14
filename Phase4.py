import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error,r2_score,mean_squared_error
import joblib

Path("outputs/Phase4").mkdir(parents=True,exist_ok=True)
df= pd.read_csv("data/processed/UrbanPulse_Phase3.csv")
df["created_date"] = pd.to_datetime(df["created_date"],errors="coerce")
print("Dataset Shape",df.shape)


df= df.replace([np.inf, -np.inf],np.nan)
df=df.dropna(subset=["created_date","resolution_hours"])
df=df[df["resolution_hours"]>=0].copy()
df=df.sort_values("created_date").reset_index(drop=True)
print("Usable Records",len(df))


numerical_features=["latitude","longitude","month","hour","day_of_week"]
categorical_features=["complaint_type","agency","borough","location_type","open_data_channel_type"]
features= numerical_features+categorical_features
X= df[features]
y = df["resolution_hours"]
df[categorical_features]= df[categorical_features].fillna("Unknown").astype(str)
print("\nSelected Categories")
print(features)
print("\nTarget variable : Resolution Hours ")

split_index=int(len(df)*0.80)
X_train=X.iloc[:split_index]
X_test= X.iloc[split_index:]
y_train=y.iloc[:split_index]
y_test=y.iloc[split_index:]
print("\nTrainning and Testing Split")
print("\nTraining Records",len(X_train))
print("\nTesting Records",len(X_test))

numerical_transformer= Pipeline(steps=[("imputer",SimpleImputer(strategy="median"))])
categorical_transformer=Pipeline(steps=[("imputer",SimpleImputer(strategy="most_frequent")),("encoder",OneHotEncoder(handle_unknown="ignore"))])
preprocessor=ColumnTransformer(transformers=[("num",numerical_transformer,numerical_features),("cat",categorical_transformer,categorical_features)])


baseline_model=Pipeline(steps=[("preprocessor",preprocessor),("model",DummyRegressor(strategy="median"))])
baseline_model.fit(X_train,y_train)
baseline_predictions=baseline_model.predict(X_test)


linear_model=Pipeline(steps=[("preprocessor",preprocessor),("model",LinearRegression())])
linear_model.fit(X_train,y_train)
linear_predictions=linear_model.predict(X_test)


random_forest_model=Pipeline(steps=[("preprocessor",preprocessor),("model",RandomForestRegressor(n_estimators=100,max_depth=15,min_samples_leaf=3,random_state=42,n_jobs=-1))])
random_forest_model.fit(X_train,y_train)
rf_predictions=random_forest_model.predict(X_test)


model_predictions={"Baseline":baseline_predictions,"Linear Regression":linear_predictions,"Random Forest":rf_predictions}
results=[]
for model_name,predictions in model_predictions.items():
    mae=mean_absolute_error(y_test,predictions)
    rmse= np.sqrt(mean_squared_error(y_test,predictions))
    r2=r2_score(y_test,predictions)
    results.append({"Model":model_name,"MAE":mae,"RMSE":rmse,"R2 Score":r2})
results_df=pd.DataFrame(results)
print("\nModel Performance  Comparison")
print(results_df.to_string(index=False))
results_df.to_csv("outputs/Phase4/model_comparison.csv",index=False)


best_model_name=results_df.loc[results_df["MAE"].idxmin(),"Model"]
if best_model_name=="Baseline":
    best_model= baseline_model
    best_predictions=baseline_predictions
elif best_model_name=="Linear Regression":
    best_model=linear_model
    best_predictions=linear_predictions
else:
    best_model= random_forest_model
    best_predictions=rf_predictions

print("\nBestPerforming Model:",best_model_name)


prediction_df=pd.DataFrame({"Actual Resolution Hours":y_test.values,"Predicted Resolution Hours":best_predictions})
prediction_df["Prediction Error"]=(prediction_df["Actual Resolution Hours"]-prediction_df["Predicted Resolution Hours"])
prediction_df.to_csv("outputs/Phase4/prediction.csv",index=False)
print(prediction_df.head(10))


plot_df=prediction_df.copy()
upper_limit=plot_df["Actual Resolution Hours"].quantile(0.99)
plot_df=plot_df[plot_df["Actual Resolution Hours"]<=upper_limit]
plt.figure(figsize=(10,6))
sns.scatterplot(data=plot_df,x="Actual Resolution Hours",y="Predicted Resolution Hours",alpha=0.5)
plt.title("Actual vs Predicted Resolution Time")
plt.xlabel("Actual Resolution Hours")
plt.ylabel("Predicted Resolution Hours")
plt.tight_layout()
plt.savefig("outputs/Phase4/actual_vs_predicted.png")
plt.show()


if best_model_name=="Random Forest":
    trained_forest=best_model.named_steps["model"]
    feature_names=best_model.named_steps["preprocessor"].get_feature_names_out()
    importance_df=pd.DataFrame({"Feature":feature_names,"Importance":trained_forest.feature_importances_})
    importance_df=importance_df.sort_values(by="Importance",ascending=False).head(20)
    importance_df.to_csv("outputs/Phase4/feature_importance.csv",index=False)
    plt.figure(figsize=(12,8))
    sns.barplot(data=importance_df,x="Importance",y="Feature")
    plt.title("Top 20 Important Features")
    plt.tight_layout()
    plt.savefig("outputs/Phase4/feature_importance.png")
    plt.show()


joblib.dump(best_model,"outputs/Phase4/UrbanPulse_best_model.pkl")
print("\nBest Model:",best_model_name)
print("\nFinal Shape",df.shape)

