# =========================
# app.py
# Unified Cognitive Health + Environment Analysis Pipeline
# =========================

import pandas as pd
import numpy as np
import sqlite3

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from sklearn.impute import SimpleImputer

from sklearn.metrics import r2_score, mean_squared_error, accuracy_score

import warnings
warnings.filterwarnings("ignore")

# =========================================================
# 1. DATABASE SETUP
# =========================================================

conn = sqlite3.connect("nhanes.db")

# =========================================================
# 2. LOAD + CLEAN NHANES DATA (INDIVIDUAL LEVEL)
# =========================================================

def load_nhanes():
    demo = pd.read_sas("data/raw/nhanes/DEMO_H.xpt")[["SEQN","RIDAGEYR","RIAGENDR","DMDHREDU"]]
    cfq  = pd.read_sas("data/raw/nhanes/CFQ_H.xpt")[["SEQN","CFDDS"]]
    bmx  = pd.read_sas("data/raw/nhanes/BMX_H.xpt")[["SEQN","BMXWT","BMXHT"]]
    smq  = pd.read_sas("data/raw/nhanes/SMQ_H.xpt")[["SEQN","SMQ020"]]
    paq  = pd.read_sas("data/raw/nhanes/PAQ_H.xpt")[["SEQN","PAQ605"]]

    # rename
    demo.columns = ["id","age","gender","education"]
    cfq.columns  = ["id","dsst"]
    bmx.columns  = ["id","weight","height"]
    smq.columns  = ["id","smoking"]
    paq.columns  = ["id","activity"]

    # clean demo
    demo["gender"] = demo["gender"].map({1:"male",2:"female"})
    demo = demo.dropna(subset=["age","gender"])

    demo = demo[demo["education"].isin([1,2,3,4,5])]
    demo["education"] = demo["education"].map({
        1: "<9th grade",
        2: "9-11th grade",
        3: "high school/ged",
        4: "some college",
        5: "college graduate",
    })

    # clean target
    cfq = cfq.dropna()
    cfq = cfq[(cfq["dsst"] >= 0) & (cfq["dsst"] <= 105)]

    # BMI
    bmx = bmx.dropna()
    bmx["bmi"] = bmx["weight"] / ((bmx["height"]/100)**2)

    # smoking + activity
    smq["smoking"] = smq["smoking"].map({1:"smoker",2:"non_smoker"})
    paq["activity"] = paq["activity"].map({1:"active",2:"inactive"})

    # merge
    df = demo.merge(cfq,on="id")\
             .merge(bmx,on="id")\
             .merge(smq,on="id")\
             .merge(paq,on="id")

    return df


nhanes = load_nhanes()


# =========================================================
# 3. NHANES REGRESSION MODEL (DSST PREDICTION)
# =========================================================

X = nhanes[["age","gender","education","bmi","smoking","activity"]]
y = nhanes["dsst"]

categorical = ["gender","education","smoking","activity"]
numeric = ["age","bmi"]

preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ("num", "passthrough", numeric)
])

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

lr = Pipeline([
    ("prep", preprocess),
    ("model", LinearRegression())
])

lr.fit(X_train,y_train)
pred_lr = lr.predict(X_test)

print("\n=== NHANES Regression Results ===")
print("R2:", r2_score(y_test,pred_lr))
print("RMSE:", np.sqrt(mean_squared_error(y_test,pred_lr)))


rf = Pipeline([
    ("prep", preprocess),
    ("model", RandomForestRegressor(n_estimators=100, random_state=42))
])

rf.fit(X_train,y_train)
pred_rf = rf.predict(X_test)


# =========================================================
# 4. POPULATION SUMMARY (CYCLE-LEVEL TREND)
# =========================================================

def cycle_summary(file, label):
    df = pd.read_sas(file)[["SEQN","CFDDS"]]
    df = df.dropna()
    df = df[(df["CFDDS"]>=0)&(df["CFDDS"]<=105)]
    df["cycle"] = label
    return df

c1 = cycle_summary("data/raw/nhanes/CFQ_G.xpt","2011-2012")
c2 = cycle_summary("data/raw/nhanes/CFQ_H.xpt","2013-2014")

cycles = pd.concat([c1,c2])

cycle_summary_df = cycles.groupby("cycle").mean().reset_index()
print("\n=== Cycle-Level Cognitive Trends ===")
print(cycle_summary_df)


# =========================================================
# 5. BRFSS + AQI EXTERNAL VALIDATION (CLASSIFICATION)
# =========================================================

brfss = pd.read_csv("data/raw/brfss/brfss_sample.csv")

brfss = brfss[["_STATE","CIMEMLOS","_AGEG5YR","EDUCA","_BMI5","SMOKE100","DIABETE4"]]

brfss.columns = ["state","cog","age","edu","bmi","smoke","diabetes"]

brfss = brfss[brfss["cog"].isin([1,2])]
brfss["cog"] = brfss["cog"].map({1:1,2:0})

state_map = {
    12: "Florida",
    16: "Idaho",
    18: "Indiana",
    23: "Maine",
    41: "Oregon",
    44: "Rhode Island",
    45: "South Carolina",
    49: "Utah",
    50: "Vermont",
    51: "Virginia",
    55: "Wisconsin"
}

brfss["state_name"] = brfss["state_fips"].astype(int).map(state_map)

aqi = pd.read_csv("data/raw/epa/daily_aqi_by_county_2022.csv")[["State Name","AQI"]]
aqi.columns = ["state","aqi"]

aqi = aqi.groupby("state").mean().reset_index()

merged = brfss.merge(
    aqi,
    left_on = "state_name",
    right_on = "state",
    how = "left"
)

features = ["aqi","age","edu","bmi","smoke","diabetes"]

X = merged[features]
y = merged["cog"]

X = SimpleImputer(strategy="median").fit_transform(X)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

log = LogisticRegression(max_iter=1000)
tree = DecisionTreeClassifier(max_depth=5)
rf_c = RandomForestClassifier(n_estimators=200)

for model in [log,tree,rf_c]:
    model.fit(X_train,y_train)
    pred = model.predict(X_test)
    print("\nModel Accuracy:", accuracy_score(y_test,pred))


# =========================================================
# 6. FEATURE IMPORTANCE (BRFSS MODEL)
# =========================================================

rf_c.fit(X_train,y_train)

importance = pd.Series(rf_c.feature_importances_,index=features)

print("\n=== Feature Importance ===")
print(importance.sort_values(ascending=False))
