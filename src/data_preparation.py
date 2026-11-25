import json                                   
from pathlib import Path                      
import matplotlib
matplotlib.use("Agg")                         
import matplotlib.pyplot as plt
import pandas as pd

#Project paths
ROOT = Path(__file__).resolve().parent.parent    
RAW = ROOT / "data" / "hour.csv"                 
OUT = ROOT / "data" / "processed"               
GRAPHS = ROOT / "results" / "graphs"              

#Dataset loading
raw = pd.read_csv(RAW)                           
print(raw.shape, raw.dtypes, raw.head(), sep="\n")  #to inspect size, column types, sample rows

#Data validation and cleaning 
df = raw.copy() #to work on a copy 
print("Missing:", df.isna().sum().sum(), "| Duplicates:", df.duplicated().sum())  
df = df.dropna().drop_duplicates() #remove rows with missing values and duplicate rows
df["dteday"] = pd.to_datetime(df["dteday"]) #convert date strings to real datetime values
df = df[df["cnt"] == df["casual"] + df["registered"]] #keep rows where total = casual + registered
df = df[df["hr"].between(0, 23) & df["weathersit"].between(1, 4)] #drop rows with invalid hour or weather code
df = df.sort_values(["dteday", "hr"]).reset_index(drop=True)  #sort by time 

#EDA
GRAPHS.mkdir(parents=True, exist_ok=True)       
for col in ["hr", "weekday", "season", "mnth", "holiday", "workingday", "weathersit"]:
    #average rentals for each value of the column - line for hour, bars for the rest
    df.groupby(col)["cnt"].mean().plot(kind="bar" if col != "hr" else "line")
    plt.title(f"Mean rentals by {col}")
    plt.ylabel("cnt")
    plt.tight_layout()                            
    plt.savefig(GRAPHS / f"eda_{col}.png", dpi=120) #to save the plot as an image
    plt.close()                                 

#Feature selection and separation
#Linear features: yr, holiday, workingday are already 0/1, so they are used directly
X_linear = df[["yr", "holiday", "workingday"]].copy()
X_linear = pd.concat([
    X_linear,
    #drop_first avoids perfect collinearity
    pd.get_dummies(pd.Categorical(df["season"], [1, 2, 3, 4]), prefix="season", drop_first=True, dtype=int),
    pd.get_dummies(pd.Categorical(df["weekday"], range(7)), prefix="weekday", drop_first=True, dtype=int),
], axis=1)                                        # axis=1 joins the columns side by side

#Nonlinear features
X_nonlinear = pd.DataFrame({
    "hr": df["hr"],
    "weathersit": df["weathersit"],
    "hr_x_workingday": df["hr"] * df["workingday"],   
    "hr_x_weathersit": df["hr"] * df["weathersit"],   
})
y = df["cnt"]                                    

#Dataset splitting (chronological)
n = len(df)                                     
i1, i2 = int(n * 0.70), int(n * 0.85) #cut points at 70% and 85%
# first 70% = train, next 15% = validation, last 15% = test (no shuffling)
bounds = {"train": slice(0, i1), "validation": slice(i1, i2), "test": slice(i2, n)}