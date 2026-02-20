import pandas as pd

#--- Load dataset ---
datapath = r"C:\pythonprojects\bank-marketing-ml\data\raw\bank-full.csv"
df = pd.read_csv(datapath, sep = ";")

#--- Remove extra space --- 
df.columns = df.columns.str.strip()

#--- Remove extra columns ---
if "duration" in df.columns:
    df = df.drop("duration", axis = 1)
    print("Column 'duration' dropped (data leakage) . ")
    
#--- convert Target to numeric ---
df["y_binary"] = df["y"].map({"yes" : 1, "no" : 0})
df = df.drop("y", axis = 1)
print("Target column converted to numeric : y_binary")

#--- Show target counts ---
print("\nTarget counts : ")
print(df["y_binary"].value_counts())

#--- Categorical columns ---
categorical_cols = ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome"]

#--- One-Hot Encoding ---
df_encoded = pd.get_dummies(df, columns = categorical_cols, drop_first = True)
print("\nCategorical columns encoded . ")

#--- preparing x, y
x = df_encoded.drop("y_binary", axis = 1)
y = df_encoded["y_binary"]

print("\nShape of x : ", x.shape)
print("Shape of y : ", y.shape) 