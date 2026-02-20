import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#--- Load dataset ---
data_path = r"C:\pythonprojects\bank-marketing-ml\data\raw\bank-full.csv"

df = pd.read_csv(data_path, sep = ";")
df.columns = df.columns.str.strip()
    
#--- Target counts ---    
print("\nTarget counts : ")
print(df["y"].value_counts())

#--- Numerical ---
numerical_cols = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]
print("\n---Numerical Features Summary--- ")
print(df[numerical_cols].describe())

#--- Histogram + KDE for numericals
for col in numerical_cols:
    plt.figure(figsize = (6, 3))
    sns.histplot(df[col], bins = 30, kde = True, color = "skyblue")
    plt.title(f"Distribution of {col}")
    plt.show()
    
    #--- Categorical ---
categorical_cols = ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome"]
for col in categorical_cols:
    print(f"\n---{col} value counts---")
    print(df[col].value_counts())
    print(f"\n---{col} vs target---")
    print(df.groupby(col)
["y"].value_counts(normalize = True).unstack())
    
    #Count plot categoricals
    plt.figure(figsize = (7, 3))
    sns.countplot(data = df, x = col, hue = "y")
    plt.title(f"{col} vs Target")
    plt.xticks(rotation = 45)
    plt.show()
