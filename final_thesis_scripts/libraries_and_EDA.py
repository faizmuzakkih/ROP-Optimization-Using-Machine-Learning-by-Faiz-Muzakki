# Importing libraries that'll be used
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, accuracy_score, classification_report
from sklearn.metrics import precision_score, recall_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, RepeatedKFold

# Importing wells data
df_A_3 = pd.read_excel(r'D:\Thesis\Wells Data\A-03 ASCII.xlsx', sheet_name='ASCII')
df_A_4 = pd.read_excel(r'D:\Thesis\Wells Data\A-04 ASCII.xlsx', sheet_name='ASCII')
df_A_5 = pd.read_excel(r'D:\Thesis\Wells Data\A-05 ASCII.xlsx', sheet_name='ASCII')


'--------------------- Exploratory Data Analysis (EDA) ---------------------------'
## Create visual for ROP Distributions across three wells
# create figure and subplots
fig, axes = plt.subplots(1, 3, figsize=(11, 4))

# list of wells and subplots titles
wells = [df_A_3, df_A_4, df_A_5]
titles = ['A-3 ROP Distribution', 'A-4 ROP Distribution', 'A-5 ROP Distribution']

# plot distributions
for i, (well, title) in enumerate(zip(wells, titles)):
    sns.histplot(well['ROP'], kde=True, ax=axes[i], bins=30)
    axes[i].set_title(title)
    axes[i].set_xlabel('ROP (m/hr)')
    axes[i].set_ylabel('Frequency')

# adjust layout
plt.tight_layout()
plt.show()
------------------------------------------------------
## Compute ROP statistics for each well
# list of wells
wells = [df_A_3, df_A_4, df_A_5]
well_names = ['A-3', 'A-4', 'A-5']

# compute statistics
for well, name in zip(wells, well_names):
    section_stats = well.groupby('Section')['ROP'].agg(['mean', 'median', lambda x: x.mode().iloc[0], 'min', 'max']).reset_index()
    section_stats.rename(columns={'<lambda_0>': 'mode'}, inplace=True)
    
    print(f'\n{name} - ROP Statistics per section:')
    print(section_stats)
------------------------------------------------------
## Add column 'WELL_ID' and merge dataset to be one dataset
# Add WELL_ID column to each dataset
df_A_3["WELL_ID"] = "A-3"
df_A_4["WELL_ID"] = "A-4"
df_A_5["WELL_ID"] = "A-5"

# Concatenate the datasets
df = pd.concat([df_A_3, df_A_4, df_A_5], ignore_index=True)
------------------------------------------------------
## Explore the merged dataset
df.head()
df.info()
df.describe()
df.columns
df.shape
