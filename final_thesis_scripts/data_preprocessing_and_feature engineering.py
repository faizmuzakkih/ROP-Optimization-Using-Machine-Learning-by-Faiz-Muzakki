## --------------------------------- DATA PREPROCESSING ---------------------------------------

# COLUMN REDUCTION
df.drop(columns=['C1', 'C2', 'C3', 'IC4', 'NC4', 'IC5', 'NC5', 'TGAS', 'GQC'], inplace=True)
-------------------------------------------------------------------
# Convert data type into numerical format
for col in df.columns:
    try:
        df[col] = df[col].astype('float64')
    except ValueError:
        # skip columns that cannot be converted
        pass
-------------------------------------------------------------------
# VISUALIZATION OF EACH FEATURE
# Define number of subplots
num_cols = len(df.drop(columns=['WELL_ID', 'Section']).columns)
rows = int(np.ceil(num_cols / 4))

# Create figure and axes
fig, axes = plt.subplots(rows, 4, figsize=(20, 15))
axes = axes.flatten()

# Plot histogram for each selected parameter
for i, col in enumerate(df.drop(columns=['WELL_ID', 'Section']).columns):
    sns.histplot(df[col], bins=30, kde=True, color="blue", ax=axes[i])
    axes[i].set_title(f'Histogram of {col}')

# Hide unused subplots
for i in range(num_cols, len(axes)):  
    fig.delaxes(axes[i])  

plt.tight_layout()
plt.show()
-------------------------------------------------------------------
# DATA SPLITTING
train_wells = ['A-3', 'A-5']  # Training wells
test_well = ['A-4']  # Testing well

df_train = df[df['WELL_ID'].isin(train_wells)].copy()
df_test = df[df['WELL_ID'].isin(test_well)].copy()
-------------------------------------------------------------------
# DATA CLEANING
# check the missing data
df_train.isnull().sum()


# check outliers
parameters = ['MDEPTH', 'TVD', 'TRPM', 'WOB', 'MTI', 'MTO', 'GPM', 'MWI', 'MWO', 'ECD']
# show boxplot
df_train[parameters].plot(kind='box', subplots=True, layout=(3, 4), sharex=False, sharey=False, figsize=(20, 10), color='k')
plt.show()


# Summarize outliers
# Outlier Handling
Q1 = df_train[parameters].quantile(0.25)
Q3 = df_train[parameters].quantile(0.75)
IQR = Q3 - Q1

# the upper and lower limit
upper_limit = Q3 + 1.5 * IQR
lower_limit = Q1 - 1.5 * IQR

# mark data as outlier or normal
outlier_status = df_train[parameters].apply(lambda x:
    ((x < lower_limit[x.name]) | (x > upper_limit[x.name])).astype(float), axis=0)

# adding the outlier and normal data in each column
outlier_count = outlier_status.sum()
normal_count = len(df_train[parameters]) - outlier_count

# create the result table
outlier_summary = pd.DataFrame({
    'Column' : df_train[parameters].columns,
    'Amount of Outlier' : outlier_count,
    'Amount of Normal' : normal_count
}).set_index('Column')
# visualize the table
outlier_summary.sort_values(by= 'Amount of Outlier', ascending=False)


# Handling outliers by capping
df_train_no_outliers = df_train.copy()

for col in parameters:
    df_train_no_outliers[col] = df_train[col].clip(lower=lower_limit[col], upper=upper_limit[col])

# visualize using boxplot
df_train_no_outliers[parameters].plot(kind='box', subplots=True, sharex=False, sharey=False, layout=(4, 4), figsize=(20, 15), title='After Outlier Handling', color='k')
plt.show()


# visualize the data distribution
num_cols = len(parameters)
rows = int(np.ceil(num_cols / 4))

# Create figure and axes
fig, axes = plt.subplots(rows, 4, figsize=(20, 15))
axes = axes.flatten()

# Plot histogram for each selected parameter
for i, col in enumerate(parameters):
    sns.histplot(df_train_no_outliers[col], bins=30, kde=True, color="blue", ax=axes[i])
    axes[i].set_title(f'Histogram of {col}')

# Hide unused subplots
for i in range(num_cols, len(axes)):  
    fig.delaxes(axes[i])  

plt.tight_layout()
plt.show()


# Check the duplicates data
duplicates = df_train_no_outliers[df_train_no_outliers.duplicated(keep=False)]

# measure amount of duplicate
duplicates_count = len(duplicates)

# measure percentage of duplicate
duplicates_percentage = (duplicates_count / len(df_train_no_outliers)) * 100

# visualize the result
print(f'Amount of duplicates row: {duplicates_count}')
print(f'Percentage duplicate: {duplicates_percentage:.2f}%')
-------------------------------------------------------------------
# ENCODING CATEGORICAL VARIABLE
dummies = pd.get_dummies(df['Section']).astype(int)
# merge the dummy variables with the original data
merged_df = pd.concat([df, dummies], axis=1)
# drop the section column
df = merged_df.drop(['Section'], axis=1)
-------------------------------------------------------------------
# SEPARATE ROP VALUES USING 75TH PERCENTILE THRESHOLD
sections = ['26"', '17 1/2"', '12 1/4"', '8 1/2"']

# Create subplots
fig, axes = plt.subplots(figsize=(10, 7), nrows=2, ncols=2)

# Loop through each layer
for i, section in enumerate(sections):
    
    # Select ROP values for the current layer
    rop_values = df_train.loc[df_train[section] == 1, 'ROP']
    
    # Compute 75th percentile (Q3) threshold
    q3 = rop_values.quantile(0.75)
        
    # KDE Plot - Separate below and above Q3
    # sns.kdeplot(rop_values, ax=axes[i // 2, i % 2], color='black')

    # Generate KDE values for shading
    x = np.linspace(rop_values.min(), rop_values.max(), num=100)
    kde = sns.kdeplot(rop_values, ax=axes[i // 2, i % 2], bw_adjust=0.8).get_lines()[0].get_data()
    x_vals, y_vals = kde[0], kde[1]
       
    # Fill areas based on Q3 threshold
    axes[i // 2, i % 2].fill_between(x_vals, y_vals, where=(x_vals < q3), color='red', alpha=0.5)
    axes[i // 2, i % 2].fill_between(x_vals, y_vals, where=(x_vals >= q3), color='green', alpha=0.5)
 
    # Add vertical threshold line
    axes[i // 2, i % 2].axvline(q3, color='black', linestyle='dashed', linewidth=2)
       
    # Titles and labels
    axes[i // 2, i % 2].set_title(f"{section} Section")
    axes[i // 2, i % 2].set_xlabel("ROP (m/hr)")
    axes[i // 2, i % 2].set_ylabel("Density")
    
plt.tight_layout()
plt.show()

# Convert ROP to discrete values (low and high)
df_train["ROP_Class"] = np.nan

# Iterate through each layer and classify ROP
for section in sections:
    rop_values = df_train.loc[df_train[section] == 1, "ROP"]  # Get ROP values for this layer
    q3 = rop_values.quantile(0.75)
    
    # Assign "Low" (0) or "High" (1) based on Q3
    df_train.loc[df_train[section] == 1, "ROP_Class"] = np.where(df_train.loc[df_train[section] == 1, "ROP"] < q3, 0, 1)

# Drop the original ROP column
df_train = df_train.drop(columns=['ROP'])


## --------------------------------- FEATURE ENGINEERING ---------------------------------------
# FEATURE SCALING
# Standardization Technique (Z-score scaling)
# define X and y
X_train = df_train_no_outliers.drop(columns=['ROP_Class', '26"', '17 1/2"', '12 1/4"', '8 1/2"'])
y_train = df_train_no_outliers[['ROP_Class']] 
X_test = df_test.drop(columns=['ROP_Class', '26"', '17 1/2"', '12 1/4"', '8 1/2"'])
y_test = df_test[['ROP_Class']]

# set up scaler, initialize
scaler = StandardScaler().set_output(transform='pandas')

# transform train and test sets
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# check the stats of the scaled data
X_train_scaled.describe().round(2)


# Check distribution after scaling
# Define number of subplots
num_cols = len(parameters)
rows = int(np.ceil(num_cols / 4))

# Create figure and axes
fig, axes = plt.subplots(rows, 4, figsize=(20, 15))
axes = axes.flatten()

# Plot histogram for each selected parameter
for i, col in enumerate(parameters):
    sns.histplot(X_train_scaled[col], bins=30, kde=True, color="blue", ax=axes[i])
    axes[i].set_title(f'Histogram of {col}')

# Hide unused subplots
for i in range(num_cols, len(axes)):  
    fig.delaxes(axes[i])  

plt.tight_layout()
plt.show()


# Visualize original data and scaled data in one plot
plt.figure(figsize=(8,6))
plt.scatter(df['WOB'], df['TRPM'], color='red', label='Input Scale', alpha=0.4)
plt.scatter(X_test_scaled['WOB'], X_test_scaled['TRPM'], color='blue', label='Standardized (Test Data)', alpha=0.5)
plt.scatter(X_train_scaled['WOB'], X_train_scaled['TRPM'], color='green', label='Standardized (Train Data)', alpha=0.5)

plt.title('WOB and TRPM')
plt.xlabel('WOB')
plt.ylabel('TRPM')
plt.legend(loc='lower right')
plt.grid()
plt.tight_layout()
plt.show()


# Replace the original df after feature scaling
scaled_features = ['MDEPTH', 'TVD', 'TRPM', 'WOB', 'MTI', 'MTO', 'GPM', 'MWI', 'MWO', 'ECD']
scaled_df = pd.DataFrame(X_train_scaled, columns=scaled_features, index=df_train_no_outliers.index)
df_train_scaled = df_train_no_outliers.assign(**scaled_df)

scaled_df_test = pd.DataFrame(X_test_scaled, columns=scaled_features, index=df_test.index)
df_test_scaled = df_test.assign(**scaled_df_test)

df_train_scaled.describe().round(2)
