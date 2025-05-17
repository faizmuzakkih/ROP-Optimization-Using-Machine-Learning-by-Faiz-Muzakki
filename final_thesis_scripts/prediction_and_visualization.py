## ------------------------------------ FEATURE SELECTION USING PEARSON CORR --------------------------------------------
# Compute correlation matrix
corr_matrix = df_train.drop(columns=['WELL_ID', '26"', '17 1/2"', '12 1/4"', '8 1/2"']).corr()

# Plot heatmap
plt.figure(figsize=(12,8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
# plt.title("Feature Correlation Heatmap")
plt.show()


## ----------------------------------------- RESULT FOR 8 1/2" SECTION -------------------------------------------------
# Filter dataset for 8 1/2" section
section_data = df_train[df_train['8 1/2"'] == 1]

# Define percentiles for GPM
gpm_25 = np.percentile(section_data['GPM'], 25)
gpm_50 = np.median(section_data['GPM'])
gpm_75 = np.percentile(section_data['GPM'], 75)

gpm_values = [gpm_25, gpm_50, gpm_75]
gpm_labels = [f'{gpm_25} GPM (25th Percentile)', f'{gpm_50} GPM (Median)', f'{gpm_75} GPM (75th Percentile)']

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, (gpm, label) in enumerate(zip(gpm_values, gpm_labels)):
    ax = axes[i]
    
    # Define RPM and WOB range
    rpm_grid, wob_grid = np.meshgrid(
        np.linspace(section_data['TRPM'].min(), section_data['TRPM'].max(), 100),
        np.linspace(section_data['WOB'].min(), section_data['WOB'].max(), 100)
    )
    
    # Flatten grid for prediction
    grid_points = np.c_[rpm_grid.ravel(), wob_grid.ravel(), np.full(rpm_grid.size, gpm)]
    
    # Create a DataFrame with appropriate column names
    grid_points_df = pd.DataFrame(grid_points, columns=['TRPM', 'WOB', 'GPM'])
    
    # Predict probability of high ROP
    probability_grid = best_RF_model.predict_proba(grid_points_df)[:, 1]
    probability_grid = probability_grid.reshape(rpm_grid.shape)
    
    # Plot contour
    contour = ax.contourf(wob_grid, rpm_grid, probability_grid, levels=50, cmap='viridis')
    fig.colorbar(contour, ax=ax, label='Probability of High ROP')
    
    # Scatter actual data points
    sns.scatterplot(x=section_data['WOB'], y=section_data['TRPM'], ax=ax, color='r',  edgecolor='k', s=20, label='Actual Data')
    
    ax.set_title(f'FR: {label}')
    ax.set_xlabel('WOB (klbs)')
    ax.set_ylabel('TRPM')
    ax.legend()

plt.suptitle('Probability Contour Maps of High ROP in 8 1/2" Section', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()


## ----------------------------------------- RESULT FOR 12 1/4" SECTION -------------------------------------------------
# Filter unscaled original training data for the desired section
section_data_original = df_train_no_outliers[df_train_no_outliers['12 1/4"'] == 1]

# Extract the same rows in scaled data if needed
section_data_scaled = df_train_scaled[df_train_scaled['12 1/4"'] == 1]  # Only used if needed

# Define GPM percentiles from the unscaled section data
gpm_25 = np.percentile(section_data_original['GPM'], 25)
gpm_50 = np.median(section_data_original['GPM'])
gpm_75 = np.percentile(section_data_original['GPM'], 75)
gpm_values = [gpm_25, gpm_50, gpm_75]
gpm_labels = ['25th Percentile', 'Median', '75th Percentile']

# Create the visualization
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, (gpm, label) in enumerate(zip(gpm_values, gpm_labels)):
    ax = axes[i]

    # Generate meshgrid using unscaled (real) WOB and RPM ranges
    rpm_vals = np.linspace(section_data_original['TRPM'].min(), section_data_original['TRPM'].max(), 100)
    wob_vals = np.linspace(section_data_original['WOB'].min(), section_data_original['WOB'].max(), 100)
    rpm_grid, wob_grid = np.meshgrid(rpm_vals, wob_vals)

    # Flatten grid and create DataFrame
    grid_points = np.c_[rpm_grid.ravel(), wob_grid.ravel(), np.full(rpm_grid.size, gpm)]
    grid_df = pd.DataFrame(grid_points, columns=['TRPM', 'WOB', 'GPM'])

    # Scale the grid (same scaler used for training)
    grid_scaled = scaler.fit_transform(grid_df)

    # Predict probabilities
    prob_grid = best_LogReg_model.predict_proba(grid_scaled)[:, 1].reshape(rpm_grid.shape)

    # Plot the contour
    contour = ax.contourf(wob_grid, rpm_grid, prob_grid, levels=50, cmap='viridis')
    fig.colorbar(contour, ax=ax, label='Probability of High ROP')

    # Plot actual unscaled data points for the section
    sns.scatterplot(x=section_data_original['WOB'], y=section_data_original['TRPM'], ax=ax, color='red',
        edgecolor='black', s=20, label='Actual Data')

    ax.set_title(f'FR: {round(gpm, 2)} GPM ({label})')
    ax.set_xlabel('WOB (klbs)')
    ax.set_ylabel('TRPM')
    ax.legend(loc='lower right')
    
plt.suptitle('Probability Contour Map of High ROP in 12 1/4" Section', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()


## ----------------------------------------- RESULT FOR 17 1/2" SECTION -------------------------------------------------
# Filter unscaled original training data for the desired section
section_data_original = df_train_no_outliers[df_train_no_outliers['17 1/2"'] == 1]

# Extract the same rows in scaled data if needed
section_data_scaled = df_train_scaled[df_train_scaled['17 1/2"'] == 1]  # Only used if needed

# Define GPM percentiles from the unscaled section data
gpm_25 = np.percentile(section_data_original['GPM'], 25)
gpm_50 = np.median(section_data_original['GPM'])
gpm_75 = np.percentile(section_data_original['GPM'], 75)
gpm_values = [gpm_25, gpm_50, gpm_75]
gpm_labels = ['25th Percentile', 'Median', '75th Percentile']

# Create the visualization
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, (gpm, label) in enumerate(zip(gpm_values, gpm_labels)):
    ax = axes[i]

    # Generate meshgrid using unscaled (real) WOB and RPM ranges
    rpm_vals = np.linspace(section_data_original['TRPM'].min(), section_data_original['TRPM'].max(), 100)
    wob_vals = np.linspace(section_data_original['WOB'].min(), section_data_original['WOB'].max(), 100)
    rpm_grid, wob_grid = np.meshgrid(rpm_vals, wob_vals)

    # Flatten grid and create DataFrame
    grid_points = np.c_[rpm_grid.ravel(), wob_grid.ravel(), np.full(rpm_grid.size, gpm)]
    grid_df = pd.DataFrame(grid_points, columns=['TRPM', 'WOB', 'GPM'])

    # Scale the grid (same scaler used for training)
    grid_scaled = scaler.fit_transform(grid_df)

    # Predict probabilities
    prob_grid = best_LogReg_model.predict_proba(grid_scaled)[:, 1].reshape(rpm_grid.shape)

    # Plot the contour
    contour = ax.contourf(wob_grid, rpm_grid, prob_grid, levels=50, cmap='viridis')
    fig.colorbar(contour, ax=ax, label='Probability of High ROP')

    # Plot actual unscaled data points for the section
    sns.scatterplot(x=section_data_original['WOB'], y=section_data_original['TRPM'], ax=ax, color='red',
        edgecolor='black', s=20, label='Actual Data')

    ax.set_title(f'FR: {round(gpm, 2)} GPM ({label})')
    ax.set_xlabel('WOB (klbs)')
    ax.set_ylabel('TRPM')
    ax.legend(loc='lower right')

plt.suptitle('Probability Contour Map of High ROP in 17 1/2" Section', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()


## ----------------------------------------- RESULT FOR 26" SECTION -------------------------------------------------
# Filter dataset for 26" section
section_data = df_train[df_train['26"'] == 1]

# Define percentiles for GPM
gpm_25 = np.percentile(section_data['GPM'], 25)
gpm_50 = np.median(section_data['GPM'])
gpm_75 = np.percentile(section_data['GPM'], 75)

gpm_values = [gpm_25, gpm_50, gpm_75]
gpm_labels = [f'{gpm_25} GPM (25th Percentile)', f'{gpm_50} GPM (Median)', f'{gpm_75} GPM (75th Percentile)']

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, (gpm, label) in enumerate(zip(gpm_values, gpm_labels)):
    ax = axes[i]
    
    # Define RPM and WOB range
    rpm_grid, wob_grid = np.meshgrid(
        np.linspace(section_data['TRPM'].min(), section_data['TRPM'].max(), 100),
        np.linspace(section_data['WOB'].min(), section_data['WOB'].max(), 100)
    )
    
    # Flatten grid for prediction
    grid_points = np.c_[rpm_grid.ravel(), wob_grid.ravel(), np.full(rpm_grid.size, gpm)]
    
    # Create a DataFrame with appropriate column names
    grid_points_df = pd.DataFrame(grid_points, columns=['TRPM', 'WOB', 'GPM'])
    
    # Predict probability of high ROP
    probability_grid = best_RF_model.predict_proba(grid_points_df)[:, 1]
    probability_grid = probability_grid.reshape(rpm_grid.shape)
    
    # Plot contour
    contour = ax.contourf(wob_grid, rpm_grid, probability_grid, levels=50, cmap='viridis')
    fig.colorbar(contour, ax=ax, label='Probability of High ROP')
    
    # Scatter actual data points
    sns.scatterplot(x=section_data['WOB'], y=section_data['TRPM'], ax=ax, color='r',  edgecolor='k', s=20, label='Actual Data')
    
    ax.set_title(f'FR: {label}')
    ax.set_xlabel('WOB (klbs)')
    ax.set_ylabel('TRPM')
    ax.legend(loc='lower right')

plt.suptitle('Probability Contour Maps of High ROP in 26" Section', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
