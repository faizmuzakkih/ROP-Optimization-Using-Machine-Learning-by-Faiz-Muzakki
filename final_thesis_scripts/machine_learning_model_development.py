## -------------------------- CHOOSING MACHINE LEARNING ALGORITHM ON EACH SECTION -------------------------
# LOGISTIC REGRESSION 
# Initialize Logistic Regression model
log_reg = LogisticRegression(random_state=42, class_weight='balanced', max_iter=7000)

# Get list of section columns (one-hot encoded)
section_cols = ['26"', '17 1/2"', '12 1/4"', '8 1/2"']

# Store results
auc_results = []

# Loop through each section
for section in section_cols:
    print(f"\n=== Section: {section} ===")

    # Filter train and test data for the current section
    train_section = df_train_scaled[df_train_scaled[section] == 1]
    test_section = df_test_scaled[df_test_scaled[section] == 1]

    # Ensure there is test data for the section
    if test_section.empty:
        print(f"Skipping Section {section} (No data in test set)")
        continue

    # Define X (input) and y (target)
    X_train = train_section.drop(columns=["ROP_Class"] + section_cols)  
    y_train = train_section["ROP_Class"]

    X_test = test_section.drop(columns=["ROP_Class"] + section_cols)  
    y_test = test_section["ROP_Class"]

    # Train Logistic Regression with class_weight="balanced"
    log_reg.fit(X_train, y_train)

    # Get predicted probabilities for positive class (1)
    y_pred_prob = log_reg.predict_proba(X_test)[:, 1]

    # Compute ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    # Store results
    auc_results.append({"Section": section, "AUC Score": roc_auc})

    # Plot ROC Curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Random-guessing line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC-AUC Curve for Logistic Regression - Section {section}')
    plt.legend(loc="lower right")
    plt.show()

# Convert results to DataFrame and display
auc_results_df = pd.DataFrame(auc_results)
print("\n=== AUC Scores for Each Section ===\n=== Using Logistic Regression ===")
print(auc_results_df)

------------------------------------------------------------------------
# SUPPORT VECTOR MACHINE (SVM)
# Store AUC results
auc_results_svm = []

# Loop through each section
for section in section_cols:
    print(f"\n=== ROC-AUC Curve for Section: {section} ===")
    
    # Filter train and test data for the current section
    train_section = df_train_scaled[df_train_scaled[section] == 1]
    test_section = df_test_scaled[df_test_scaled[section] == 1]

    # Ensure there is test data for the section
    if test_section.empty:
        print(f"Skipping Section {section} (No data in test set)")
        continue

    # Define X (features) and y (target)
    X_train = train_section.drop(columns=["ROP_Class"] + section_cols)
    y_train = train_section["ROP_Class"]
    X_test = test_section.drop(columns=["ROP_Class"] + section_cols)
    y_test = test_section["ROP_Class"]

    # Train SVM with probability enabled
    svm_model = SVC(kernel="rbf", probability=True, random_state=42, class_weight='balanced')
    svm_model.fit(X_train, y_train)

    # Get predicted probabilities for positive class (1)
    y_pred_prob = svm_model.predict_proba(X_test)[:, 1]

    # Compute ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    # Store results
    auc_results_svm.append({"Section": section, "AUC Score": roc_auc})

    # Plot ROC Curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Random-guessing line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC-AUC Curve for SVM - Section {section}')
    plt.legend(loc="lower right")
    plt.show()

# Convert results to DataFrame and display
auc_results_df_svm = pd.DataFrame(auc_results_svm)
print("\n=== AUC Scores for Each Section ===\n=== Using SVM ===")
print(auc_results_df_svm)

------------------------------------------------------------------------
# RANDOM FOREST
# Store AUC results
auc_results = []

# Loop through each section
for section in section_cols:
    print(f"\n=== ROC-AUC Curve for Section: {section} ===")

    # Filter train and test data for the current section
    train_section = df_train[df_train[section] == 1]
    test_section = df_test[df_test[section] == 1]

    # Ensure there is test data for the section
    if test_section.empty:
        print(f"Skipping Section {section} (No data in test set)")
        continue

    # Define X (features) and y (target)
    X_train = train_section.drop(columns=["ROP_Class"] + section_cols)  
    y_train = train_section["ROP_Class"]

    X_test = test_section.drop(columns=["ROP_Class"] + section_cols)  
    y_test = test_section["ROP_Class"]

    # Train Random Forest Classifier
    rf = RandomForestClassifier(random_state=70, class_weight='balanced')
    rf.fit(X_train, y_train)

    # Get predicted probabilities for positive class (1)
    y_pred_prob = rf.predict_proba(X_test)[:, 1]

    # Compute ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    # Store results
    auc_results.append({"Section": section, "AUC Score": roc_auc})

    # Plot ROC Curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC-AUC Curve for Random Forest - Section {section}')
    plt.legend(loc="lower right")
    plt.show()

# Convert results to DataFrame and display
auc_results_df = pd.DataFrame(auc_results)
print("\n=== AUC Scores for Each Section ===\n=== Using Random Forest ===")
print(auc_results_df)


## -------------------------- HYPERPARAMETER TUNING AND MODEL TRAINING & EVALUATION -------------------------
## RANDOM FOREST MODEL TRAINING
# Define sections to include
section_cols = ['26"', '17 1/2"', '12 1/4"', '8 1/2"']

# Filter train and test data for the selected sections
train_section = df_train[(df_train['8 1/2"'] == 1)]
test_section = df_test[(df_test['8 1/2"'] == 1)]

# Define X (features) and y (target)
X_train_RF = train_section.drop(columns=["ROP_Class"] + section_cols)
y_train_RF = train_section["ROP_Class"]
X_test_RF = test_section.drop(columns=["ROP_Class"] + section_cols)
y_test_RF = test_section["ROP_Class"]

# Create the parameter grid based on the results of random search
param_grid_RF = {
    'bootstrap': [True],
    'max_depth': [10, 11, 12],
    'max_features': ['sqrt'],
    'min_samples_leaf': [30, 31, 32],
    'min_samples_split': [34, 35, 36],
    'n_estimators': [250, 280, 300]
}

# Create a base model
base_model_grid = RandomForestClassifier(class_weight='balanced')

# Instantiate the grid search model
grid_search = GridSearchCV(estimator=base_model_grid, param_grid=param_grid_RF, cv=10, n_jobs=-1, verbose=2, error_score='raise')

# Fit the grid search to the filtered training data
grid_search.fit(X_train_RF, y_train_RF)

# Make the method to evaluate the model
def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    
    # compute classification metrics
    accuracy = accuracy_score(test_labels, predictions)
    precision = precision_score(test_labels, predictions, average='binary')
    recall = recall_score(test_labels, predictions, average='binary')
    
    # Print the results
    print('Model Performance (Classification)')
    print(f'Accuracy: {accuracy:.2f}')
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
        
    print('\nConfusion Matrix')
    print(confusion_matrix(test_labels, predictions))
    
    print('\nClassification Report')
    print(classification_report(test_labels, predictions))
    
    return accuracy, precision, recall

# Evaluate the model with hyperparameter generated from GridSearchCV
best_RF_model = grid_search.best_estimator_
grid_accuracy = evaluate(best_RF_model, X_train_RF, y_train_RF)

# Perform stratified k-fold cross-validation & standard deviation
rskf = RepeatedKFold(n_splits=15, n_repeats=5, random_state=70)
cv_scores = cross_val_score(best_RF_model, X_train_RF, y_train_RF, cv=rskf, scoring='accuracy')
print(f'Cross-Validation Accuracy Scores: {cv_scores}')
print(f'Mean CV Accuracy: {np.mean(cv_scores):.2f}')
print(f'Standard Deviation: {np.std(cv_scores):.2f}')

------------------------------------------------------------------------
## SUPPORT VECTOR MACHINE (SVM) MODEL TRAINING
# Filter train and test data for the selected sections
train_section = df_train_scaled[df_train_scaled['26"'] == 1]
test_section = df_test_scaled[df_test_scaled['26"'] == 1]

# Define X (features) and y (target)
X_train_SVM = train_section.drop(columns=["ROP_Class"] + section_cols)
y_train_SVM = train_section["ROP_Class"]
X_test_SVM = test_section.drop(columns=["ROP_Class"] + section_cols)
y_test_SVM = test_section["ROP_Class"]

# Create the parameter grid based on the results of random search
param_grid_SVM = {
    'C': [0.0001, 0.001, 0.01, 0.1],
    'kernel': ['linear']
}

# Create a base model
base_model_SVM = SVC(class_weight='balanced', probability=True)

# Instantiate the grid search model
grid_search_SVM = GridSearchCV(estimator=base_model_SVM, param_grid=param_grid_SVM, cv=10, scoring='recall', n_jobs=-1, verbose=2, error_score='raise')

# Fit the grid search to the filtered training data
grid_search_SVM.fit(X_train_SVM, y_train_SVM)

# Evaluate the model with hyperparameter generated from GridSearchCV
best_SVM_model = grid_search_SVM.best_estimator_
random_accuracy = evaluate(best_SVM_model, X_test_SVM, y_test_SVM)

# Perform stratified k-fold cross-validation
rskf = RepeatedKFold(n_splits=15, n_repeats=5, random_state=70)
cv_scores = cross_val_score(best_SVM_model, X_train_SVM, y_train_SVM, cv=rskf, scoring='accuracy')
print(f'Cross-Validation Accuracy Scores: {cv_scores}')
print(f'Mean CV Accuracy: {np.mean(cv_scores):.2f}')
print(f'Standard Deviation: {np.std(cv_scores):.2f}')

------------------------------------------------------------------------
## LOGISTIC REGRESSION MODEL TRAINING
# Filter train and test data for the selected sections
train_section = df_train_scaled[(df_train_scaled['17 1/2"'] == 1) | (df_train_scaled['12 1/4"'] == 1)]
test_section = df_test_scaled[(df_test_scaled['17 1/2"'] == 1) | (df_test_scaled['12 1/4"'] == 1)]

# Define X (features) and y (target)
X_train_LogReg = train_section.drop(columns=["ROP_Class"] + section_cols)
y_train_LogReg = train_section["ROP_Class"]
X_test_LogReg = test_section.drop(columns=["ROP_Class"] + section_cols)
y_test_LogReg = test_section["ROP_Class"]

# Create the parameter grid based on the results of random search
param_grid_LogReg = {
    'penalty': ['elasticnet'],
    'C': [0.007, 0.01, 0.12],
    'solver': ['saga'],
    'max_iter': [1000, 1200, 1500],
    'l1_ratio': [0.7, 0.8, 0.9]
}

# Create a base model
base_model_LogReg = LogisticRegression(class_weight='balanced')

# Instantiate the grid search model
grid_search_LogReg = GridSearchCV(estimator=base_model_LogReg, param_grid=param_grid_LogReg, cv=10, scoring='accuracy', n_jobs=-1, verbose=2, error_score='raise')

# Fit the grid search to the filtered training data
grid_search_LogReg.fit(X_train_LogReg, y_train_LogReg)

# Evaluate the model with hyperparameter generated from GridSearchCV
best_LogReg_model = grid_search_LogReg.best_estimator_
grid_accuracy = evaluate(best_LogReg_model, X_test_LogReg, y_test_LogReg)

# Perform stratified k-fold cross-validation
rskf = RepeatedKFold(n_splits=15, n_repeats=5, random_state=70)
cv_scores = cross_val_score(best_LogReg_model, X_train_LogReg, y_train_LogReg, cv=rskf, scoring='accuracy')
print(f'Cross-Validation Accuracy Scores: {cv_scores}')
print(f'Mean CV Accuracy: {np.mean(cv_scores):.2f}')
print(f'Standard Deviation: {np.std(cv_scores):.2f}')
