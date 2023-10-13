# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
# Importing the libraries
import pandas as pd
import numpy as np 
from sklearn.impute import SimpleImputer
import os
import matplotlib.pyplot as plt
from pickle import dump, load

# %matplotlib inline

# +
# Setting up directory paths
base_dir = "/mnt/c/Users/ransh/Documents/IE University/Year 5/Sem1/MLOps/cern_classifer/"
data_dir = 'data/raw/'

# Changing the working directory 
os.chdir(base_dir)


# +
# Importing the Data 
train_main = pd.read_csv(data_dir + 'train_main.csv')
train_collision = pd.read_csv(data_dir + 'train_collision.csv')

test_main = pd.read_csv(data_dir + 'test_main.csv')
test_collision = pd.read_csv(data_dir + 'test_collision.csv')

# +
print(f'The Main train dataset has a shape of {train_main.shape}')
print(f'The collision train dataset has a shape of {train_collision.shape}')

print(f'\nThe Main test dataset has a shape of {test_main.shape}')
print(f'The collision test dataset has a shape of {test_collision.shape}')
# -

train_main.head() 

train_collision.head()

test_main.head() 

test_collision.head()

# 1. Data Joining

# Merging the train dataset based on the id column
all_train = pd.merge(train_main, train_collision, on='id')
all_train.head()
all_train.to_csv("data/interim/train.csv", index=False)

# Merging the test dataset based on the id column
all_test = pd.merge(test_main, test_collision, on='id')
all_test.head()
all_test.to_csv("data/interim/test.csv", index=False)

# 2. Data Exploration

all_train.info()

all_train.describe()

# +
num_cols = [col for col in all_train.columns[1:] if all_train[col].dtype != 'O']
fig, axes = plt.subplots(4, 4, figsize=(12, 5))  # 4 rows, 4 columns
axes = axes.flatten()

for i, col_name in enumerate(num_cols):
    axes[i].hist(all_train[col_name])
    axes[i].set_title(f'Histogram of {col_name}')

plt.tight_layout()
plt.show()
# -

for col in num_cols:
    if all_train[col].nunique() <= 6:
        print(f'{col}: {all_train[col].nunique()}')

# 3. Data Cleaning

# Checking for missing values
(all_train.isnull().sum()/all_train.shape[0])*100

# +
missing_value_cols = ['MR', 'E1']

fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # 1 row, 2 columns

for i, col_name in enumerate(missing_value_cols):
    axes[i].hist(all_train[col_name])
    axes[i].set_title(f'Histogram of {col_name}')

plt.tight_layout()
plt.show()
# -

# Since both the variables `MR` & `E1` are skewed, a log transformation can be carried out to make them normal-like, followed by a mean imputation to fill missing values. 

# Log transformation 
all_train['MR'] = np.log10(all_train['MR'])
all_train['E1'] = np.log10(all_train['E1'])


# +
# Plot after transformation 
fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # 1 row, 2 columns

for i, col_name in enumerate(missing_value_cols):
    axes[i].hist(all_train[col_name])
    axes[i].set_title(f'Histogram of {col_name}')

plt.tight_layout()
plt.show()
# -

# Checking for missing values on the test dataset
(all_test.isnull().sum()/all_test.shape[0])*100

# +
# Imputing the Missing values with the mean value
imp_median = SimpleImputer(missing_values=np.nan, strategy='mean')
all_train[['E1', 'MR']] = imp_median.fit_transform(all_train[['E1', 'MR']])

all_test['MR'] = np.log10(all_test['MR'])
all_test['E1'] = np.log10(all_test['E1'])
all_test[['E1', 'MR']] = imp_median.transform(all_test[['E1', 'MR']])
# -

# 4. Feature Engineering

from sklearn.preprocessing import OneHotEncoder


# +
def create_buckets(data, cols, buckets = None): 
    if buckets is None: 
        for col in cols:
            data[col], buckets = pd.qcut(data[col], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'], retbins=True, precision=0)
        return data, buckets
    else:
        for col in cols:
            data[col] = pd.cut(data[col], bins=buckets, labels=['Q1', 'Q2', 'Q3', 'Q4'], include_lowest=True)
        return data

col_to_bucket = ['Lumi']
train_bucketed, buckets = create_buckets(all_train, col_to_bucket)
test_bucketed = create_buckets(all_test, col_to_bucket, buckets)


# -

def convert_to_categorical(data, cols, train = True): 
    if train:
         ohe_encoder = OneHotEncoder(sparse_output=False, drop='first')
         encoded_data = ohe_encoder.fit_transform(data[cols])
         dump(ohe_encoder, open('notebooks/encoder.pkl', 'wb'))
    else:
         ohe_encoder = load(open('notebooks/encoder.pkl', 'rb'))
         encoded_data = ohe_encoder.transform(data[cols])   
        
    column_names = ohe_encoder.get_feature_names_out(input_features=cols)
    encoded_df = pd.DataFrame(encoded_data, columns=column_names)
    data = data.drop(cols, axis=1)
    df = pd.concat([data, encoded_df], axis=1)
    
    return df


cols_to_encode = ['Run', 'nBJets', 'Lumi']
train_cat_encoded = convert_to_categorical(train_bucketed, cols_to_encode, train = True)
test_cat_encoded = convert_to_categorical(test_bucketed, cols_to_encode, train = False)

# Categorical Encoding for the train set 
factor = pd.factorize(train_cat_encoded["jets"])
train_cat_encoded["jets"] = factor[0]
defs = factor[1]

# 5. Model Building

# +
from sklearn.model_selection import train_test_split

X = train_cat_encoded.drop(["jets", "id"], axis=1)
y = train_cat_encoded['jets']
X_train, X_valid, y_train, y_valid = train_test_split(X,y, test_size=0.3, random_state=42)

# +
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
# -

clf = RandomForestClassifier(n_estimators = 500, criterion = "entropy")
clf.fit(X_train, y_train)
preds = clf.predict(X_valid)
accuracy_score(preds, y_valid)

# 6. Model Evaluation

# +
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score

print(classification_report(preds, y_valid))


# +
precision_macro = precision_score(preds, y_valid, average='macro')
recall_macro = recall_score(preds, y_valid, average='macro')
f1_macro = f1_score(preds, y_valid, average='macro')

precision_micro = precision_score(preds, y_valid, average='micro')
recall_micro = recall_score(preds, y_valid, average='micro')
f1_micro = f1_score(preds, y_valid, average='micro')

print('Macro-averaged precision:', precision_macro)
print('Macro-averaged recall:', recall_macro)
print('Macro-averaged F1-score:', f1_macro)

print('\nMicro-averaged precision:', precision_micro)
print('Micro-averaged recall:', recall_micro)
print('Micro-averaged F1-score:', f1_micro)
# -

# 7. Model Improvement 

# +
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris



# Define a grid of hyperparameters to search through
param_grid = {
    'n_estimators': [250, 500, 750, 1000],  # Number of trees in the forest
    'max_depth': [None, 10, 20, 30],  # Maximum depth of each tree
    'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split a node
    'min_samples_leaf': [1, 2, 4],  # Minimum number of samples required at each leaf node
}

# Create GridSearchCV object with the classifier and parameter grid
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, n_jobs=-1)

# Fit the GridSearchCV to your data
grid_search.fit(X_train, y_train)

# Print the best hyperparameters and corresponding accuracy score
best_params = grid_search.best_params_
best_score = grid_search.best_score_
print("Best Hyperparameters:", best_params)
print("Best Score (Accuracy):", best_score)

# -

preds = grid_search.predict(X_valid)
print(classification_report(preds, y_valid))

# +
import pickle
best_model = grid_search.best_estimator_

with open('models/best_model_1.pkl', 'wb') as model_file:
    pickle.dump(best_model, model_file)

# -

best_model = load(open('models/best_model_1.pkl', 'rb'))

# 8. Interpretability 

# +
# Feature Importance
import seaborn as sns

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': best_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Feature Importance for Random Forest Classifer')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

# +
import shap


explainer = shap.TreeExplainer(best_model, X_train)
shap_values = explainer.shap_values(X_valid)
# shap.initjs()
# -

shap.summary_plot(shap_values, X_valid)

import lime
import lime.lime_tabular

# +
predict_fn_rf = lambda x: best_model.predict_proba(x).astype(float)

explainer = lime.lime_tabular.LimeTabularExplainer(X_train,
                                                   training_labels = y_train.values,
                                                   feature_names = X.columns,
                                                   class_names=defs,
                                                   mode="classification")
# -

X_valid[3]

# High confidence negative prediction
exp = explainer.explain_instance(X_valid[3], predict_fn_rf)
exp.show_in_notebook()


