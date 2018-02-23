###########################################################################
###########################################################################
## Kaggle -- Telstra Network Disruptions                                 ##
## Chris Steingass                                                       ##
###########################################################################
###########################################################################

#%%
# -------------------------------------------------------------------------
# IMPORT LIBRARIES, SET OPTIONS
# -------------------------------------------------------------------------
# System
import os
from os import listdir, makedirs, getcwd, remove
from os.path import isfile, join, abspath, exists, isdir

# Base
import pandas as pd, numpy as np, matplotlib.pyplot as plt

# Options
pd.options.display.max_rows = 1000
pd.options.display.max_columns = 1000
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# Path
path = getcwd() + '/Dropbox/_Code/'

#%%
# -------------------------------------------------------------------------
# READ DATA
# -------------------------------------------------------------------------
filenames = [
    'event_type.csv',
    'log_feature.csv',
    'resource_type.csv',
    'severity_type.csv',
    'test.csv',
    'train.csv'
    ]

for filename in filenames:
    dataset = pd.read_csv(path + 'Kaggle_Telstra/data/' + filename)
    globals()[filename.split('.')[0]] = dataset

    print(filename)
    dataset.head(3)
    print('')

#%%
# -------------------------------------------------------------------------
# SECTION 1/2 -- FEATURE SELECTION:
# Telstra provided seven data sets for this submission. Using these seven data sets, I wanted to create a training set that contains as much useful information as possible. Generally, this involved some combination of (1) directly merging sets by 'id', (2) creating frequency counts for each 'id', and (3) pivoting data sets to transform them from long, skinny sets to short wide sets.
# -------------------------------------------------------------------------

#%%
# Concatenate the provided training and test sets (ultimately, I want both my training and test sets to have the same features, so concatenating them now and performing all my operations on the concatenated set is simpler).
master = train.append(test)

# Change cell values from strings to integer values
# master.location = master.location.apply(lambda x: int(x.split(' ')[1]))
event_type.event_type = event_type.event_type.apply(lambda x: int(x.split(' ')[1]))
log_feature.log_feature = log_feature.log_feature.apply(lambda x: int(x.split(' ')[1]))
resource_type.resource_type = resource_type.resource_type.apply(lambda x: int(x.split(' ')[1]))
severity_type.severity_type = severity_type.severity_type.apply(lambda x: int(x.split(' ')[1]))

#%%
# SEVERITY_TYPE set: Merge the SEVERITY_TYPE set with the MASTER set by 'id'.
master = pd.merge(left=master, right=severity_type, how='inner', on='id')

#%%
# EVENT_TYPE set: Create a data set where each column is one 'event_type' and each row is a unique 'id'. Each cell should be the number of times the 'event_type' has occured for the 'id'. In addition, create a column that counts the number of events for each 'id'.

# Pivot
event_type_pivot = event_type.pivot_table(index='id', values='event_type', columns='event_type', aggfunc=len, fill_value=0)
# Rename
event_type_pivot.columns = ['event_type_' + str(x) for x in event_type_pivot.columns]
# Merge
master = pd.merge(left=master, right=event_type_pivot, left_on='id', right_index=True, how='inner')

# Create frequency column: count(events) for 'id'
event_freq = pd.DataFrame(event_type.groupby('id').count())
event_freq['id'] = event_freq.index
event_freq.columns = ['event_freq', 'id']
# Merge
master = pd.merge(left=master, right=event_freq, on='id', how='inner')

#%%
# LOG_FEATURE set: Create a data set where each column is one 'log_feature' and each row is a unique 'id'. Fill each cell using the 'volume' field of the original set.

# Pivot
log_feature_pivot = log_feature.pivot(index = 'id', columns = 'log_feature', values='volume')
# Rename columns
log_feature_pivot.columns = ['log_feature_' + str(x) for x in log_feature_pivot.columns]
# Fill zeroes (instead of NaN)
for col in log_feature_pivot.columns:
    log_feature_pivot[col] = log_feature_pivot[col].apply(lambda x: np.nan_to_num(x))
# Merge
master = pd.merge(left=master, right=log_feature_pivot, left_on='id', right_index=True, how='inner')

#%%
# RESOURCE_TYPE set: Create a data set where each column is one 'resource_type' and each row is a unique 'id'. Each cell should be the number of times the 'resource_type' has occurred for the 'id'.

# Pivot
resource_type_pivot = resource_type.pivot_table(index='id', values='resource_type', columns='resource_type', aggfunc=len, fill_value=0)
# Rename
resource_type_pivot.columns = ['resource_type_' + str(x) for x in resource_type_pivot.columns]
# Merge
master = pd.merge(left=master, right=resource_type_pivot, how='inner', left_on='id', right_index=True)

#%%

# MISC: Create frequency column: how often does every location appear in the dataset
location_freq = pd.DataFrame(master.groupby('location').count()['id'])
location_freq['location'] = location_freq.index
location_freq.columns = ['location_freq', 'location']
# Merge
master = pd.merge(left=master, right=location_freq, on='location', how='inner')

#%%
# One Hot Encoding of location column

encoded = pd.get_dummies(master, prefix=['location', 'severity_type'], prefix_sep='_', dummy_na=False, columns=['location', 'severity_type'], sparse=False, drop_first=False)


#%%
# -------------------------------------------------------------------------
# SECTION 2/2 -- MODELING:
# With the MASTER set, I fit two machine learning algorithms (Extreme Gradient Boosting and Random Forest), using SKLearn's GridSearchCV package to tune the respective hyperparameters. Using the VotingClassifier package, I ensembled the two models using a simple weighted soft-voting mechanism.
# -------------------------------------------------------------------------

# -------------------------------------------------------------------------
# IMPORT LIBRARIES, SET OPTIONS
# -------------------------------------------------------------------------
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

# Split master set into training and testing sets depending on whether 'fault_severity' is known. Split into X and y.
encoded.index = encoded.id
train_X = encoded[encoded.fault_severity.isnull() == False].loc[:, 'id':]
train_y = encoded[encoded.fault_severity.isnull() == False]['fault_severity']
test_X = encoded[encoded.fault_severity.isnull() == True].loc[:, 'id':]

# Set StratifiedKFold options
skfold = StratifiedKFold(n_splits=4, shuffle = True)

#%%
# FIRST MODEL: EXTREME GRADIENT BOOSTING CLASSIFIER (SKLEARN)
from xgboost import XGBClassifier
# Define hyperparameter grid
params = {
        'min_child_weight': [1,3,5],
        'subsample': [0.5,0.7,1],
        'colsample_bytree': [0.3,.7,1],
        'max_depth': [3,4,8],
        'learning_rate': list(np.arange(0.01, 0.1, 0.02)),
        'n_estimators': [1000]}

# Define XGBoost function
boost = XGBClassifier(n_estimators=1000, objective='multi:softprob', silent=True, nthread=4)

# Set GridSearchCV options
xgb_gridsearch = GridSearchCV(boost, param_grid=params, scoring='log_loss', n_jobs=4, cv=skfold.split(train_X,train_y), verbose=2)

# Fit
xgb_gridsearch.fit(train_X, train_y)

#%%
# SECOND MODEL: RANDOM FOREST CLASSIFIER (SKLEARN)
from sklearn.ensemble import RandomForestClassifier
# Define hyperparameter grid
params = {
        'n_estimators': [1000],
        'max_features': ['auto'],
        'max_depth': [50], #list(range(10,100,10)),
        'min_samples_split': [10],
        'min_samples_leaf': [1],
        'bootstrap': [0,1]
        }

# Define Random Forest function
trees = RandomForestClassifier()

# Set GridSearchCV options
rf_gridsearch = GridSearchCV(trees, param_grid=params, scoring='log_loss', n_jobs=4, cv=skfold.split(train_X,train_y), verbose=2)

# Fit
rf_gridsearch.fit(train_X, train_y)

#%%

# ENSEMBLE USING SKLEARN WEIGHTED VOTING
from sklearn.ensemble import VotingClassifier
ensemble = VotingClassifier(estimators=[('boost', xgb_gridsearch.best_estimator_), ('trees', rf_gridsearch.best_estimator_)], voting='soft', weights=[1,0])

ensemble = ensemble.fit(train_X, train_y)

# GENERATE SUBMISSION
submission = pd.DataFrame(ensemble.predict_proba(test_X))
# Attach 'id' from TEST set
submission['id'] = test.id
# Rename columns
submission.columns = ['predict_0', 'predict_1', 'predict_2', 'id']
# Reorder columns
submission = submission[['id', 'predict_0', 'predict_1', 'predict_2']]
# Write to .csv
submission.to_csv(path + 'bonus_submission.csv', index=None)



# -------------------------------------------------------------------------
# SOURCES:
# http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html for instructions on implementing SKLearn's VotingClassifier
# http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html for instructions on implementing SKLearn's RandomForestClassifier
# https://www.kaggle.com/tilii7/hyperparameter-grid-search-with-xgboost for instructions on hyperparameter tuning for XGBoost
# https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74 for information on model ensembling
# https://www.kaggle.com/giovannibruner/randomforest-with-gridsearchcv for more instructions on using GridSearchCV
# stackoverflow.com for bug hunting and coding issues
# crossvalidated.com for bug hunting and coding issues
# -------------------------------------------------------------------------
