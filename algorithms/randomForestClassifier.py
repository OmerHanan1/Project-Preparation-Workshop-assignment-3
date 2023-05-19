import pandas as pd
import sqlite3
import copy
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


"""**Random forest classifier:** \

Random forests or random decision forests is an ensemble learning method for classification, regression and other tasks that operates by constructing a multitude of decision trees at training time. For classification tasks, the output of the random forest is the class selected by most trees. For regression tasks, the mean or average prediction of the individual trees is returned.Random decision forests correct for decision trees' habit of overfitting to their training set. Random forests generally outperform decision trees, but their accuracy is lower than gradient boosted trees. However, data characteristics can affect their performance.
"""

# Load the data from the SQLite database
conn = sqlite3.connect('database.sqlite')
match_data = pd.read_sql_query("SELECT * from match", conn)

# Create a new column 'home_team_win' and populate it with 1 or 0
# match_data['home_team_win'] = match_data.apply(lambda row: 1 if row['home_team_goal'] > row['away_team_goal'] else 0, axis=1)
match_data['home_team_win'] = match_data.apply(lambda row: 1 if row['home_team_goal'] > row['away_team_goal'] else 0 if row['home_team_goal'] == row['away_team_goal'] else -1, axis=1)

# Close the connection to the database
conn.close()


# Select only the relevant features

features = ['date', 'league_id', 'home_team_api_id', 'away_team_api_id', 'home_team_win', 'B365H', 'B365D', 'B365A', 'BWH', 'BWA', 'possession']
match_data = match_data[features]

# Drop rows with missing values
match_data.dropna(inplace=True)

# Convert the date column to datetime format
match_data['date'] = pd.to_datetime(match_data['date'])

# Encode categorical features using one-hot encoding
features_for_get_dummies = copy.deepcopy(features)
features_for_get_dummies.remove('date')
features_for_get_dummies.remove('home_team_win')
match_data = pd.get_dummies(match_data, columns=features_for_get_dummies)


# Split the data into training and testing sets based on the date
train_data = match_data[(match_data['date'].dt.year < 2015) | (match_data['date'].dt.year > 2016)]
test_data = match_data[(match_data['date'].dt.year == 2015) | (match_data['date'].dt.year == 2016)]

# Add missing columns to test_data
missing_cols = set(train_data.columns) - set(test_data.columns)
for col in missing_cols:
    test_data[col] = 0

# Split the training and testing sets into X and y
X_train = train_data.drop(['home_team_win', 'date'], axis=1)
y_train = train_data['home_team_win']
X_test = test_data.drop(['home_team_win', 'date'], axis=1)
y_test = test_data['home_team_win']

# Train a random forest classifier on the training set
rfc = RandomForestClassifier(random_state=45)
rfc.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = rfc.predict(X_test)

# Evaluate the performance of the model
rfc_accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', rfc_accuracy)
