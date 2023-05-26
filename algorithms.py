from tkinter import messagebox
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from colorama import Fore, Style
import time
import copy

# region Data preprocessing
match_data = pd.read_csv("match_data_with_features_only.csv")

# Select only the relevant features-
features = ['date', 'home_team_api_id', 'away_team_api_id', 'home_team_name', 'away_team_name',
            'home_team_win', 'B365H', 'B365D', 'B365A', 'BWH', 'BWA']
match_data = match_data[features]

# Convert the date column to datetime format
match_data['date'] = pd.to_datetime(match_data['date'])

# original data
original_match_data = copy.deepcopy(match_data)
original_train_data = original_match_data[(original_match_data['date'].dt.year < 2015)
                                          | (original_match_data['date'].dt.year > 2016)]
original_test_data = original_match_data[(original_match_data['date'].dt.year == 2015)
                                         | (original_match_data['date'].dt.year == 2016)]

# Encode categorical features using one-hot encoding
features_for_get_dummies = copy.deepcopy(features)
features_for_get_dummies.remove('date')
features_for_get_dummies.remove('home_team_win')
match_data = pd.get_dummies(match_data, columns=features_for_get_dummies)

# Split the data into training and testing sets based on the date
train_data = match_data[(match_data['date'].dt.year < 2015)
                        | (match_data['date'].dt.year > 2016)]
test_data = match_data[(match_data['date'].dt.year == 2015)
                       | (match_data['date'].dt.year == 2016)]

# Add missing columns to test_data
missing_cols = set(train_data.columns) - set(test_data.columns)
for col in missing_cols:
    test_data[col] = 0

# Split the training and testing sets into X and y
X_train = train_data.drop(['home_team_win', 'date'], axis=1)
y_train = train_data['home_team_win']
X_test = test_data.drop(['home_team_win', 'date'], axis=1)
y_test = test_data['home_team_win']


def trainModel(model, X_train, y_train):
    # Set fit to maximum 40 seconds:
    start_time = time.time()

    # Fit the model
    model.fit(X_train, y_train)

    # Calculate the elapsed time
    elapsed_time = time.time() - start_time

    # Print the elapsed time
    print(Fore.YELLOW + "Elapsed time: " +
          str(elapsed_time) + " seconds" + Style.RESET_ALL)
    model

    return model

# endregion


"""**Random forest classifier:** \

Random forests or random decision forests is an ensemble learning method for classification, 
regression and other tasks that operates by constructing a multitude of decision trees at training time. 
For classification tasks, the output of the random forest is the class selected by most trees. 
For regression tasks, the mean or average prediction of the individual trees is returned.
Random decision forests correct for decision trees' habit of overfitting to their training set. 
Random forests generally outperform decision trees, but their accuracy is lower than gradient boosted trees. 
However, data characteristics can affect their performance.

"""
# Train a random forest classifier on the training set
rfc = RandomForestClassifier(random_state=50)

print(Fore.YELLOW + "Start training the RFC model..." + Style.RESET_ALL)

# Fit the model with a loop that checks the elapsed time
trainModel(rfc, X_train, y_train)

print(Fore.GREEN + "Finished training the RFC model..." + Style.RESET_ALL)


"""**Multilayer perceptron classifier:** \

A multilayer perceptron (MLP) is a fully connected class of feedforward artificial neural network (ANN). 
The term MLP is used ambiguously, sometimes loosely to mean any feedforward ANN, 
sometimes strictly to refer to networks composed of multiple layers of perceptrons (with threshold activation). 
Multilayer perceptrons are sometimes colloquially referred to as "vanilla" neural networks, 
especially when they have a single hidden layer.

"""
# Train a MLP classifier on the training set
mlp = MLPClassifier(hidden_layer_sizes=(
    20,), activation='relu', solver='adam', max_iter=2)

print(Fore.YELLOW + "Start training the MLP model..." + Style.RESET_ALL)

trainModel(mlp, X_train, y_train)

print(Fore.GREEN + "Finished training the MLP model..." + Style.RESET_ALL)


"""**Decision tree classifier** \

A decision tree is a decision support hierarchical model that uses a - 
tree-like model of decisions and their possible consequences, 
including chance event outcomes, resource costs, and utility. 
It is one way to display an algorithm that only contains conditional control statements.

"""
# Train the classifier on your data
dtc = DecisionTreeClassifier(max_depth=2)

print(Fore.YELLOW + "Start training the DTC model..." + Style.RESET_ALL)

trainModel(dtc, X_train, y_train)

print(Fore.GREEN + "Finished training the DTC model..." + Style.RESET_ALL)


def get_teams_with_match_row(home_team_name):
    """
    returns all teams that has a match with 'team_name'
    """
    matching_teams = original_test_data[original_test_data['home_team_name'].str.contains(
        home_team_name)]
    teams = matching_teams['away_team_name'].tolist()
    team_list = list(set(teams))
    team_list.sort()
    return team_list


def get_all_test_teams():
    """
    returns all teams in test data
    """
    # away_teams = original_test_data['away_team_name'].unique()
    home_teams = original_test_data['home_team_name'].unique()
    # distinct_teams = np.union1d(away_teams, home_teams)
    team_list = list(home_teams)  # list(distinct_teams)
    team_list.sort()
    return team_list


def get_all_dates_of_matches(team_1_name, team_2_name):
    """
    returns all dates of matches
    """
    try:
        # extract all rows
        matching_rows = original_test_data[(original_test_data['home_team_name'] == team_1_name) & (
            original_test_data['away_team_name'] == team_2_name)]

        # extract all dates from rows
        matching_dates = matching_rows['date']
        if type(matching_dates) == list:
            distinct_dates = list(set(matching_dates))
        else:
            distinct_dates = matching_dates
        distinct_dates = np.datetime_as_string(distinct_dates, unit='D')
        return distinct_dates
    except:
        return []


def prediction(team1, team2, match_date, algorithm):
    if algorithm == 'RFC':
        model = rfc
    elif algorithm == 'MLP':
        model = mlp
    elif algorithm == 'DTC':
        model = dtc
    else:
        print('Invalid model')
        return "Invalid model"

    try:
        to_perdict, true_label = getRowFromData(team1, team2, match_date)
    except Exception as e:
        return Exception(f"Wrong Input team1={team1} team2={team2} didn't had a game")

    if type(to_perdict) == type(None):
        print('No data found')
        return None
    to_perdict = to_perdict.drop(columns=['date', 'home_team_win'])
    y_predict = model.predict(to_perdict)

    print('Predicted label: ', y_predict)
    print('True label: ', true_label)

    if y_predict == 1:
        y_predict = team1 + ' wins'
    elif y_predict == 0:
        y_predict = 'Draw'
    else:
        y_predict = team2 + ' wins'

    # Make the true label a dataframe
    true_label = pd.DataFrame(true_label)
    true_label = true_label.iloc[0, 0]

    if true_label == 1:
        true_label = team1 + ' wins'
    elif true_label == 0:
        true_label = 'Draw'
    else:
        true_label = team2 + ' wins'

    return y_predict, true_label


def getRowFromData(team1, team2, match_date):
    # print(match_date)
    clean_date_format = pd.to_datetime(match_date.replace(
        "'", "").replace("[", "").replace("]", ""))
    # print(clean_date_format)

    condition = (
        (match_data['home_team_name_' + team1] == 1) &
        (match_data['away_team_name_' + team2] == 1) &
        (match_data['date'] == clean_date_format)
    )

    filtered_data = match_data[condition]
    if filtered_data.empty:
        return None, None

    return filtered_data, filtered_data['home_team_win']
