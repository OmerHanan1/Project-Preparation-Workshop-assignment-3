import pandas as pd
import sqlite3
import copy
from colorama import Fore, Back, Style
import logging
import time

# region Data preprocessing

# Load the data from the SQLite database
conn = sqlite3.connect('database.sqlite')
match_data = pd.read_sql_query("SELECT * from match", conn)

# Create a new column 'home_team_win' and populate it with 1 or 0
# match_data['home_team_win'] = match_data.apply(lambda row: 1 if row['home_team_goal'] > row['away_team_goal'] else 0, axis=1)
match_data['home_team_win'] = match_data.apply(lambda row: 1 if row['home_team_goal'] > row['away_team_goal'] else 0 if row['home_team_goal'] == row['away_team_goal'] else -1, axis=1)

# Close the connection to the database
conn.close()

"""**Fetch Teams by names:** \

Fetch the teams by names from the database and return the team ids.
The User chooses the teams by names and the function returns the team ids.

"""
def fetchTeamIdByName(name):
    conn = sqlite3.connect('database.sqlite')
    team_data = pd.read_sql_query('SELECT id from team where team_long_name = "'+name+'"', conn)
    conn.close()

    return team_data['id'][0]

# Select only the relevant features-
features = ['home_team_api_id', 'away_team_api_id', 'home_team_win']

match_data = match_data[features]
print(match_data.columns.size)

# Drop rows with missing values
match_data.dropna(inplace=True)

# # Encode categorical features using one-hot encoding
# features_for_get_dummies = copy.deepcopy(features)
# features_for_get_dummies.remove('home_team_win')
# match_data = pd.get_dummies(match_data, columns=features_for_get_dummies)

# print(match_data.columns.size)
# print(match_data.head())

train_data = pd.get_dummies(match_data)
# print(train_data.columns.size)
# print(train_data.head())
# test_data = match_data # TODO: Update the values of the test data according to the UI. (!!!)

# # Add missing columns to test_data
# missing_cols = set(train_data.columns) - set(test_data.columns)
# for col in missing_cols:
#     test_data[col] = 0

##################################################
# TODO: Update the values of the test data according to the UI. (!!!)
##################################################

# Split the training and testing sets into X and y
X_train = train_data.drop(['home_team_win'], axis=1)
y_train = train_data['home_team_win']
# X_test = test_data.drop(['home_team_win'], axis=1)
# y_test = test_data['home_team_win']

def trainModel(model, X_train, y_train):
    # Set fit to maximum 40 seconds:
    start_time = time.time()

    # Fit the model
    model.fit(X_train, y_train)

    # Calculate the elapsed time
    elapsed_time = time.time() - start_time

    # Print the elapsed time
    print(Fore.YELLOW + "Elapsed time: " + str(elapsed_time) + " seconds" + Style.RESET_ALL)

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

from sklearn.ensemble import RandomForestClassifier

# Train a random forest classifier on the training set
rfc = RandomForestClassifier(random_state=5)

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

from sklearn.neural_network import MLPClassifier

# Train a MLP classifier on the training set
mlp = MLPClassifier(hidden_layer_sizes=(5,), activation='relu', solver='adam', max_iter=3)

print(Fore.YELLOW + "Start training the MLP model..." + Style.RESET_ALL)

trainModel(mlp, X_train, y_train)

print(Fore.GREEN + "Finished training the MLP model..." + Style.RESET_ALL)

"""**Decision tree classifier** \

A decision tree is a decision support hierarchical model that uses a - 
tree-like model of decisions and their possible consequences, 
including chance event outcomes, resource costs, and utility. 
It is one way to display an algorithm that only contains conditional control statements.

"""

from sklearn.tree import DecisionTreeClassifier

# Train the classifier on your data
dtc = DecisionTreeClassifier(max_depth=3)

print(Fore.YELLOW + "Start training the DTC model..." + Style.RESET_ALL)

trainModel(dtc, X_train, y_train)

print(Fore.GREEN + "Finished training the DTC model..." + Style.RESET_ALL)

####################################################################################################
    # TODO: Update this function. should use the trained model in order to predict the winner.
    #       Currently calculating the probability of team A winning in general, and not against team B.
    #       Should return the probability of team A winning against team B.
####################################################################################################


def rfClassifier(teamA, teamB, model= rfc):
    
    print(X_train.columns)
    print(X_train)

    teamA_id = fetchTeamIdByName(teamA)
    teamB_id = fetchTeamIdByName(teamB)

    print(teamA_id)
    print(teamB_id)

    test_data =pd.DataFrame(columns=X_train.columns)
    test_data.at[0, 'home_team_api_id'] = teamA_id
    test_data.at[0, 'away_team_api_id'] = teamB_id

    print(test_data)
    
    y_pred = model.predict(test_data)
    return(y_pred)



def mlpClassifier(teamA, teamB, model= mlp):
    teamA_df = pd.DataFrame(columns=X_train.columns)
    teamB_df = pd.DataFrame(columns=X_train.columns)

    # Add the team's features to the DataFrame
    for col in teamA_df.columns:
        teamA_df.at[0, col] = teamA[col]
        teamB_df.at[0, col] = teamB[col]

    # Make predictions on the team's features
    teamA_pred = model.predict_proba(teamA_df)

    # Return the probability of team A winning
    return teamA_pred[0][1]

def dtClassifier(teamA, teamB, model= dtc):
    teamA_df = pd.DataFrame(columns=X_train.columns)
    teamB_df = pd.DataFrame(columns=X_train.columns)

    # Add the team's features to the DataFrame
    for col in teamA_df.columns:
        teamA_df.at[0, col] = teamA[col]
        teamB_df.at[0, col] = teamB[col]

    # Make predictions on the team's features
    teamA_pred = model.predict_proba(teamA_df)

    # Return the probability of team A winning
    return teamA_pred[0][1]