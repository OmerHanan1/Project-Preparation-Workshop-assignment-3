from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import sqlite3
from colorama import Fore, Back, Style
import time
import copy

# region Data preprocessing
match_data = pd.read_csv("match_data_with_features_only.csv")
print(match_data)

"""**Fetch Teams by names:** \

Fetch the teams by names from the database and return the team ids.
The User chooses the teams by names and the function returns the team ids.

"""


def fetchTeamIdByName(name):
    conn = sqlite3.connect('database.sqlite')
    team_data = pd.read_sql_query(
        f"SELECT id from team where team_long_name = '{name}'", conn)
    conn.close()

    return team_data['id'][0]


# Select only the relevant features-
features = ['date', 'home_team_api_id', 'away_team_api_id', 'home_team_name', 'away_team_name',
            'home_team_win', 'possession', 'shoton', 'shotoff', 'B365H', 'B365D', 'B365A', 'BWH', 'BWA']

original_data = match_data
match_data = match_data[features]
match_data['date'] = pd.to_datetime(match_data['date'])

# Drop rows with missing values
match_data.dropna(inplace=True)
original_match_data = copy.deepcopy(match_data)

train_data = pd.get_dummies(match_data)

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
    print(Fore.YELLOW + "Elapsed time: " +
          str(elapsed_time) + " seconds" + Style.RESET_ALL)

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
    20,), activation='relu', solver='adam', max_iter=500)

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
dtc = DecisionTreeClassifier(max_depth=45)

print(Fore.YELLOW + "Start training the DTC model..." + Style.RESET_ALL)

trainModel(dtc, X_train, y_train)

print(Fore.GREEN + "Finished training the DTC model..." + Style.RESET_ALL)


def prediction(teamA, teamB, model):
    if model == 'RFC':
        model = rfc
    elif model == 'MLP':
        model = mlp
    elif model == 'DTC':
        model = dtc
    else:
        print('Invalid model')
        return

    teamA_id = fetchTeamIdByName(teamA)
    teamB_id = fetchTeamIdByName(teamB)

    test_data = pd.DataFrame(columns=X_train.columns)
    test_data.at[0, 'home_team_api_id'] = teamA_id
    test_data.at[0, 'away_team_api_id'] = teamB_id

    y_pred = model.predict(test_data)
    return (y_pred)
