import pandas as pd
import sqlite3
from colorama import Fore, Back, Style
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

# Drop rows with missing values
match_data.dropna(inplace=True)
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
rfc = RandomForestClassifier(random_state=18)

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
mlp = MLPClassifier(hidden_layer_sizes=(25,), activation='relu', solver='adam', max_iter=100)

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

    test_data =pd.DataFrame(columns=X_train.columns)
    test_data.at[0, 'home_team_api_id'] = teamA_id
    test_data.at[0, 'away_team_api_id'] = teamB_id
    
    y_pred = model.predict(test_data)
    return(y_pred)