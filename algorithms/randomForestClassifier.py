import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

"""**Random forest classifier:** \

Random forests or random decision forests is an ensemble learning method for classification, 
regression and other tasks that operates by constructing a multitude of decision trees at training time. 
For classification tasks, the output of the random forest is the class selected by most trees. 
For regression tasks, the mean or average prediction of the individual trees is returned.
Random decision forests correct for decision trees' habit of overfitting to their training set. 
Random forests generally outperform decision trees, but their accuracy is lower than gradient boosted trees. 
However, data characteristics can affect their performance.

"""
def randomForestClassifier(X_train, y_train, X_test, y_test):

    # Train a random forest classifier on the training set
    rfc = RandomForestClassifier(random_state=45)
    rfc.fit(X_train, y_train)

    # Make predictions on the testing set
    y_pred = rfc.predict(X_test)

    # Evaluate the performance of the model
    rfc_accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy:', rfc_accuracy)
    return rfc_accuracy
