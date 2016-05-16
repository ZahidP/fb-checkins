import pandas as pd
import numpy as np
import sklearn as sklearn
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
# from sklearn.neural_network import MLPClassifier

#nnet = MLPClassifier(algorithm='l-bfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1).fit(train, trainy)


def explore(nrows):
    data = pd.read_csv('../data/sample.csv')
    data = data.iloc[0:nrows]
    holdout = data.iloc[nrows:nrows+5000]

    # train = data['']
    train = data.ix[:,data.columns != 'place_id']
    trainy = data.ix[:,data.columns == 'place_id'].place_id.ravel()

    holdoutx = holdout.ix[:,data.columns != 'place_id']
    holdouty = holdout.ix[:,data.columns == 'place_id'].place_id.ravel()


    # tuned_parameters = [
    #     {'algorithm':'l-bfgs', 'alpha':1e-5,
    #     'hidden_layer_sizes':(5, 2), 'random_state':1},
    #     {'algorithm':'l-bfgs', 'alpha':1e-5,
    #     'hidden_layer_sizes':(5, 2), 'random_state':3},
    #     {'algorithm':'l-bfgs', 'alpha':1e-3,
    #     'hidden_layer_sizes':(5, 2), 'random_state':1}
    # ]

    tuned_parameters = [
      {'n_estimators': [400], 'max_depth': [3,4], 'learning_rate': [0.1,0.08]},
      {'n_estimators': [500], 'max_depth': [3,4], 'learning_rate': [0.08,0.06]},
      {'n_estimators': [700], 'max_depth': [3,4], 'learning_rate': [0.04,0.02]},
      {'n_estimators': [800], 'max_depth': [3,4], 'learning_rate': [0.03,0.015]},
    ]
    scores = ['precision', 'recall']
    # for score in scores:
    #     print("# Tuning hyper-parameters for %s" % score)
    #     print()
    #     clf = GridSearchCV(GradientBoostingClassifier(), tuned_parameters, cv=3,
    #                        scoring='%s_weighted' % score).fit(train, trainy)
    #     print("Best parameters set found on development set:")
    #     print(clf.best_params_)
    #     print("Grid scores on development set:")
    #     for params, mean_score, scores in clf.grid_scores_:
    #         print("%0.3f (+/-%0.03f) for %r"
    #               % (mean_score, scores.std() * 2, params))
    #     print("Detailed classification report:")
    #     print("The model is trained on the full development set.")
    #     print("The scores are computed on the full evaluation set.")
    #     y_true, y_pred = holdouty, clf.predict(holdout)
    #     print(classification_report(y_true, y_pred))

    print('Starting RandomForestClassifier')
    clf3 = RandomForestClassifier(n_estimators=100, max_features='auto',verbose=3).fit(train, trainy)

    print(clf3.score(holdoutx, holdouty))
