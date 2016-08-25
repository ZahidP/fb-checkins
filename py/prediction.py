import pandas as pd
import numpy as np
import time
import utility as ut
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import scale, LabelEncoder, PolynomialFeatures, scale


if __name__ == '__main__':
    print('Main')


def train_and_predict(bins: int, qx: int, qy: int, grid_location: str,
                      n_est: int, depth: int, sample: bool):
    print('Starting:')
    if sample:
        train_file = '../data/grid/full/sample_' + grid_location + '_.csv'
    else:
        train_file = '../data/grid/full/' + grid_location + '_.csv'
    test_file = '../data/grid/' + grid_location + '_test.csv'
    data = pd.read_csv(train_file)
    test = pd.read_csv(test_file)
    print('Number of training records: ' + str(len(data)))
    if sample:
        test = test.iloc[0:50000]

    data = data.sort_values('place_id').copy()
    keep = data.place_id.value_counts() > 19
    keep = keep[keep]
    data = data[data['place_id'].isin(keep.index.tolist())]

    print(data.place_id.value_counts()[0:5])
    print('============')
    print('Number of training records (subset): ' + str(len(data)))
    print(data[0:2])
    print('Number of test records: ' + str(len(test)))
    print(test[0:2])
    print('...')
    print('Generate Dictionaries')
    sub_dict = False
    qxqy = [qx, qy]

    start = time.time()
    grid_d_train = ut.grid_generate(data, bins, sub_dict, True, qxqy)
    grid_d_test = ut.grid_generate(test, bins, sub_dict, False, qxqy)

    # Diagnostic Variables
    fit_scores = []  # scores for each grid location
    grid_populations = []  # size of particular grid location's population
    avg_vals = []
    a_few_models = []
    misclassified = []
    corrects = []
    prob_a_list = []
    if sample:
        bins = 4

    print('Time: ' + str(time.time() - start) + ' seconds')
    print('Generate Models and Make Predictions:')
    start = time.time()
    preds = []
    ff = 0
    # traverse the bins "rows"
    for i in range(0, bins):
        # traverse the bins "columns"
        for j in range(0, bins):
            loop_start = time.time()
            ff += 1
            # otherwise grab the grid_d and create lists that way (training)
            df_train = pd.DataFrame(grid_d_train[(i, j)])
            df_test = pd.DataFrame(grid_d_test[(i, j)])

            train_grid_pop = len(grid_d_train[(i, j)])
            grid_populations.append(train_grid_pop)
            # format dataframes, number to strings, colnames etc
            df_train, df_test = ut.format_df(df_train, df_test)
            print('Classifier Number: ' + str(ff))

            if ff % 100 == 0 or ff < 4:
                print('\n' + '==================================')
                print('Classifier Number: ' + str(ff))
                print('====================')
                print('Number of rows in training grid location: ' + str(train_grid_pop))
                print(df_train[0:2])

            # create holdout dataframes
            holdout_size = 0.025
            if sample:
                holdout_size = 0.15

            keep = df_train.place_id.value_counts() > 3
            keep = keep[keep]
            df_train = df_train[df_train['place_id'].isin(keep.index.tolist())]
            df_holdin, df_holdout = train_test_split(df_train, test_size=holdout_size)

            # split holdouts, X, y
            x, x_holdout, x_test, y, y_holdout = ut.split_dfs(df_holdin, df_holdout, df_test)
            if ff % 100 == 0 or ff < 4:
                print('Number of ids in holdout not in training set: ' + str(len(set(y_holdout) - set(y))))
                print('Unique values in train (hold_in): ' + str(len(df_holdin.place_id.unique())))
                print('Unique values in train: ' + str(len(df_train.place_id.unique())))
                print('Unique values in holdout: ' + str(len(df_holdout.place_id.unique())))

            keep = df_holdout.place_id.value_counts() > 2
            keep = keep[keep]
            df_holdout = df_holdout[df_holdout['place_id'].isin(keep.index.tolist())]

            # clf = RandomForestClassifier(n_estimators=n_est,
            #                              max_features='auto',
            #                              criterion='entropy',
            #                              max_depth=depth,
            #                              n_jobs=2,
            #                              verbose=0).fit(x, y)

            # [x_test, x_holdout, clf] = k_neighbors(x, y, x_holdout, y_holdout, x_test)
            # [x_test, x_holdout, clf2] = sgd_prediction(x, y, x_holdout, y_holdout, x_test)
            [x_test, x_holdout, clf] = r_forest(x, y, x_holdout, y_holdout, x_test)
            # [x_test, x_holdout, clf3] = try_all(x, y, x_holdout, y_holdout, x_test)

            if ff < 4:
                a_few_models.append(clf)

            avgs = [np.average(df_holdout.x), np.average(df_holdout.y), np.average(df_holdout.accuracy)]
            avg_vals.append(avgs)

            # use the most recently made predictor to predict the corresponding test data
            # start_mini = time.time()
            pred_list = clf.predict(x_test)

            # split this into a function
            pred_probs = clf.predict_proba(x_test)
            # sort the predictions
            all_top3 = ut.sort_predictions(pred_probs=pred_probs, clf3=clf)
            # print('Predict and sort: ' + str(time.time() - start_mini) + 'seconds')
            pred_rows = df_test.ix[:, df_test.columns == 'row_id'].values.ravel()
            predicted_holdouts = clf.predict(x_holdout)
            holdout_rows = df_holdout.ix[:, df_holdout.columns == 'row_id'].values.ravel()
            # append these predictions to the dictionary
            # start_mini = time.time()
            pred = list(zip(pred_rows))
            # pred1 = list(zip(holdout_rows, fits_holdout))
            # ys = list(zip(holdout_rows, y_holdout))
            # yspred1 = zip(ys, pred1)
            # yspred2 = zip(ys, pred1)
            # misclass = [i for i, j in yspred1 if i[1] != j[1]]
            # correct = [i for i, j in yspred2 if i[1] == j[1]]
            correct = 0
            misclass = 0
            misclassified.append(misclass)
            corrects.append(correct)
            # print('Misclassified: ' + str(time.time() - start_mini) + 'seconds')
            preds.append(pred)
            prob_a_list.append(all_top3)
            if ff % 100 == 0 or ff < 3:
                print('Predictions, rows, test data')
                # print(pred_list[0:2])
                print(pred_rows[0:2])
                print(df_test[0:2])
                print('finding misclassified' + '\n' + '-----------------------')
                # print(pred[0:2])
                # print(pd.DataFrame(pred[0:2]))
                print('Pred length:')
                print(len(prob_a_list))
                print('Loop Time: ' + str(time.time() - loop_start) + ' seconds')
    print('Time: ' + str(time.time() - start) + ' seconds')
    retval = [preds, prob_a_list, fit_scores, grid_populations, a_few_models, corrects, misclassified]
    return retval


def r_forest(x, y, x_holdout, y_holdout, x_test):
    poly = PolynomialFeatures(2)
    X = scale(x)
    X = poly.fit_transform(X)
    clf = ut.grid_search(train=X, trainy=y, holdout=x_holdout, holdouty=y_holdout, model='rf')
    clf.fit(X, y)
    x_holdout = poly.fit_transform(x_holdout)
    x_test = poly.fit_transform(x_test)
    return [x_test, x_holdout, clf]


def sgd_prediction(x, y, x_holdout, y_holdout, x_test):
    # le = LabelEncoder()
    # y = le.fit_transform(y)
    poly = PolynomialFeatures(3)
    X = scale(x)
    X = poly.fit_transform(X)
    # Training Classifier
    print(y[0:5])
    # clf = SGDClassifier(loss='modified_huber', n_iter=5, random_state=0, n_jobs=-1, verbose=0)
    clf = ut.grid_search(train=X, trainy=y, holdout=x_holdout, holdouty=y_holdout, model='sgd')
    # clf.fit(X, y)
    x_holdout = poly.fit_transform(x_holdout)
    x_test = poly.fit_transform(x_test)
    return [x_test, x_holdout, clf]


def k_neighbors(x, y, x_holdout, y_holdout, x_test):
    # poly = PolynomialFeatures(2)
    X = scale(x)
    fw = [1.5, 2.5, 1.5, 1, 1.25, 1, 1/25]
    X[:, 0] *= 4.75
    X[:, 1] *= 6
    X[:, 3] *= 2
    X[:, 5] *= 1.5
    X[:, 7] *= 2.5
    # X = poly.fit_transform(X)
    # clf = KNeighborsClassifier(n_neighbors=20, weights='distance', metric='manhattan').fit(X, y)
    # x_holdout = poly.fit_transform(x_holdout)
    # x_test = poly.fit_transform(x_test)
    clf = ut.grid_search(train=X, trainy=y, holdout=x_holdout, holdouty=y_holdout, model='knn')

    return [x_test, x_holdout, clf]
