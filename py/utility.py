import pandas as pd
import numpy as np
import time
import math
import operator
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import BernoulliRBM
from sklearn.neighbors import KNeighborsClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import scale, LabelEncoder
from sklearn.linear_model import SGDClassifier


if __name__ == "__main__":
    print("two.py is being run directly")
else:
    print("two.py is being imported into another module")


# named lambda function for pandas apply
def spreads(group):
    accuracy = group.accuracy
    x = group.x
    y = group.y
    [xmax, xmin] = [x.max(), x.min()]
    [ymax, ymin] = [y.max(), y.min()]
    [accmax, accmin] = [accuracy.max(), accuracy.min()]
    xspread = xmax - xmin
    yspread = ymax - ymin
    accspread = accmax - accmin
    return pd.Series(dict(xspread=xspread,
                          xmax=xmax,
                          ymax=ymax,
                          yspread=yspread,
                          accmax=accmax,
                          acc=accspread))


# creates grid for x,y data
# @param data: Pandas dataframe
# @param bins: int number of horizontal/vertical boxes for grid ()
# @param sub_dict:
def grid_generate(data, bins, sub_dict, train, qxqy):
    grid = []
    grid_dict = {}
    for i in range(bins):
        grid.append([])
    # Divide bins up
    step = 1/bins
    #data.place_id = data.place_id.astype('str')
    alldata = data.values.tolist()
    for i in range(0,len(alldata)):
        grid_dict = to_dict(alldata[i], bins, grid_dict, sub_dict, train, qxqy)
    return grid_dict


# creates a dictionary that returns the following format
# (x,y): [ {place_id: [[acc,day,hr],[acc,day,hr]] },
#           { place_id: [acc,day,hr] }]


def convert_time(df):
    minutes = df['time']
    hours = minutes/60
    days = hours/24
    weeks = days/7
    months = days/30
    day = days % 7
    hr = hours % 24
    month = months % 12
    day_of_year = days % 365
    df['day'] = day.astype('int')
    df['hr'] = hr.astype('int')
    df['month'] = month.astype('int')
    df['day_of_year'] = day_of_year
    return df


def to_dict(row,bins,grid_dict,sub_dict,train, qxqy):
    row_id, x, y, accuracy, time = row[0], row[1], row[2], row[3], row[4]
    if train:
        place_id = row[5]
    else:
        place_id = 0
    qx = qxqy[0]         # quadrant
    qy = qxqy[1]
    qmx = qx * 5    # quadrant multiplier to scale
    qmy = qy * 5    # quadrant multiplier to scale
    minutes = time
    hours = minutes/60
    days = hours/24
    months = days/30
    weeks = days/7
    month = months % 12
    days %= 7
    day = int(days)
    hr = divmod(hours, 24)[1]
    month = int(month)
    day_of_year = divmod(days, 365)[1]
    day = math.floor(day)
    hr = math.floor(hr)
    eps = 0.00001
    x, y, accuracy = float(x), float(y), int(accuracy)
    x_d_y = x / (y + eps)
    x_t_y = x * y
    sint = math.sin(minutes*(np.pi/60))
    X = [row_id, x, y, accuracy, day, hr, month, day_of_year, x_d_y, x_t_y, sint]


    if sub_dict:
        other_data = {place_id: [X]}
    x = math.floor(((x-qmx)*bins)/5)
    y = math.floor(((y-qmy)*bins)/5)
    # if we already have something at this grid location
    if (x, y) in grid_dict:
        # if we want to group by place_id keys
        if sub_dict:
            print('sub_dict')
            if place_id in grid_dict[(x, y)]:
                if train:
                    grid_dict[(x, y)][place_id].append(X)
            else:
                if train:
                    grid_dict[(x,y)][place_id] = [X]
        # otherwise just append to list
        else:
            if train:
                X.insert(0, place_id)
            grid_dict[(x, y)].append(X)
    # if there is nothing at this grid location we create a new dictionary
    else:
        if sub_dict:
            grid_dict[(x, y)] = other_data
        else:
            if train:
                X.insert(0,place_id)
            grid_dict[(x, y)] = [X]
    return grid_dict


# reduces each key/val pair by
# summing all entries
def reducelists(vals: list):
    print('reducelists')
    a = len(vals)
    sums = [sum(i) for i in zip(*vals)]
    avgs = [x/a for x in sums]
    avgs = avgs.insert(a, 0)
    return avgs


# reduces the grids dictionaries
def reduceit(gdict):
    print('reduceit')
    for key in gdict:
        for key2 in gdict[key]:
            gdict[key][key2] = reducelists(gdict[key][key2], key2)
    return gdict


def format_df(df_train, df_test):
    # start = time.time()
    df_train.columns = ['place_id','row_id', 'x', 'y', 'accuracy', 'day', 'hr', 'month', 'day_of_year', 'x_d_y', 'x_t_y', 'sint']
    df_train.place_id = df_train.place_id.apply(float)
    df_train.place_id = df_train.place_id.apply(int)
    # print('Number of unique place_ids: ' + str(len(df_train.place_id.unique())))
    # print('====================')
    df_test.columns = ['row_id', 'x', 'y', 'accuracy', 'day', 'hr', 'month', 'day_of_year', 'x_d_y', 'x_t_y', 'sint']
    df_train.row_id = df_train.row_id.apply(float).apply(int).apply(str)
    df_train.day, df_test.day = df_train.day.astype('int'), df_test.day.astype('int')
    df_train.hr, df_test.hr = df_train.hr.astype('int'), df_test.hr.astype('int')
    df_train.month, df_test.month = df_train.month.astype('int'), df_test.month.astype('int')
    df_test.row_id = df_test.row_id.apply(float).apply(int).apply(str)
    # print('Format df: ' + str(time.time() - start) + 'seconds')
    return df_train, df_test


# Splits dataframes to X and y lists
def split_dfs(df_holdin, df_holdout, df_test):
    x = df_holdin.ix[:,[not x for x in df_holdin.columns.isin(['place_id', 'row_id'])]]
    x_holdout = df_holdout.ix[:,[not x for x in df_holdout.columns.isin(['place_id', 'row_id'])]]
    x_test = df_test.ix[:,[not x for x in df_test.columns.isin(['place_id', 'row_id'])]]
    # le1 = LabelEncoder()
    # le2 = LabelEncoder()
    # y = le.fit_transform(df_holdin.place_id.values)
    y = df_holdin.place_id.ravel()
    y_holdout = df_holdout.place_id.ravel()
    return x, x_holdout, x_test, y, y_holdout


def grid_search(train, trainy, holdout, holdouty, model):
    model_dict = {
        'rf': {
            'params': [{'n_estimators': [125], 'max_depth': [9, 11], 'criterion': ['gini', 'entropy']}],
            'model': RandomForestClassifier()
        },
        'knn': {
            'params': [{'n_neighbors': [10, 12], 'weights': ['distance'], 'metric': ['manhattan']}],
            'model': KNeighborsClassifier()
        },
        'sgd': {
            'params': [{'loss': ['modified_huber'], 'n_iter': [15, 25, 30], 'random_state': [0, 1]}],
            'model': SGDClassifier()
        }
    }
    best_models = []
    scores = ['accuracy']
    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()
        clf = GridSearchCV(estimator=model_dict[model]['model'],
                           param_grid=model_dict[model]['params'],
                           cv=2,
                           scoring=score,
                           n_jobs=2,
                           verbose=1).fit(train, trainy)
        print("Best parameters set found on development set:")
        print(clf.best_params_)
        print("Grid scores on development set:")
        for params, mean_score, scores in clf.grid_scores_:
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean_score, scores.std() * 2, params))
        print("Detailed classification report:")
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        # print()
        # y_true, y_pred = holdouty, clf.predict(holdout)
        # print(classification_report(y_true, y_pred))
        # print()
    return clf.best_estimator_


def sort_predictions(pred_probs, clf3):
    all_top = []
    for probs in range(0, len(pred_probs)):
        class_probs = dict(zip(clf3.classes_, pred_probs[probs]))
        class_dict = sorted(class_probs.items(), key=operator.itemgetter(1))
        class_dict.reverse()
        top_picks = class_dict[0:3]
        top_picks = [x[0] for x in top_picks]
        all_top.append(top_picks)
    return all_top


def fill_in(df_here, alist):
    for i in range(0,len(alist)):
        f = alist[i]
        a = pd.DataFrame([[f, 2000]])
        a.columns = ['row_id','location']
    return df_here


# Makes sure there are no missing entries in final predictions dataframe.
def fill_in_df(test, results):
    """ Makes sure there are no missing entries in final predictions dataframe.
        @param test Dataframe of test data
        @param results Dataframe of predictions
        returns {results1} predictions dataframe with row count equal to test1 row count
    """
    a = results.row_id.ravel()
    b = test.row_id.ravel()
    print(len(a))
    print(len(b))
    diff = set(b) - set(a)
    alist = list(diff)
    for i in range(0, len(alist)):
        f = alist[i]
        a = pd.DataFrame([[f, '8129178888 8129178888 8129178888']])
        a.columns = ['row_id', 'place_id']
        results = results.append(a)
    return results


def subset(fname, chunksize, nlines):
    nchunk = 0
    for df in pd.read_csv('../../train.csv', chunksize=chunksize):
        # df = pd.read_csv('../../train.csv', chunksize=chunksize, skiprows=i)
        if nchunk == 0:
            mode = 'w'
            header = 'infer'
        else:
            mode = 'a'
            header = None
        top_half = df[df.y < 5]
        bottom_half = df[df.y >= 5]
        # top left and top right
        subset = top_half[top_half.x < 5]
        subset.to_csv('../data/grid/full/topleft_' + fname + '.csv', index=False, mode=mode, header=header)
        subset = top_half[top_half.x >= 5]
        subset.to_csv('../data/grid/full/topright_' + fname + '.csv', index=False, mode=mode, header=header)
        # bottom left and bottom right
        subset = bottom_half[bottom_half.x < 5]
        subset.to_csv('../data/grid/full/bottomleft_' + fname + '.csv', index=False, mode=mode, header=header)
        subset = bottom_half[bottom_half.x >= 5]
        subset.to_csv('../data/grid/full/bottomright_' + fname + '.csv', index=False, mode=mode, header=header)
        nchunk += 1
    return subset


def merge_predictions(predictions, prob_a):
    """ Merges top 3 predictions in returned dataframe predictions column, space separated.
        @param predictions list of lists of [row_id, location]
        @param prob_a list of lists of [prob1, prob2, prob3]
        returns {df_final} DataFrame with columns [row_id predictions]
    """
    pr1 = predictions[0]
    pra1 = prob_a[0]
    df_temp1, df_temp2 = pd.DataFrame(pr1), pd.DataFrame(pra1)
    # shouldn't location be prediction?
    df_temp1.columns, df_temp2.columns = ['row_id'], ['prb1', 'prb2', 'prb3']
    df_final = df_temp1.merge(df_temp2, left_index=True, right_index=True)
    df_final = df_final.astype(str)
    df_final['place_id'] = df_final['prb1'] + ' ' + df_final['prb2'] + ' ' + df_final['prb3']
    del df_final['prb1']
    del df_final['prb2']
    del df_final['prb3']
    for j in range(1, len(predictions)):
        df_temp1 = pd.DataFrame(predictions[j])
        df_temp2 = pd.DataFrame(prob_a[j])
        df_temp1.columns = ['row_id']
        df_temp2.columns = ['prb1', 'prb2', 'prb3']
        if j % 50 == 0:
            print('Lengths: temp1, temp2')
            print(len(df_temp1))
            print(len(df_temp2))
            print(j)
        df_temp = df_temp1.merge(df_temp2, left_index=True, right_index=True)
        df_temp = df_temp.astype(str)
        df_temp['place_id'] = df_temp['prb1'] + ' ' + df_temp['prb2'] + ' ' + df_temp['prb3']
        del df_temp['prb1']
        del df_temp['prb2']
        del df_temp['prb3']
        df_final = df_final.append(df_temp)
    return df_final
