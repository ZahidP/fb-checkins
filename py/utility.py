import pandas as pd
import numpy as np
import time
import math
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, GradientBoostingRegressor
from sklearn.cross_validation import train_test_split

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
        grid_dict = to_dict(alldata[i],bins,grid_dict, sub_dict,train, qxqy)
    return grid_dict

# creates a dictionary that returns the following format
# (x,y): [ {place_id: [[acc,day,hr],[acc,day,hr]] },
#           { place_id: [acc,day,hr] }]
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
    time = time/(60*60)
    day = divmod(time, 168)[1]/24
    hr = divmod(time, 1)[1]*24
    day = math.floor(day)
    hr = math.floor(hr)
    X = [row_id, x, y, accuracy, day, hr]
    if sub_dict:
        other_data = {place_id: [X]}
    x = math.floor(((x-qmx)*bins)/5)
    y = math.floor(((y-qmy)*bins)/5)
    # if we already have something at this grid location
    if (x,y) in grid_dict:
        # if we want to group by place_id keys
        if sub_dict:
            print('sub_dict')
            if place_id in grid_dict[(x,y)]:
                if train:
                    grid_dict[(x,y)][place_id].append(X)
            else:
                if train:
                    grid_dict[(x,y)][place_id] = [X]
        # otherwise just append to list
        else:
            if train:
                X.insert(0,place_id)
            grid_dict[(x,y)].append(X)
            gdl_2 = math.floor(len(grid_dict[(x,y)])/4)
            # print(grid_dict[(x,y)][0:gdl_2])
    # if there is nothing at this grid location we create a new dictionary
    else:
        if sub_dict:
            grid_dict[(x,y)] = other_data
        else:
            if train:
                X.insert(0,place_id)
            grid_dict[(x,y)] = [X]
    return grid_dict


# reduces each key/val pair by
# summing all entries
def reducelists(vals,key):
    print('reducelists')
    a = len(vals)
    sums = [sum(i) for i in zip(*vals)]
    avgs = [x/a for x in sums]
    avgs = avgs.insert(a,0)
    return avgs


# reduces the grids dictionaries
def reduceit(gdict):
    print('reduceit')
    for key in gdict:
        for key2 in gdict[key]:
            gdict[key][key2] = reducelists(gdict[key][key2],key2)
    return gdict


def doall(data, estimators, bins, sub_dict, clfs, train):
    print('Starting:')
    print('============')
    print('Number of records:')
    print(len(data))
    print('Generate Dictionary')

    start = time.time()
    grid_d = grid_generate(data,bins,sub_dict,train)
    if sub_dict:
        grid_d = reduceit(grid_d)
    print('Time: ' + str(time.time()-start) + ' seconds')
    print('Generate Models')
    start = time.time()
    dfdicts = []
    preds = []
    ff = 0
    # traverse the bins "rows"
    for i in range(0,bins):
        # traverse the bins "columns"
        for j in range(0,bins):
            ff += 1
            print('Classifier Number: ' + str(ff))
            # if we have sub dict let's create lists for prediction (testing)
            if sub_dict:
                dfdict = pd.DataFrame.from_dict(grid_d[(i,j)],orient='index')
                len(dfdict)
                y = dfdict.index.tolist()
                x = dfdict.values.tolist()
            # otherwise grab the grid_d and create lists that way (training)
            else:
                dfdict = pd.DataFrame(grid_d[(i,j)])
                print('Grid row length: ' + str(len(dfdict)))
                if train:
                    dfdict.columns = ['row_id','place_id', 'x', 'y','accuracy','day','hr']
                    dfdict.place_id = dfdict.place_id.apply(float)
                    dfdict.place_id = dfdict.place_id.apply(int)
                else:
                    dfdict.columns = ['row_id','x', 'y', 'accuracy','day','hr']

                dfdict.row_id = dfdict.row_id.apply(float)
                dfdict.row_id = dfdict.row_id.apply(int)
                dfdict.row_id = dfdict.row_id.astype('str')
                # dfdict.x = dfdict.x.astype('str')
                # dfdict.y = dfdict.y.astype('str')
                print(dfdict[0:10])
                x = dfdict.ix[:,dfdict.columns != 'place_id'].values.tolist()
                y = dfdict.ix[:,dfdict.columns == 'place_id'].values.ravel()
            if train:
                clf3 = RandomForestClassifier(n_estimators=80, max_features='auto',max_depth=5, n_jobs=2, verbose=1).fit(x, y)
                estimators[(i,j)] = clf3
            else:
                pred_list = clfs[(i,j)].predict(x)
                pred_rows = dfdict.ix[:,dfdict.columns == 'row_id'].values.ravel()
                preds.append(zip(pred_rows, pred_list))
    if train:
        print('Done building trees')
    else:
        print('Done with predictions')
    print('Time: ' + str(time.time()-start) + ' seconds')
    if train:
        retval = estimators
    else:
        retval = preds
    return retval

def format_df(df_train, df_test):
    df_train.columns = ['place_id','row_id', 'x', 'y','accuracy','day','hr']
    df_train.place_id = df_train.place_id.apply(float)
    df_train.place_id = df_train.place_id.apply(int)
    print('Number of unique place_ids: ' + str(len(df_train.place_id.unique())))
    print('====================')
    df_test.columns = ['row_id','x', 'y', 'accuracy','day','hr']

    df_train.row_id = df_train.row_id.apply(float).apply(int).apply(str)

    df_test.row_id = df_test.row_id.apply(float).apply(int).apply(str)
    return df_train, df_test

# Splits dataframes to X and y lists
def split_dfs(df_holdin, df_holdout, df_test):
    x = df_holdin.ix[:,[not x for x in df_holdin.columns.isin(['place_id', 'row_id'])]].values.tolist()
    x_holdout = df_holdout.ix[:,[not x for x in df_holdout.columns.isin(['place_id', 'row_id'])]].values.tolist()
    x_test = df_test.ix[:,[not x for x in df_test.columns.isin(['place_id', 'row_id'])]].values.tolist()
    y = df_holdin.ix[:,df_holdin.columns == 'place_id'].values.ravel()
    y_holdout = df_holdout.ix[:,df_holdout.columns == 'place_id'].values.ravel()
    return x, x_holdout, x_test, y, y_holdout

def train_and_predict(bins, qx, qy, grid_location, n_est, depth, sample):
    print('Starting:')
    train_file = '../data/grid/' + grid_location + '_train.csv'
    test_file = '../data/grid/' + grid_location + '_test.csv'
    data = pd.read_csv(train_file)
    test = pd.read_csv(test_file)

    if sample:
        data = data.iloc[0:175000]
        test = test.iloc[0:50000]

    data = data.sort_values('place_id').copy()
    keep = data.place_id.value_counts() > 3
    keep = keep[keep]

    data = data[data['place_id'].isin(keep.index.tolist())]
    print('============')
    print('Number of training records (subset): ' + str(len(data)))
    print(data[0:5])
    print('Number of test records: ' + str(len(test)))
    print(test[0:5])
    print('...')
    print('Generate Dictionaries')
    sub_dict = False
    qxqy = [qx, qy]

    start = time.time()
    grid_d_train = grid_generate(data,bins,sub_dict,True, qxqy)
    grid_d_test = grid_generate(test,bins,sub_dict,False, qxqy)

    data = data.sort_values('x').sort_values('y')
    test = 0

    # Diagnostic Variables
    fit_scores = []             # scores for each grid location
    grid_populations = []       # size of particular grid location's population
    avg_vals = []
    a_few_models = []
    misclassified = []

    if sample:
        bins = 6

    print('Time: ' + str(time.time()-start) + ' seconds')
    print('Generate Models and Make Predictions:')
    start = time.time()
    dfdicts = []
    preds = []
    ff = 0
    # traverse the bins "rows"
    for i in range(0,bins):
        # traverse the bins "columns"
        for j in range(0,bins):
            loop_start = time.time()
            ff += 1
            print('\n' + '==================================')
            print('Classifier Number: ' + str(ff))
            print('====================')
            # otherwise grab the grid_d and create lists that way (training)
            df_train = pd.DataFrame(grid_d_train[(i,j)])
            df_test = pd.DataFrame(grid_d_test[(i,j)])
            train_grid_pop = len(grid_d_train[(i,j)])
            grid_populations.append(train_grid_pop)
            print('Number of rows in training grid location: ' + str(train_grid_pop))

            # format dataframes, number to strings, colnames etc
            df_train, df_test = format_df(df_train, df_test)

            # create holdout dataframes
            holdout_size = 0.05
            if sample:
                holdout_size = 0.1
            df_holdin, df_holdout = train_test_split(df_train, test_size = 0.05)

            # split holdouts, X, y
            x, x_holdout, x_test, y, y_holdout = split_dfs(df_holdin, df_holdout, df_test)


            print('Number of ids in holdout not in training set: ' + str(len(set(y_holdout) - set(y))))
            print('Unique values in train (hold_in): ' + str(len(df_holdin.place_id.unique())))
            print('Unique values in train: ' + str(len(df_train.place_id.unique())))
            print('Unique values in holdout: ' + str(len(df_holdout.place_id.unique())))
            print('length of holdout: ' + str(len(y_holdout)) + '\n')
            print('length of train:' + str(len(y)) + '\n')

            print(df_holdin.ix[:,df_train.columns != ('place_id' or 'row_id')][0:5])
            print('RandomForestClassifier')
            print('======================')
            clf3 = RandomForestClassifier(n_estimators=n_est,
            max_features='auto',
            max_depth=depth,
            n_jobs=2,
            verbose=1).fit(x, y)

            if (i*j) < 2:
                a_few_models.append(clf3)

            fits_holdout = clf3.predict(x_holdout)
            [str(x) for x in fits_holdout]
            print('\n')
            print('Holdout df:')
            print(df_holdout[0:5])
            print('Holdout preds:')
            print('-- Predicted:')
            print(fits_holdout[0:5])
            print('-- Actual:')
            print(y_holdout[0:5])
            print('\n')
            print('Holdout score:')
            fit_score = clf3.score(x_holdout,y_holdout)
            fit_scores.append(fit_score)
            avgs = [np.average(df_holdout.x), np.average(df_holdout.y), np.average(df_holdout.accuracy)]
            avg_vals.append(avgs)

            print(fit_score)
            print('Average Holdout Score:')
            print(str(np.average(fit_scores)) + '\n')
            print('Done building RandomForestClassifier')
            # use the most recently made predictor to predict the corresponding test data
            pred_list = clf3.predict(x_test)
            print('Done with predictions')
            pred_rows = df_test.ix[:,df_test.columns == 'row_id'].values.ravel()
            print('Predictions, rows, test data')
            print(pred_list[0:5])
            print(pred_rows[0:5])
            print(df_test[0:5])

            holdout_rows = df_holdout.ix[:,df_holdout.columns == 'row_id'].values.ravel()
            # append these predictions to the dictionary
            pred = list(zip(holdout_rows, fits_holdout))
            ys = list(zip(holdout_rows, y_holdout))
            print(pred)
            print('finding misclassified' + '\n' + '-----------------------')
            misclass = [i for i,j in zip(ys, pred) if i[1] != j[1]]
            #print(misclass)
            misclassified.append(misclass)
            print('\n')

            print(pred[0:5])
            print(pd.DataFrame(pred[0:5]))
            preds.append(pred)
            print('Loop Time: ' + str(time.time()-loop_start) + ' seconds')

    print('Time: ' + str(time.time()-start) + ' seconds')
    retval = [preds, fit_scores, grid_populations, a_few_models, misclassified]
    return retval

# [predictions, fit_scores, grid_populations, percent_uniques]

def fill_in(df_here, alist):
    for i in range(0,len(alist)):
        f = alist[i]
        a = pd.DataFrame([[f, 2000]])
        a.columns = ['row_id','location']
    return df_here

def fill_in_df(test1, results1):
  test1.columns = ['row_id','x','y','accuracy','time']
  t1 = test1.row_id.ravel()
  r1 = results1.row_id.ravel()
  a = set(t1) - set(r1)
  alist = list(a)
  del results1['Unnamed: 0']

  for i in range(0,len(alist)):
      f = alist[i]
      a = pd.DataFrame([[f, 2000]])
      a.columns = ['row_id','location']
      results1 = results1.append(a)

  return results1

def subset(df, fname):
    tophalf = df[df.y < 5]
    bottomhalf = df[df.y >= 5]

    # topleft and topright
    subset = tophalf[tophalf.x < 5]
    subset.to_csv('../data/grid/grid2/topleft_' + fname + '.csv', index=False)
    subset = tophalf[tophalf.x >= 5]
    subset.to_csv('../data/grid/grid2/topright_' + fname + '.csv', index=False)
    # bottomleft and bottomright
    subset = bottomhalf[bottomhalf.x < 5]
    subset.to_csv('../data/grid/grid2/bottomleft_' + fname + '.csv', index=False)
    subset = bottomhalf[bottomhalf.x >= 5]
    subset.to_csv('../data/grid/grid2/bottomright_' + fname + '.csv', index=False)





# # creates grid for x,y data
# # @param data: Pandas dataframe
# # @param boxes: int number of horizontal/vertical boxes for grid ()
# def grid_generate(data, bins):
#     grid = []
#     grid_dict = {}
#     for i in range(bins):
#         grid.append([])
#     # Divide bins up
#     step = 1/bins
#     # lx = lambda x: math.floor(x/step)
#     # ly = lambda y: math.floor(y/step)
#     # whichbin = lambda item,x,dim: return item[dim] == x
#     # binx = [lx(x) for x in data['x'].values.tolist()]
#     # biny = [ly(y) for y in data['y'].values.tolist()]
#     data.place_id = data.place_id.astype('str')
#     alldata = data.values.tolist()
#     [to_dict(row) for row in xy]
#     #other_data = data[['place_id','accuracy','time']].values.tolist()
#     #return grid_dict[(binx,biny)].append(place_id)
