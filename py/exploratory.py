import utility as ut
import pandas as pd
import time



def explore(nrows):
    data = pd.read_csv('../data/sample.csv')
    data = data[0:nrows]
    print('Unique Locations')
    unique_places = len(data.place_id.unique())
    print(data.place_id.nunique())
    print('Group By Place ID')
    gbdata = data.groupby('place_id')
    # Should we run analysis on groupby results or
    # loop through unique places and run analyses on those subsets
    print('Do some analysis on groups')
    spread = pd.Series('')
    start = time.time()
    spread = gbdata.apply(ut.spreads)
    finish = time.time()
    print('Apply time: ' + str(finish-start))
    print('Rows: ' + str(nrows))
    dfspread = pd.DataFrame(spread)
    #spread = dfspread.apply(ut.splitup)
    return spread
