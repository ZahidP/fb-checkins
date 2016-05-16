import utility as ut
import pandas as pd
import time
import seaborn as sns
import numpy as np

def explorewith(nrows,option):
    data = pd.read_csv('../data/sample.csv')
    data = data[0:nrows]


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
    start = time.time()
    spread = gbdata.apply(ut.spreads)
    finish = time.time()
    print('Apply time: ' + str(finish-start))
    print('Rows: ' + str(nrows))
    dfspread = pd.DataFrame(spread)
    #spread = dfspread.apply(ut.splitup)
    return spread

def heatmaps(nrows):
    data = pd.read_csv('../data/sample.csv')
    data = data[0:nrows]
    #data = data[['x','y']].values.tolist()
    data = np.histogram2d(data['x'],data['y'],bins=20)[0]
    sns.heatmap(data)
    sns.plt.show()

# def extraheatmaps(nrows, *args):
#     # maybe start looking into geopandas
#     # the essence behind this is that we are segmenting everything in 2 dimensions
#     # instead of just 1
