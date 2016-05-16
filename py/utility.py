import pandas as pd
import time
import math

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
# @param boxes: int number of horizontal/vertical boxes for grid ()
def grid_generate(data, bins):
    grid = []
    grid_dict = {}
    for i in range(bins):
        grid.append([])
    # Divide bins up
    step = 1/bins
    data.place_id = data.place_id.astype('str')
    alldata = data.values.tolist()
    [to_dict(row,bins,grid_dict) for row in alldata]
    #other_data = data[['place_id','accuracy','time']].values.tolist()
    #return grid_dict[(binx,biny)].append(place_id)
    return grid_dict


def to_dict(row,bins,grid_dict):
    row_id, x, y, accuracy, time = row[0], row[1], row[2], row[3], row[4]
    place_id = row[5]
    other_data = {place_id: [[accuracy, time]]}
    x = math.floor(x*bins/10)
    y = math.floor(y*bins/10)
    if (x,y) in grid_dict:
        if place_id in grid_dict[(x,y)]:
            grid_dict[(x,y)][place_id].append([accuracy, time])
        else:
            grid_dict[(x,y)][place_id] = [[accuracy, time]]
    else:
        grid_dict[(x,y)] = other_data

# [lambda item: item/20 for item in xco]




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
