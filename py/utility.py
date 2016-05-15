import pandas as pd
import time
import seaborn as sb

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

def splitup(line):
    accuracy = group.accuracy
