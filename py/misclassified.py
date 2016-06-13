import utility as ut
import pandas as pd

def get_misclassified(results, location):
    #results = ut.train_and_predict(20, 1, 1, 'bottomright', 200, 7, True)
    misclassified = results[4]
    a = []
    for i in range(0,len(misclassified)):
        [a.append(x) for x in misclassified[i]]
    misclass = pd.DataFrame(a)
    misclass.columns = ['row_id', 'place_id']
    # get all the place_ids that showed up more than twice
    a = misclass.place_id.value_counts() > 2
    # get a list of all place ids
    b = misclass.place_id.value_counts()
    # subset by those that show up more than twice
    c = b[a]
    fname = '../data/grid/' + location + '_train.csv'
    train = pd.read_csv(fname)
    # subset by the row_id if it's in the index of c
    mc2 = misclass.place_id.isin(c.index.tolist())
    print(len(mc2))
    mc3 = misclass[mc2]
    print(len(mc3))
    print('Number of misclassified: ' + str(len(mc3)))
    train.row_id = train.row_id.astype('str')
    wrong = train[train.row_id.isin(mc3.row_id.tolist())]

    return wrong
