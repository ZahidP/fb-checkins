import pandas as pd


def get_misclassified(results, location):
    """ Matches misclassified place_ids with their full row data.
        The place_ids must have been misclassified at least twice.
        @param results List of lists of misclassified [row_id, place_id]
        @param location String for which grid location we would like to use.
        returns {wrong} DataFrame of all misclassified rows.
    """
    misclassified = results
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
    fname = '../data/grid/full/' + location + '_.csv'
    train = pd.read_csv(fname)
    # subset by the row_id if it's in the index of c
    mc2 = misclass.place_id.isin(c.index.tolist())
    print(len(mc2))
    mc3 = misclass[mc2]
    print('Number of misclassified: ' + str(len(mc3)))
    train.row_id = train.row_id.astype('str')
    wrong = train[train.row_id.isin(mc3.row_id.tolist())]
    return wrong


def correct_incorrect_scatter():
    # Incorrect rows from holdout set
    incorrect = pd.read_csv('../topright_incorrect2.csv')
    # Correct rows from holdout set
    correct = pd.read_csv('../topright_correct2.csv')
    # All points 5<X<10, 0<Y<5 (what I called topright quadrant)
    topright = pd.read_csv('../topright_.csv')
    # Get the place_ids we got wrong most
    topright_wrong.place_id.value_counts().sort_values(ascending=False)[0:20]
    # "7858314184" was the place_id of one of the rows we got wrong most
    correct78 = correct[correct.place_id == 7858314184]
    incorrect78 = incorrect[incorrect.place_id == 7858314184]
    # Make sure the samples are near the occurrences of 7858314184
    topright_sample = topright_sample[topright_sample.x > 5.62]
    topright_sample = topright_sample[topright_sample.x < 5.78]
    topright_sample = topright_sample[topright_sample.y > 3.68]
    topright_sample = topright_sample[topright_sample.y < 3.75]
    # Get a sample of the topright data set
    tr_samp = topright_sample.sample(1000)
    # Get all the occurrences of 7858314184
    tr_samp78 = topright_sample[topright_sample.place_id == 7858314184]
    # Make the plots
    plt.scatter(correct78.x,correct78.y,correct78.accuracy,color='g')
    plt.scatter(incorrect78.x,incorrect78.y,incorrect78.accuracy,color='r')
    plt.scatter(tr_samp.x,tr_samp.y,tr_samp.accuracy,alpha=0.10,color='gray')
    plt.scatter(tr_samp78.x,tr_samp78.y,tr_samp78.accuracy,alpha=0.10,color='blue')
    plt.title('Correct (g) vs Incorrect (r) vs Place 78.. vs Other (gray),  Accuracy (radius)')
    plt.savefig('../visualizations/place_78583_scatter_full.png')
    plt.show()
    plt.scatter(correct78.x,correct78.y,correct78.accuracy,color='g')
    plt.scatter(incorrect78.x,incorrect78.y,incorrect78.accuracy,color='r')
    plt.scatter(tr_samp.x,tr_samp.y,tr_samp.accuracy,alpha=0.15,color='gray')
    plt.title('Correct (g) vs Incorrect (r) vs Other (gray),  Accuracy (radius)')
    plt.savefig('../visualizations/place_78583_scatter_half.png')
    plt.show()
