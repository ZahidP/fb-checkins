import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn import tree
from sklearn.externals.six import StringIO
import pydot
from IPython.display import Image



# plt.show()
# fig, ax = plt.subplots(2)
# right.hr.sort_values(ascending=False)[fifth:ninety-fifth].hist(bins=20)
# wrong.hr.sort_values(ascending=False)[fifth:ninety-fifth].hist(bins=20)
# plt.show()
# fig, axs = plt.subplots(2)
# wrong.hr.sort_values(ascending=False)[fifth:ninety-fifth].hist(bins=20, ax=axs[0,0])
# wrong.hr.sort_values(ascending=False)[fifth:ninety-fifth].hist(ax=axs[0,0])
# wrong.hr.sort_values(ascending=False)[fifth:ninety-fifth].hist(bins=20, ax=axs[0])
# right.hr.sort_values(ascending=False)[fifth:ninety-fifth].hist(bins=20, ax=axs[1])


def plot_right_wrong_histogram(right: pd.DataFrame,
                               wrong: pd.DataFrame,
                               col: str, bins: int,
                               fname: str,
                               save: bool):
    '''
    Plot 2 histograms of a particular column for correct and incorrect predictions.
    If save is true the filename will be <col>-hist-<fname>.png
    :param right: The dataframe of correct predictions.
    :param wrong: The dataframe of incorrect predictions.
    :param col: The column to plot.
    :param bins: How many bins for the histogram.
    :param fname: Any additional filename data.
    :param save: Whether or not to save the plot.
    :return: void
    '''
    # to remove any outliers, shouldn't make a big difference in what we're trying to find
    r_per = {'fifth': int(len(right)*0.05), 'ninety_fifth': int(len(right)*0.95)}
    w_per = {'fifth': int(len(wrong) * 0.05), 'ninety_fifth': int(len(wrong)*0.95)}
    fig, ax = plt.subplots(2)
    ax[0].set_title(col + ' counts: correct')
    ax[1].set_title(col + ' counts: incorrect')
    right[col].sort_values(ascending=False)[r_per['fifth']:r_per['ninety_fifth']].hist(bins=bins, ax=ax[0])
    wrong[col].sort_values(ascending=False)[w_per['fifth']:w_per['ninety_fifth']].hist(bins=bins, ax=ax[1])
    # save the figure
    tstamp = time.localtime()
    tstamp = str(tstamp.tm_mon) + '-' + str(tstamp.tm_mday) + '-' + str(tstamp.tm_min)
    file_name = col + '-hist'
    if fname:
        file_name = file_name + '-' + fname
    if save:
        plt.savefig('../visualizations/' + file_name + tstamp + '.png')
    # show the plot
    plt.show()


def plot_feature_importances(importances):
    a = ['x', 'y', 'accuracy', 'day', 'hr', 'day_of_year']
    left_bar = list(range(0, 6))
    left_bar = [x/10 for x in left_bar]
    fig, ax = plt.subplots()
    ax.bar(left_bar, importances, width=0.08)
    ax.set_title('Feature Importances')
    ax.set_title('Accuracy: Correct vs Incorrect')
    ax.set_xticklabels(a)
    tstamp = time.localtime()
    tstamp = str(tstamp.tm_mon) + '-' + str(tstamp.tm_mday) + '-' + str(tstamp.tm_min)
    plt.savefig('../visualizations/feature-importances-' + tstamp + '.png')
    plt.show()


def plot_tree(random_forest):
    # i_tree = 0
    # for tree_in_forest in random_forest.estimators_:
    #     with open('tree_' + str(i_tree) + '.dot', 'w') as my_file:
    #         my_file = tree.export_graphviz(tree_in_forest, out_file=my_file)
    tree1 = random_forest.estimators_[1]
    tree.export_graphviz(tree1, out_file='tree_file.dot')

# from sklearn.externals.six import StringIO
# import pydot
# from IPython.display import Image
# pydot.graph_from_dot_data(dotfile.getvalue()).write_png("dtree2.png")
'''
class_names = rand_forest.classes_
tree1 = rand_forest.estimators_[1]
feature_names = ["x", "y", "accuracy", "day", "hr", "month", "day_of_year"]
'''


def export_tree(feature_names, class_names, clf):
    dot_data = StringIO()
    tree.export_graphviz(clf, out_file=dot_data)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    Image(graph[0].create_png())

