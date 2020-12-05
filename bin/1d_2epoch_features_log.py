import os
import sys
# specify the path to util.py file
sys.path.insert(1, os.path.join(os.getcwd(), 'bin'))
import util
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn import tree
import matplotlib.pyplot as plt
import numpy as np

from sklearn.tree import export_graphviz
import graphviz

if __name__ == '__main__': 
    # load list of trained rfr
    list_rfr = pickle.load(open('data/1d-2epoch/list-rfr-full-log','rb'))
    # list_rfr = pickle.load(open('data/1d-2epoch/list-rfr-exclude-log','rb'))
    # load training data set
    list_train_dict = pickle.load(open('data/1d-2epoch/train-data-full','rb'))
    # list_train_dict = pickle.load(open('data/1d-2epoch/train-data-exclude','rb'))

    for n in range(4):
        train_dict = list_train_dict[n]
        # examine rfr features
        list_importances = [rfr.feature_importances_ for rfr in list_rfr]
        # print(type(list_importances[1]))
        # print(len(list_importances[1]))
        importances = list_importances[n]

        # check the sum of importances for each rfr
        # should sum to 1
        # list_sum = [sum(list(importances)) for importances in list_importances]
        # print(list_sum)
        # print(list_importances)
        std = np.std([tree.feature_importances_ 
                    for tree in list_rfr[n].estimators_], axis=0)

        # sort features based on descending order of importance
        # indices = np.argsort(importances)
        indices = np.argsort(importances)[::-1]

        # Load X
        # need to convert from list to np array
        X = np.array([train_dict[params].data.flatten() for params in train_dict])
        # print(X.shape[1])

        # Print the feature ranking
        print("Feature ranking:")
        for f in range(X.shape[1]):
            print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

        # Plot the impurity-based feature importances of the forest
        plt.figure()
        plt.title("Feature importances")
        plt.bar(range(X.shape[1]), importances[indices],
                color="r", yerr=std[indices], align="center")
        plt.xticks(range(X.shape[1]), indices)
        plt.xlim([-1, X.shape[1]])
        plt.show()

        # visualize a random tree in each RFR (tree #49)
        # random_tree = list_rfr[3].estimators_[49]
        
        # tree.plot_tree(random_tree, filled=True)
        # plt.savefig("theta10000_tree_49.pdf", dpi=300, bbox_inches='tight') 
        
        # dot_data = tree.export_graphviz(random_tree)
        # graph = graphviz.Source(dot_data) 
        # graph.render("theta10000_tree_49_2")