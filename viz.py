import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree

# https://www.youtube.com/watch?v=tNa99PG8hR8
# Machine Learning Recipes #1
# 19 March 2017

# Goals
#  1. Import dataset
#  2. Train a classifier
#  3. Predict label for new flower
#  4. Visualize the tree

# importing the iris dataset
# 150 data points, 4 features, 3 possible targets
# https://en.wikipedia.org/wiki/Iris_flower_data_set
# http://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html
iris = load_iris()

# set aside some of the dataset for testing
# the classifier after training it
# testing is a very important part of ML
test_idx = [0, 50, 100]

# training data
# remove testing data from iris.target and iris.data
train_data = np.delete(iris.data, test_idx, axis=0)
train_target = np.delete(iris.target, test_idx)

# testing data
test_data = iris.data[test_idx]
test_target = iris.target[test_idx]

# create the classifier, a decision tree
clf = tree.DecisionTreeClassifier()
# train the classifier on our training data
clf = clf.fit(train_data, train_target)

# actual targets for testing data
# [0 1 2]
print test_target
# targets predicted by tree for features of testing data
print clf.predict(test_data)
# Out: [0 1 2]
# predictive labels match testing data


# viz code
from sklearn.externals.six import StringIO
import pydot
dot_data = StringIO()
tree.export_graphviz(clf,
                         out_file=dot_data,
                         feature_names=iris.feature_names,
                         class_names=iris.target_names,
                         filled=True, rounded=True,
                         impurity=False)

graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("iris.pdf")