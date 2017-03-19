from sklearn import tree

# https://www.youtube.com/watch?v=cKxRvEZd3Mw
# Machine Learning Recipes #1
# 19 March 2017


# [weight, texture]
# weight in grams
# texture: 0 is bumpy, 1 is smooth
features = [[140, 1], [130, 1], [150, 0], [170, 0]]

# 0 is apple, 1 is orange
labels = [0, 0, 1, 1]

# create the classifier, a decision tree
# just an "empty box of rules"
clf = tree.DecisionTreeClassifier()

# train the classifier
# "creates" the rules of the classifier
clf = clf.fit(features, labels)

# predict the type of a 160g bumpy fruit
# similar to an orange because it's heavy and bumpy
print clf.predict([[160, 0]])