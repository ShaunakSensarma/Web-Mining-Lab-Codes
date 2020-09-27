# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 23:57:38 2020

@author: Shaunak_Sensarma
"""


import pandas as pd

credit_data = pd.read_excel(
    "Credit card approval.xls", names=[i for i in range(16)])
print(credit_data.head())
print("\n\n")
credit_data = credit_data.fillna(0)
for a in [0, 3, 4, 5, 6, 8, 9, 10, 11, 12, 15]:
    credit_data[a] = credit_data[a].astype("category").cat.codes
print("CREDIT DATA = ", credit_data.head())
print("\n\n")
credit_data = credit_data.replace("?", 0)


#DECISION TREE ALGORITHM

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    credit_data.drop(15, axis=1), credit_data[15], test_size=0.33, random_state=42
)

from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()

print(clf.fit(X_train, y_train))
print("\n\n")

print("Cross Validation Score = ",cross_val_score(clf, X_test, y_test, cv=5))
print("\n\n")
from sklearn.metrics import confusion_matrix
print("Confusion matrix")
print(confusion_matrix(y_test, clf.predict(X_test)))
print("\n\n")
tn, fp, fn, tp = confusion_matrix(y_test, clf.predict(X_test)).ravel()
print(tn, fp, fn, tp)
print("\n\n")

from sklearn.metrics import precision_recall_fscore_support

print("precision_recall_fscore_support = ",precision_recall_fscore_support(y_test, clf.predict(X_test), average="weighted"))
print("\n\n")
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score

print("Mean Squared Error = ",mean_squared_error(y_test, clf.predict(X_test)))
print("\n\n")


print("Accuracy Score =",accuracy_score(y_test, clf.predict(X_test)))
print("\n\n")


from sklearn import metrics
import matplotlib.pyplot as plt

fpr, tpr, thresholds = metrics.roc_curve(y_test, clf.predict(X_test))

plt.plot(fpr)

auc = metrics.auc(fpr, tpr)
print("AUC VALUE = ", auc)




#NAIVE BAYES ALGORITHM



from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB

clf = GaussianNB()

print(clf.fit(X_train, y_train))

print(cross_val_score(clf, X_test, y_test, cv=5))
from sklearn.metrics import confusion_matrix
print("\n\nConfusion Matrix:\n")
print(confusion_matrix(y_test, clf.predict(X_test)))

tn, fp, fn, tp = confusion_matrix(y_test, clf.predict(X_test)).ravel()
print(tn, fp, fn, tp)

print("\n\n")
from sklearn.metrics import precision_recall_fscore_support
print("precision_recall_fscore_support",precision_recall_fscore_support(y_test, clf.predict(X_test), average="weighted"))

from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score


print("\nmean_squared_error = ",mean_squared_error(y_test, clf.predict(X_test)))
print("\naccuracy_score = ",accuracy_score(y_test, clf.predict(X_test)))

from sklearn import metrics
import matplotlib.pyplot as plt

fpr, tpr, thresholds = metrics.roc_curve(y_test, clf.predict(X_test))
plt.plot(fpr)

auc = metrics.auc(fpr, tpr)
print("\nAUC VALUE = ",auc)


from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier

clf = AdaBoostClassifier()

print("\nAdaboost..  ",clf.fit(X_train, y_train))

print("CROSS validation score = ", cross_val_score(clf, X_test, y_test, cv=5))
from sklearn.metrics import confusion_matrix

print("\nConfusion matrix...")
print(confusion_matrix(y_test, clf.predict(X_test)))
print("\n")
tn, fp, fn, tp = confusion_matrix(y_test, clf.predict(X_test)).ravel()
print(tn, fp, fn, tp)


from sklearn.metrics import precision_recall_fscore_support

print("precision_recall_fscore_support = ",precision_recall_fscore_support(y_test, clf.predict(X_test), average="weighted"))

from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
print("\n\n")
print("mean_squared_error = ",mean_squared_error(y_test, clf.predict(X_test)))

print("\naccuracy_score = ",accuracy_score(y_test, clf.predict(X_test)))


from sklearn import metrics
import matplotlib.pyplot as plt

fpr, tpr, thresholds = metrics.roc_curve(y_test, clf.predict(X_test))

plt.plot(fpr)

auc = metrics.auc(fpr, tpr)
print(auc)

