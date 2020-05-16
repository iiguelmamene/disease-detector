#!/usr/bin/env python
# coding: utf-8

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # this is used for the plot the graph
import os
import seaborn as sns # used for plot interactive graph.
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
import sklearn.metrics.cluster as smc
from sklearn.model_selection import KFold


from matplotlib import pyplot
import itertools

get_ipython().run_line_magic('matplotlib', 'inline')
import random

random.seed(42)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

def draw_confusion_matrix(y, yhat, classes):
    '''
        Draws a confusion matrix for the given target and predictions
        Adapted from scikit-learn and discussion example.
    '''
    plt.cla()
    plt.clf()
    matrix = confusion_matrix(y, yhat)
    plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    num_classes = len(classes)
    plt.xticks(np.arange(num_classes), classes, rotation=90)
    plt.yticks(np.arange(num_classes), classes)

    fmt = 'd'
    thresh = matrix.max() / 2.
    for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
        plt.text(j, i, format(matrix[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if matrix[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()


import sys
assert sys.version_info >= (3, 5) # python>=3.5
import sklearn
assert sklearn.__version__ >= "0.20" # sklearn >= 0.20

import numpy as np #numerical package in python
import os
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt #plotting package

# to make this notebook's output identical at every run
np.random.seed(42)

#matplotlib magic for inline figures
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib # plotting library
import matplotlib.pyplot as plt

# Where to save the figures
ROOT_DIR = "."
IMAGES_PATH = os.path.join(ROOT_DIR, "images")
os.makedirs(IMAGES_PATH, exist_ok=True)

import os
import tarfile
import urllib
DATASET_PATH = os.path.join("datasets", "heartdisease")


import pandas as pd

def load_disease_data(disease_path):
    csv_path = os.path.join(disease_path, "heartdisease.csv")
    return pd.read_csv(csv_path)

heartdisease = load_disease_data(DATASET_PATH)


heartdisease.head()

heartdisease.describe()

heartdisease.info()

heartdisease.isnull().values.any()

heartdisease['sick'] *= 1

heartdisease.hist(bins=50, figsize=(20,15))
plt.show()

heartdisease["sick"].hist()
plt.show()

heartdisease["sick"].value_counts()

import seaborn as sns
corr_matrix = heartdisease.corr()

myhm = sns.heatmap(
    corr_matrix,
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(220, 20, sep=20, as_cmap=True),
    square = True
)

myhm.set_xticklabels(
    myhm.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
)

heartdisease_labels = heartdisease['sick']
heartdisease = heartdisease.drop('sick', axis = 1)


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(heartdisease, heartdisease_labels, test_size = 0.3, random_state = 42)

print (X_train.shape, y_train.shape)
print (X_test.shape, y_test.shape)

heartdisease_num = heartdisease.drop(["sex", "cp", "restecg", "slope", "fbs", "thal"], axis=1)

numerical_features = list(heartdisease_num)
categorical_features = ["sex", "cp", "restecg", "slope", "fbs", "thal"]

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

num_pipeline = Pipeline([
        ('std_scaler', StandardScaler()),
    ])

full_pipeline = ColumnTransformer([
        ("num", num_pipeline, numerical_features),
        ("cat", OneHotEncoder(), categorical_features),
    ])

heartdisease_prepared = full_pipeline.fit_transform(heartdisease)

X_train1, X_test1, y_train1, y_test1 = train_test_split(heartdisease_prepared, heartdisease_labels, test_size = 0.3, random_state = 42)

print (X_train1.shape, y_train1.shape)
print (X_test1.shape, y_test1.shape)

# SVM

from sklearn.svm import SVC

svc = SVC(C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=True, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', random_state=None)
svc.fit(X_train, y_train)

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
y_pred = svc.predict(X_test)
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Precision: ", precision_score(y_test, y_pred))
print("Recall: ", recall_score(y_test, y_pred))
print("F1 Score: ", f1_score(y_test, y_pred))

from sklearn.metrics import classification_report, confusion_matrix
draw_confusion_matrix(y_test,y_pred, ["Healthy", "Sick"])

from sklearn.metrics import roc_curve, auc

my_score = svc.predict_proba(X_test)
fpr, tpr, threshold = roc_curve(y_test, my_score[:,1], pos_label=1)
plt.figure(1)
plt.plot(fpr, tpr, color='red', label='ROC')
plt.xlabel("False Positives Rate")
plt.ylabel("True Positives Rate")
plt.show()

svc1 = SVC(C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=True, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', random_state=None)
svc1.fit(X_train1, y_train1)

y_pred1 = svc1.predict(X_test1)
print("Accuracy: ", accuracy_score(y_test1, y_pred1))
print("Precision: ", precision_score(y_test1, y_pred1))
print("Recall: ", recall_score(y_test1, y_pred1))
print("F1 Score: ", f1_score(y_test1, y_pred1))

from sklearn.metrics import classification_report, confusion_matrix
draw_confusion_matrix(y_test1,y_pred1, ["Healthy", "Sick"])

my_score = svc1.predict_proba(X_test1)
fpr, tpr, threshold = roc_curve(y_test1, my_score[:,1], pos_label=1)
plt.figure(1)
plt.plot(fpr, tpr, color='red', label='ROC')
plt.xlabel("False Positives Rate")
plt.ylabel("True Positives Rate")
plt.show()

# SVM
svc2 = SVC(C=1.0, kernel='linear', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=True, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', random_state=None)
svc2.fit(X_train1, y_train1)

y_pred2 = svc2.predict(X_test1)
print("Accuracy: ", accuracy_score(y_test1, y_pred2))
print("Precision: ", precision_score(y_test1, y_pred2))
print("Recall: ", recall_score(y_test1, y_pred2))
print("F1 Score: ", f1_score(y_test1, y_pred2))
draw_confusion_matrix(y_test1,y_pred2, ["Healthy", "Sick"])

my_score1 = svc1.predict_proba(X_test1)
fpr, tpr, threshold = roc_curve(y_test1, my_score1[:,1], pos_label=1)
plt.figure(1)
plt.plot(fpr, tpr, color='red', label='ROC')
plt.xlabel("False Positives Rate")
plt.ylabel("True Positives Rate")
plt.show()


# Logistic Regression
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(solver='sag', max_iter=10).fit(X_train1, y_train1)
y_predlog = logreg.predict(X_test1)

print("Accuracy: ", accuracy_score(y_test1, y_predlog))
print("Precision: ", precision_score(y_test1, y_predlog))
print("Recall: ", recall_score(y_test1, y_predlog))
print("F1 Score: ", f1_score(y_test1, y_predlog))
draw_confusion_matrix(y_test1,y_predlog, ["Healthy", "Sick"])


my_score1 = logreg.predict_proba(X_test1)
fpr, tpr, threshold = roc_curve(y_test1, my_score1[:,1], pos_label=1)
plt.figure(1)
plt.plot(fpr, tpr, color='red', label='ROC')
plt.xlabel("False Positives Rate")
plt.ylabel("True Positives Rate")
plt.show()

# Logistic Regression
logreg = LogisticRegression(solver='sag', max_iter=10000).fit(X_train1, y_train1)
y_predlog = logreg.predict(X_test1)

print("Accuracy: ", accuracy_score(y_test1, y_predlog))
print("Precision: ", precision_score(y_test1, y_predlog))
print("Recall: ", recall_score(y_test1, y_predlog))
print("F1 Score: ", f1_score(y_test1, y_predlog))
draw_confusion_matrix(y_test1,y_predlog, ["Healthy", "Sick"])


my_score1 = logreg.predict_proba(X_test1)
fpr, tpr, threshold = roc_curve(y_test1, my_score1[:,1], pos_label=1)
plt.figure(1)
plt.plot(fpr, tpr, color='red', label='ROC')
plt.xlabel("False Positives Rate")
plt.ylabel("True Positives Rate")
plt.show()


# Logistic Regression
logreg = LogisticRegression(penalty='none', solver='sag', max_iter=10000).fit(X_train1, y_train1)
y_predlog = logreg.predict(X_test1)

print("Accuracy: ", accuracy_score(y_test1, y_predlog))
print("Precision: ", precision_score(y_test1, y_predlog))
print("Recall: ", recall_score(y_test1, y_predlog))
print("F1 Score: ", f1_score(y_test1, y_predlog))
draw_confusion_matrix(y_test1,y_predlog, ["Healthy", "Sick"])


my_score1 = logreg.predict_proba(X_test1)
fpr, tpr, threshold = roc_curve(y_test1, my_score1[:,1], pos_label=1)
plt.figure(1)
plt.plot(fpr, tpr, color='red', label='ROC')
plt.xlabel("False Positives Rate")
plt.ylabel("True Positives Rate")
plt.show()

# k-Nearest Neighbors algorithm

from sklearn.neighbors import KNeighborsClassifier
knn_alg = KNeighborsClassifier()
knn_alg.fit(X_train1, y_train1)
y_pred_knn = knn_alg.predict(X_test1)

print("Accuracy: ", accuracy_score(y_test1, y_pred_knn))

knn_list = [1, 2, 3, 5, 10, 20, 50, 100]

for i in knn_list:
    knn_alg = KNeighborsClassifier(n_neighbors=i)
    knn_alg.fit(X_train1, y_train1)
    y_pred_knn = knn_alg.predict(X_test1)
    print("n_neighbors =", i)
    print("Accuracy for n_neighbors: ", accuracy_score(y_test1, y_pred_knn))
    print("")
