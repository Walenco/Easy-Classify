#!/usr/bin/env python
# encoding:utf-8

import math
import easy_excel
from time import clock

from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.externals.joblib import Memory
from sklearn.grid_search import GridSearchCV
from sklearn.datasets import load_svmlight_file
from sklearn import cross_validation, metrics
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
mem = Memory("./mycache")


@mem.cache
def get_data(name):
    data = load_svmlight_file(name)
    return data[0], data[1]


def loop_classifier(lab, clf, train_x, train_y, test_x=None, test_y=None, cv=None):
    global second
    try:
        clf.fit(train_x, train_y)
        if clf.classes_.__len__() == 2:
            if cv is not None:
                forecast = cross_validation.cross_val_predict(clf, train_x, train_y, cv=cv)
                test_y = train_y
            else:
                forecast = clf.predict(test_x)
            mat = metrics.confusion_matrix(test_y, forecast)
            tp = float(mat[0][0])
            fp = float(mat[1][0])
            fn = float(mat[0][1])
            tn = float(mat[1][1])
            ac = '%0.4f' % metrics.accuracy_score(test_y, forecast)
            fc = '%0.4f' % metrics.f1_score(test_y, forecast)
            pos = tp+fn
            neg = fp+tn
            if (tp+fp) == 0:
                precision = 1E99
            else:
                precision = tp/(tp+fp)
            if (tp+fn) == 0:
                recall = 1E99
            else:
                recall = se = tp/(tp+fn)
            if (tn+fp) == 0:
                sp = 1E99
            else:
                sp = tn /(tn+fp)
            if se == 1E99 or sp == 1E99:
                gm = 1E99
            else:
                gm = math.sqrt(se*sp)
            f_measure = f_score = fc
            if (tp+fp)*(tn+fn)*(tp+fn)*(tn+fp) == 0:
                mcc = 1E99
            else:
                mcc = (tp*tn-fn*fp)/(math.sqrt((tp+fp)*(tn+fn)*(tp+fn)*(tn+fp)))
            print 'F-Score: ', fc, '\n', 'Accuary: ', ac
            print 'Time cost: ', clock() - second
            # Label,Accuary,Precision,Recall,SE,SP,GM,F_measure,F-Score,MCC,Matrix,TP,FN,FP,TN
            return [lab, ac, precision, recall, se, sp, gm, f_measure, f_score, mcc, ' ', tp, fn, fp, tn, pos, neg]
        else:
            print 'mutiple classification'
            return None
    except Exception, e:
        print e
        # print 'the model is unsupported.'
        return None

#####################################################################################

names = ["Nearest Neighbors", "LibSVM", "Linear SVM", "RBF SVM", "Decision Tree",
         "Random Forest", "AdaBoost", "Naive Bayes"]
classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="rbf", C=1),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    AdaBoostClassifier(),
    GaussianNB()]

second = clock()
X, y = get_data("train.libsvm")
folds = 5
results = []
print 'Time cost on loading data: ', clock() - second

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.33, random_state=0)
for name, model in zip(names, classifiers):
    print u'>>>', name, 'is training...'
    out = loop_classifier(name, model, X_train, y_train, test_x=X_test, test_y=y_test)
    # out = loop_classifier(name, model, X, y, cv=5)
    if out is not None:
        results.append(out)

print '====================='
if easy_excel.save(results):
    print 'Save "results.xls" successfully.'
else:
    print 'Fail to save "results.xls". Please close "results.xls" first.'
