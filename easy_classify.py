#!/usr/bin/env python
# encoding:utf-8
import sys
import getopt
import math
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

import easy_excel
mem = Memory("./mycache")


@mem.cache
def get_data(file_name):
    data = load_svmlight_file(file_name)
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
            pos = int(tp+fn)
            neg = int(fp+tn)
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
        return None

#####################################################################################

# 接收命令行参数，-i接收输入libsvm格式文件，-c接收交叉验证折数，-t接收训练集分割率
opts, args = getopt.getopt(sys.argv[1:], "hi:c:t:")
input_file = ""
split_rate = 0.33
cv = 0
for op, value in opts:
    if op == "-i":
        input_file = str(value)
        if not input_file.__contains__('.libsvm'):
            print 'Your INPUT must be a libsvm format file.'
            sys.exit()
    elif op == "-c":
        cv = int(value)
    elif op == "-t":
        split_rate = float(value)
    elif op == "-h":
        print 'command: python easy_classify.py -i {input_file.libsvm} -c {int: cross validate folds}'
        print 'command: python easy_classify.py -i {input_file.libsvm} -t {float: test size rate of file}'
        sys.exit()

# 设置分类器名称及分类器模型
names = ["Nearest Neighbors", "LibSVM", "Linear SVM", "RBF SVM", "Decision Tree",
         "Random Forest", "AdaBoost", "LinearSVC", "Naive Bayes"]
classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="rbf", C=1),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    AdaBoostClassifier(),
    LinearSVC(),
    GaussianNB()]

# 导入原始数据
second = clock()
X, y = get_data(input_file)
folds = 5
results = []
print 'Time cost on loading data: ', clock() - second

# 对数据切分或交叉验证，得出结果
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=split_rate, random_state=0)
for name, model in zip(names, classifiers):
    if cv == 0:
        print u'>>>', name, 'is training...'
        out = loop_classifier(name, model, X_train, y_train, test_x=X_test, test_y=y_test)
    else:
        print u'>>>', name, 'is cross validating...'
        out = loop_classifier(name, model, X, y, cv=cv)
    if out is not None:
        results.append(out)

# 保存结果至Excel
print '====================='
if easy_excel.save(str(X_train.shape[1]), results):
    print 'Save "results.xls" successfully.'
else:
    print 'Fail to save "results.xls". Please close "results.xls" first.'
