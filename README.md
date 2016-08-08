# Easy-Classify
version 0.11

##Easy-Classify是什么?

Easy-Classify是一个基于python的sklearn包，自动生成二分类Excel实验报告的小脚本。分类器目前集成：

* Nearest Neighbors
* LibSVM
* Linear SVM
* RBF SVM
* Decision Tree
* Random Forest
* AdaBoost
* LinearSVC
* Naive Bayes
* ......

##运行环境

* python 2.7
* python的scikit-learn包用于跑分类器：pip install scikit-learn
* python的xlwt用于写入excel结果报告：pip intall xlwt

##输入输出
 
 * 输入：包含全部正反例的libsvm格式文件。文件正反例标签为{0,1}，3维的libsvm格式如：
 
 ```txt
  1 1:7.964601769911504 2:0.8849557522123894 3:1.1799410029498525
  0 1:9.583333333333334 2:0.8333333333333334 3:4.1666666666666660
  1 1:6.427423674343867 2:0.8569898232458489 3:5.9989287627209430
  0 1:12.50000000000000 2:2.2727272727272730 3:5.1136363636363640
```
 * 输出：Excel实验表格，如results.xls文件所示
 
##使用命令

* 交叉验证：
 ```ssh
  python easy_classify.py -i {input_file.libsvm} -c {int: cross validate folds}
  # 如：python easy_classify.py -i train.libsvm -c 5
```

* 训练测试：
 ```ssh
  python easy_classify.py -i {input_file.libsvm} -t {float: test size rate of file}
  # 如：python easy_classify.py -i train.libsvm -t 0.33
```

* 帮助：
 ```ssh
  python easy_classify.py -h
```

##升级日志
 * 2016-08-08，version 0.2:
   * 完成基本功能框架，集成九种常见分类器，支持并行，自动生成Excel测试报告
   * 未来：分类器参数自动调优，多种维度测试文件同时输入实验，集成更多分类器（libD3C，神经网络等）
 
