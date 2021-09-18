# -*- coding: utf-8 -*-
# https://blog.csdn.net/qq_36940806/article/details/100104155

from sklearn.datasets import load_iris
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
import pandas as pd

# 显示所有列
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
import lightgbm as lgb

pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 100)


def stacking(model, train_data, train_target, test_data, n_fold):
    """
    :param model:  模型算法
    :param train_data:  训练集(不含带预测的目标特征)
    :param train_target:  需要预测的目标特征
    :param test_data:   测试集
    :param n_fold:   交叉验证的折数
    :return:
    """
    skf = StratifiedKFold(n_splits=n_fold, random_state=1)  # StratifiedKFold 默认分层采样
    train_pred = np.zeros((train_data.shape[0], 1), int)  # 存储训练集预测结果
    test_pred = np.zeros((test_data.shape[0], 1), int)  # 存储测试集预测结果 行数：len(test_data) ,列数：1列
    for skf_index, (train_index, val_index) in enumerate(skf.split(train_data, train_target)):
        print('第 ', skf_index + 1, ' 折交叉验证开始... ')
        # 训练集划分
        x_train, x_val = train_data.iloc[train_index], train_data.iloc[val_index]
        y_train, y_val = train_target.iloc[train_index], train_target.iloc[val_index]
        # 模型构建
        y_train = np.ravel(y_train)  # 向量转成数组
        model.fit(X=x_train, y=y_train)
        # 模型预测
        accs = accuracy_score(y_val, model.predict(x_val))
        print('第 ', skf_index + 1, ' 折交叉验证 :  accuracy ： ', accs)

        # 训练集预测结果
        val_pred = model.predict(x_val)
        for i in range(len(val_index)):
            train_pred[val_index[i]] = val_pred[i]
        # 保存测试集预测结果
        test_pred = np.column_stack((test_pred, model.predict(test_data)))  # 将矩阵按列合并

    test_pred_mean = np.mean(test_pred, axis=1)  # 按行计算均值(会出现小数)
    test_pred_mean = pd.DataFrame(test_pred_mean)  # 转成DataFrame
    test_pred_mean = test_pred_mean.apply(lambda x: round(x))  # 小数需要四舍五入成整数
    return np.ravel(test_pred_mean), train_pred

def load_data(inputfile):
    """
    :param inputfile:训练样本的文件名
    :return:feature_data(mat)特征
            label_data(mat)标签
            k(int)类别的个数
    """
    f = open(inputfile)
    feature_data = []
    label_data = []
    for line in f.readlines():
        feature_temp = []
        # feature_temp.append(1)
        lines = line.strip().split("\t")
        for i in range(len(lines) - 1):
            feature_temp.append(float(lines[i]))
        label_data.append(int(lines[-1]))
        feature_data.append(feature_temp)
    f.close()
    return np.array(feature_data), np.array(label_data).T
    # return np.mat(feature_data), np.mat(label_data).T

def err_test(y_pred, y_test, x_test):
    err = 0
    for i in range(len(y_pred)):
        if y_pred[i] != y_test[i]:
            err += 1
            print("序号: ", i, " 检测值: ", str(y_pred[i]), " 真实值： ", str(y_test[i]))
    print("err = ", str(err), "\tall = 18050 ", "\tacc = ", str((18050 - err) / 18050))

    err = 0
    for i in range(len(y_pred)):
        if y_pred[i] != y_test[i]:
            err += 1
            # print("序号: ", i,'\t图片', ((i+1) //361 +1), )
            yushu = (i + 1) % 361
            hang = yushu // 19 + 1
            lie = yushu % 19
            print('图片序号:', ((i + 1) // 361 + 1), " 位置：(", hang, ',', lie, ')')
            print("1*1=", str(x_test[i][0]), " 1*2=",str(x_test[i][2])," pred= ",str(y_pred[i])," 真实值=",str(y_test[i]))
    print("err = ", str(err), "\tall = 18050 ", "\tacc = ", str((18050 - err) / 18050))

def make_txt(y_pred, x_test, y_test,txt_path):
    for i in range(len(y_pred)):
        with open(txt_path, "a") as file:
            file.write("1*1="+str(int(x_test[i][0]))+ "\t1*2="+ str(int(x_test[i][2]))+"\tpred="+ str(int(y_pred[i]))+"\t真实值="+ str(int(y_test[i])) + "\n")
    file.close()


if __name__ == '__main__':
    # # 导入鸢尾花数据集
    # iris = load_iris()
    # x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=1)

    # 读入数据
    inputfile = "F:/ZYY/chess/model_ensemble/data/train/train_data11_12.txt"
    inputfile2 = "F:/ZYY/chess/model_ensemble/data/train/test_data11_12.txt"
    x_train, y_train = load_data(inputfile)
    x_test, y_test = load_data(inputfile2)

    x_train = pd.DataFrame(x_train)
    y_train = pd.DataFrame(y_train)
    x_test = pd.DataFrame(x_test)
    y_test = pd.DataFrame(y_test)

    # 三个初级学习器进行初级训练
    # 随机森林算法进行训练
    rf = RandomForestClassifier(n_jobs=-1, max_depth=100, n_estimators=800)
    print('==============================随机森林模型==============================')
    rf_test_pred, rf_train_pred = stacking(model=rf, train_data=x_train, train_target=y_train, test_data=x_test,
                                           n_fold=5)
    rf_test_pred = pd.DataFrame(rf_test_pred)
    rf_train_pred = pd.DataFrame(rf_train_pred)

    # 用决策树算法进行训练
    dt = DecisionTreeClassifier(random_state=1)
    print('==============================决策树模型==============================')
    dt_test_pred, dt_train_pred = stacking(model=dt, train_data=x_train, train_target=y_train, test_data=x_test,
                                           n_fold=5)
    dt_test_pred = pd.DataFrame(dt_test_pred)
    dt_train_pred = pd.DataFrame(dt_train_pred)

    # 用K近邻算法进行训练
    knn = KNeighborsClassifier()
    print('==============================K近邻模型==============================')
    knn_test_pred, knn_train_pred = stacking(model=knn, train_data=x_train, train_target=y_train, test_data=x_test,
                                             n_fold=5)
    knn_test_pred = pd.DataFrame(knn_test_pred)
    knn_train_pred = pd.DataFrame(knn_train_pred)

    # rf_train_pred,dt_train_pred,knn_train_pred 合并生成次级训练集 train_set
    # rf_test_pred,dt_test_pred,knn_test_pred 合并生成次级测试集集 test_set
    train_set = pd.concat([rf_train_pred, dt_train_pred, knn_train_pred], axis=1)
    test_set = pd.concat([rf_test_pred, dt_test_pred, knn_test_pred], axis=1)

    # knn_test_pred, knn_train_pred = stacking(model=knn, train_data=x_train, train_target=y_train, test_data=x_test,
    #                                          n_fold=5)
    # dt_test_pred, dt_train_pred = stacking(model=dt, train_data=x_train, train_target=y_train, test_data=x_test,
    #                                        n_fold=5)
    # rf_test_pred, rf_train_pred = stacking(model=rf, train_data=x_train, train_target=y_train, test_data=x_test,
    #                                        n_fold=5)
    # for i in range(18050):
    #     if knn_test_pred[i]!= dt_test_pred[i] or knn_test_pred[i] != rf_test_pred[i] or dt_test_pred[i]!= rf_test_pred[i]:
    #         print(i,"  三个结果不同哎")

    # # lightgbm作为次级学习器进行训练
    # lgb = lgb.LGBMClassifier(
    #     objective='multiclass',
    #     num_class=3,
    #     boosting_type='gbdt',
    #     n_estimators=2000,
    #     subsample=0.8,
    #     learning_rate=0.1,
    #     n_jobs=4,
    #     reg_alpha=0.1,
    #     reg_lambda=0.1,
    #     num_leaves=55
    # )
    # y_train = np.ravel(y_train)  # 向量转成数组
    # lgb.fit(train_set, y_train)
    # y_pred = lgb.predict(test_set)

    print('==============================第二模型 xgb==============================')
    # params = {
    #     'learning_rate': 0.1,
    #     'lambda_l1': 0.1,
    #     'lambda_l2': 0.2,
    #     'max_depth': 4,
    #     'objective': 'multiclass',
    #     'num_class': 3,
    #     # lightgbm.basic.LightGBMError: b'Number of classes should be specified and greater than 1 for multiclass training'
    # }
    #
    # print('Start training...')
    # # 训练 cv and train
    # lgb_train = lgb.Dataset(x_train, y_train)  # 将数据保存到LightGBM二进制文件将使加载更快
    # lgb_eval = lgb.Dataset(x_test, y_test, reference=lgb_train)  # 创建验证数据
    # gbm = lgb.train(params, lgb_train, num_boost_round=20, valid_sets=lgb_eval,
    #                 early_stopping_rounds=5)  # 训练数据需要参数列表和数据集
    # gbm = lgb.train(params, lgb_train, num_boost_round=20, valid_sets=lgb_eval,
    #                 early_stopping_rounds=5)  # 训练数据需要参数列表和数据集
    # gbm.save_model('model.txt')  # 训练后保存模型到文件
    # y_pred = gbm.predict(x_test, num_iteration=gbm.best_iteration)  # 如果在训练期间启用了早期停止，可以通过best_iteration方式从最佳迭代中获得预测
    # y_pred = y_pred.argmax(axis=1)

    param = {}
    # use softmax multi-class classification
    param['objective'] = 'multi:softmax'
    # scale weight of positive examples
    param['eta'] = 0.05  # 学习率
    param['max_depth'] = 5  # 4--6, train4取5比较好
    param['silent'] = 1  # 不用管
    param['nthread'] = 4  # 不用管
    param['num_class'] = 3

    xg_train = xgb.DMatrix(train_set, label=y_train)
    xg_test = xgb.DMatrix(test_set, label=y_test)

    watchlist = [(xg_train, 'train'), (xg_test, 'test')]
    num_round = 6  # 默认为6
    bst = xgb.train(param, xg_train, num_round, watchlist)

    y_pred = bst.predict(xg_test)
    x_test2, y_test2 = load_data(inputfile2)

    # 模型评价
    err_test(y_pred, y_test2, x_test2)

    # 导出结果
    txt_path = 'F:/ZYY/chess/model_ensemble/data/train/TXT/11-21-stacking.txt'
    make_txt(y_pred, x_test2, y_test2, txt_path)