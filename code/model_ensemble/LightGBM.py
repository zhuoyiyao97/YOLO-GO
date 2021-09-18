# https://blog.csdn.net/huacha__/article/details/81057150?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-1.control&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-1.control
# 函数的更多使用方法参见LightGBM官方文档：http://lightgbm.readthedocs.io/en/latest/Python-Intro.html

import numpy as np
import json
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

# iris = load_iris()  # 载入鸢尾花数据集
# data = iris.data
# target = iris.target
# X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2)

# 加载你的数据
print('Load data...')
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

def err_test(y_pred, y_test, x_test):
    print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)  # 计算真实值和预测值之间的均方根误差

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

inputfile = "F:/ZYY/chess/model_ensemble/data/train/train_data11_12.txt"
inputfile2 = "F:/ZYY/chess/model_ensemble/data/train/test_data11_12.txt"
X_train, y_train= load_data(inputfile)
X_test, y_test = load_data(inputfile2)

# 创建成lgb特征的数据集格式
lgb_train = lgb.Dataset(X_train, y_train)  # 将数据保存到LightGBM二进制文件将使加载更快
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)  # 创建验证数据

# 将参数写成字典下形式
# params = {
#     'task': 'train',
#     'boosting_type': 'gbdt',  # 设置提升类型
#     'objective': 'multiclass',  # 目标函数
#     'metric': {'l2', 'auc'},  # 评估函数
#     'num_leaves': 31,  # 叶子节点数
#     'learning_rate': 0.05,  # 学习速率
#     'feature_fraction': 0.9,  # 建树的特征选择比例
#     'bagging_fraction': 0.8,  # 建树的样本采样比例
#     'bagging_freq': 5,  # k 意味着每 k 次迭代执行bagging
#     'verbose': 1,  # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
#     'num_class': 3,
# }
params={
    'learning_rate':0.1,
    'lambda_l1':0.1,
    'lambda_l2':0.2,
    'max_depth':4,
    'objective':'multiclass',
    'num_class':3,  #lightgbm.basic.LightGBMError: b'Number of classes should be specified and greater than 1 for multiclass training'
}

print('Start training...')
# 训练 cv and train
gbm = lgb.train(params, lgb_train, num_boost_round=20, valid_sets=lgb_eval, early_stopping_rounds=5)  # 训练数据需要参数列表和数据集

print('Save model...')

gbm.save_model('model.txt')  # 训练后保存模型到文件

print('Start predicting...')
# 预测数据集
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)  # 如果在训练期间启用了早期停止，可以通过best_iteration方式从最佳迭代中获得预测
y_pred = y_pred.argmax(axis=1)

# 评估模型
err_test(y_pred, y_test, X_test)

# 导出结果
txt_path = 'F:/ZYY/chess/model_ensemble/data/train/TXT/11-12.txt'
# make_txt(y_pred, X_test, y_test,txt_path)