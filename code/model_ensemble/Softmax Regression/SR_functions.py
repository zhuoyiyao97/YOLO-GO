import numpy as np
import matplotlib.pyplot as plt
import random as rd


#   模型训练函数
def gradientAscent(feature_data, label_data, k, maxCycle, alpha):
    """
    利用梯度下降法训练Softmax模型
    :param feature_data:（mat）特征
    :param label_data:（mat）标签
    :param k:（int）类别的个数
    :param maxCycle:（int）最大迭代个数
    :param alpha: （float）学习速率
    :return:weights（mat）权重
    """
    m, n = np.shape(feature_data)
    weights = np.mat(np.ones((n, k)))
    i = 0
    while i <= maxCycle:
        err = np.exp(feature_data * weights)
        if i % 1000 == 0:
            print("\t-----iter:", i, ",cost:", cost(err, label_data))
        row_sum = -err.sum(axis=1)
        row_sum = row_sum.repeat(k, axis=1)
        err = err / row_sum
        for x in range(m):
            err[x, label_data[x, 0]] += 1
        weights = weights + (alpha / m) * feature_data.T * err
        i += 1
    return weights


def cost(err, label_data):
    """
    计算损失函数值
    :param err:（mat）,exp值
    :param label_data:（）mat标签的值
    :return:sum_cost/m  (float):损失函数的值
    """
    m = np.shape(err)[0]
    sum_cost = 0.0
    for i in range(m):
        if err[i, label_data[i, 0]] / np.sum(err[i, :]) > 0:
            sum_cost -= np.log(err[i, label_data[i, 0]] / np.sum(err[i, :]))
        else:
            sum_cost -= 0
    return sum_cost / m


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
    # print(feature_data)
    # feature_data = np.mat(feature_data)
    # print(feature_data)
    return np.mat(feature_data), np.mat(label_data).T, len(set(label_data))


def save_model(filename, weights):
    """
    保存最终的模型
    :param filename:(string)文件名
    :param w: （mat）SR 模型的权重
    :return:
    """
    f_w = open(filename, "w")
    m, n = np.shape(weights)
    for i in range(m):
        w_tmp = []
        for j in range(n):
            w_tmp.append(str(weights[i, j]))
        f_w.write("\t".join(w_tmp) + "\n")
    f_w.close()


def figurePlot(feature, label):
    """
    根据标签画出各个类别的点图
    :param feature:（mat）特征
    :param label:（mat）标签
    :return:
    """
    point_red = []
    point_blue = []
    point_yellow = []
    point_green = []
    r = np.shape(feature)[0]
    for index in range(r):
        temp = -2  # 从右到左第二个数
        if label[index] == 0:
            point_red.append(feature[index, temp])
            point_red.append(feature[index, temp + 1])  # 如果一次放两个元素feature[index, 1:3]，会被当做一个整体存放在list中
        elif label[index] == 1:
            point_blue.append(feature[index, temp])
            point_blue.append(feature[index, temp + 1])
        elif label[index] == 2:
            point_yellow.append(feature[index, temp])
            point_yellow.append(feature[index, temp + 1])
        else:
            point_green.append(feature[index, temp])
            point_green.append(feature[index, temp + 1])
    # 画出图形
    point_red = np.mat(point_red).reshape(-1, 2)  # list转换成一维矩阵，再转换成二维矩阵
    point_blue = np.mat(point_blue).reshape(-1, 2)
    point_green = np.mat(point_green).reshape(-1, 2)
    point_yellow = np.mat(point_yellow).reshape(-1, 2)
    plt.scatter(point_red[:, 0].tolist(), point_red[:, 1].tolist(), c='r')  # scatter函数只接受list，不接受matrix
    plt.scatter(point_blue[:, 0].tolist(), point_blue[:, 1].tolist(), c='b')
    plt.scatter(point_green[:, 0].tolist(), point_green[:, 1].tolist(), c='g')
    plt.scatter(point_yellow[:, 0].tolist(), point_yellow[:, 1].tolist(), c='y')
    plt.show()


# 模型测试函数
def load_weights(weights_path):
    """
    训练好的softmax模型
    :param weights_path:（string）文件名（文件的存储位置）
    :return:weights（mat）将权重存储的矩阵中
            m（int）权重的行数
            n（int）权重的列数
    """
    f = open(weights_path)
    weights = []
    for line in f.readlines():
        w_tmp = []
        lines = line.strip().split("\t")
        for x in lines:
            w_tmp.append(float(x))
        weights.append(w_tmp)
    f.close()
    print(weights)
    weights = np.mat(weights)
    m, n = np.shape(weights)
    return weights, m, n


def load_testData(num, m):
    """
    导入测试数据
    :param num:（int）生成测试样本的个数
    :param m:（int）样本的维数
    :return:testDataSet（mat）生成测试样本
    """
    test_DataSet = np.mat(np.ones((num, m)))
    for i in range(num):
        # 随机生成[-3,3]之间的随机数,rd.random生成(0,1)之间的随机数
        test_DataSet[i, 1] = rd.random() * 6 - 3
        # 随机生成[0,15]之间的随机数
        test_DataSet[i, 2] = rd.random() * 15
    return test_DataSet


def predict(test_data, weights):
    """
    利用训练好的Softmax模型对测试数据进行预测
    :param test_data:（mat）测试数据的特征
    :param weights:（mat）模型的权重
    :return:h = argmax（axis=1）##行方向的最大值所在的列数，axis=1代表行
    """
    h = test_data * weights
    return h.argmax(axis=1)  # 获得所属的类别


def save_result(filename, result):
    """
    保存最终的预测结果
    :param filename:（string）保存结果的文件名
    :param result:（mat）最终的结果
    :return:
    """
    f_result = open(filename, 'w')
    m = np.shape(result)[0]
    for i in range(m):
        f_result.write(str(result[i, 0]) + "\n")
    f_result.close()
