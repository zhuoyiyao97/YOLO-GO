# https://blog.csdn.net/u013421629/article/details/78810182
from __future__ import division
import numpy as np

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

inputfile = "F:/ZYY/chess/model_ensemble/data/train/train_data11_12.txt"
inputfile2 = "F:/ZYY/chess/model_ensemble/data/train/test_data11_12.txt"
X_train, y_train, k = load_data(inputfile)
X_test, y_test, k  = load_data(inputfile2)
# 从sklearn.preprocessing里选择导入数据标准化模块。
from sklearn.preprocessing import StandardScaler
# 从sklearn.neighbors里选择导入KNeighborsClassifier，即K近邻分类器。
from sklearn.neighbors import KNeighborsClassifier

# 对训练和测试的特征数据进行标准化。
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

# 使用K近邻分类器对测试数据进行类别预测，预测结果储存在变量y_predict中。
knc = KNeighborsClassifier()
knc.fit(X_train, y_train)
y_predict = knc.predict(X_test)

# 使用模型自带的评估函数进行准确性测评。
print ('The accuracy of K-Nearest Neighbor Classifier is', knc.score(X_test, y_test))
# 依然使用sklearn.metrics里面的classification_report模块对预测结果做更加详细的分析。
from sklearn.metrics import classification_report
print (classification_report(y_test, y_predict, target_names=['0' ,'1' ,'2']))


# txt_path = 'F:/ZYY/chess/model_ensemble/data/test/TXT/knn.txt'

err = 0
for i in range(len(y_predict)):
    if y_predict[i] != y_test[i]:
        err += 1
        print("序号: ", i, " 检测值: ", str(y_predict[i]), " 真实值： ", str(y_test[i]))
        # with open(txt_path, "a") as file:
        #     file.write("序号: "+ str(i)+ " 检测值: "+ str(y_predict[i])+ " 真实值： "+ str(y_test[i]) + '\n')

print("err = ", str(err), "\tall = 18050 ", "\tacc = ", str((18050 - err) / 18050))
with open(txt_path, "a") as file:
    file.write("err = "+str(err)+ "\tall = 18050 "+ "\tacc = "+ str((18050 - err) / 18050))
file.close()
