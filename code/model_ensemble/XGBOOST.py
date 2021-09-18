import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_squared_error

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


# 读入数据
inputfile = "F:/ZYY/chess/model_ensemble/data/train/train_data11_21.txt"
inputfile2 = "F:/ZYY/chess/model_ensemble/data/train/test_data11_21.txt"
# inputfile = "train_data4.txt"
# inputfile2 = "F:/ZYY/chess/model_ensemble/data/train/test_data4.txt"
X_train, y_train= load_data(inputfile)
X_test, y_test = load_data(inputfile2)

# 加载numpy的数组到DMatrix对象
xg_train = xgb.DMatrix(X_train, label=y_train)
xg_test = xgb.DMatrix(X_test, label=y_test)
# 1.训练模型
# setup parameters for xgboost
param = {}
# use softmax multi-class classification
param['objective'] = 'multi:softmax'
# scale weight of positive examples
param['eta'] = 0.05 # 学习率
param['max_depth'] = 5 # 4--6, train4取5比较好
param['silent'] = 1 # 不用管
param['nthread'] = 4 # 不用管
param['num_class'] = 3

watchlist = [(xg_train, 'train'), (xg_test, 'test')]
num_round = 6 # 默认为6
bst = xgb.train(param, xg_train, num_round, watchlist);

pred = bst.predict(xg_test);
print('predicting, classification precision=%f' % (
            1 - sum(int(pred[i]) != y_test[i] for i in range(len(y_test))) / float(len(y_test))))

txt_path = 'F:/ZYY/chess/model_ensemble/data/test/TXT/xgboost.txt'

err = 0
for i in range(len(pred)):
    if pred[i] != y_test[i]:
        err += 1
        print(i)
        # print("序号: ", i,'\t图片', ((i+1) //361 +1), )
        yushu = (i+1)%361
        hang = yushu // 19 + 1
        lie = yushu % 19
        print('图片序号：', ((i + 1) // 361 + 1), "\t位置：(" ,hang,',',lie,')', " 检测值: ", str(pred[i]), " 真实值： ", str(y_test[i]))

        # with open(txt_path, "a") as file:
        #     file.write("序号: "+ str(i)+ " 检测值: "+ str(pred[i])+ " 真实值： "+ str(y_test[i]) + '\n')

print("err = ", str(err), "\tall = 18050 ", "\tacc = ", str((18050 - err) / 18050))
# with open(txt_path, "a") as file:
#     file.write("err = "+str(err)+ "\tall = 18050 "+ "\tacc = "+ str((18050 - err) / 18050))
# file.close()