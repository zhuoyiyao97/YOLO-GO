# coding:UTF-8
import numpy as np
import SR_functions as Sf


if __name__ == "__main__":
    # inputfile = "SoftInput"
    inputfile = "F:/ZYY/chess/model_ensemble/data/train/train_data11_1221.txt"
    # 1.导入训练数据
    print("-----------1.load data------------")
    feature, label, k = Sf.load_data(inputfile)
    # 画图
    Sf.figurePlot(feature, label)
    # 2.导入测试数据
    print("-----------2.training------------")
    weights = Sf.gradientAscent(feature, label, k, 50000, 0.2)
    Sf.save_model("weights", np.mat(weights))

