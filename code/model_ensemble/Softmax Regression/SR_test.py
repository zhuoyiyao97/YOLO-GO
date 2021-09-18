import SR_functions as Sf

if __name__ == "__main__":
    # 1.导入Softmax模型
    print("----------------1.load model---------------")
    w, m, n = Sf.load_weights("weights")
    # 2.导入测试数据
    print("----------------2.load test data---------------")
    # test_data = Sf.load_testData(4000, m)
    test_data, label2, k2 = Sf.load_data("F:/ZYY/chess/model_ensemble/data/train/test_data11_1221.txt")
    print("----------------3.get prediction---------------")
    result = Sf.predict(test_data, w)
    print("----------------4.save result---------------")
    Sf.save_result("test_result", result)
    err = 0
    for i in range(len(result)):
        if result[i] != label2[i]:
            err += 1
            print("序号: ", i, " 检测值: ", str(result[i]), " 真实值： ", str(label2[i]))
    print("err = ",str(err),"\tall = 18050 ","\tacc = ",str((18050-err)/18050))

    # 画图
    # Sf.figurePlot(test_data, result)
