# 读excel 把每一个sheet都保存为一个txt
import numpy as np
import xlrd

# excel_path = "F:/ZYY/chess/images/train/1_3053.xlsx"
# save_file = "F:/ZYY/chess/images/train/labels0312/"

excel_path = "F:/ZYY/chess/images/test3/check/matrix-ensemble/186_324.xlsx"
save_file = "F:/ZYY/chess/images/test3/labels/"

# 打开文件，获取excel文件的workbook（工作簿）对象
excel = xlrd.open_workbook(excel_path)
for sheet in excel.sheets():
    i = 0
    # 获取sheet的名字(list类型)
    sheet_name = sheet.name
    print (sheet_name)
    # 按sheet名字获取sheet内容
    table = excel.sheet_by_name(sheet_name)
    # 读取 19*19大小的内容
    start = 0 # 开始的行
    end = 19  # 结束的行
    rows = end - start
    list_values = []
    for x in range(start, end):
        values = []
        row = table.row_values(x)
        for i in range(0, 19):
            values.append(int(row[i]))
        list_values.append(values)
    datamatrix = np.array(list_values)
    # 判别是否为空，空的记录下来

    # 将读到的内容写在txt中
    save_f = save_file + sheet_name + '.txt'
    # line = []
    with open(save_f, "w") as file:  # 只需要将之前的”w"改为“a"即可，代表追加内容
        file.write(str(datamatrix))