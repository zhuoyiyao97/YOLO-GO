# 生成四角坐标
import xml.etree.ElementTree as ET
import numpy as np
import xml.dom.minidom
import os

def cal4corner(anno_path, save_path):
    # 获取文件夹中的文件
    imagelist = os.listdir(anno_path)
    coord = np.zeros(8)

    for image in imagelist:
        image_pre, ext = os.path.splitext(image)

        xml_file = anno_path + image_pre + '.xml'
        if xml_file != './anno/.DS_Store.xml':
            DOMTree = xml.dom.minidom.parse(xml_file)
            collection = DOMTree.documentElement
            objects = collection.getElementsByTagName("object")
        i = 0
        for object in objects:
            bndbox = object.getElementsByTagName('bndbox')[0]
            xmin = bndbox.getElementsByTagName('xmin')[0]
            xmin_data = xmin.childNodes[0].data
            ymin = bndbox.getElementsByTagName('ymin')[0]
            ymin_data = ymin.childNodes[0].data
            xmax = bndbox.getElementsByTagName('xmax')[0]
            xmax_data = xmax.childNodes[0].data
            ymax = bndbox.getElementsByTagName('ymax')[0]
            ymax_data = ymax.childNodes[0].data
            corner_x = ( int(xmin_data) + int(xmax_data) ) / 2
            corner_y = ( int(ymin_data) + int(ymax_data) ) / 2
            if i > 7:
                print(image)
                continue
            coord[i] = corner_x
            coord[i+1] = corner_y
            i += 2

        f = save_path + image.split('.')[0] + '.txt'
        with open(f, "w") as file:  # 只需要将之前的”w"改为“a"即可，代表追加内容
            for i in range(7):
                file.write(str(coord[i]).split('.')[0])
                file.write('\t')
            file.write(str(coord[7]).split('.')[0])
            file.close()

def main():
    # XML文件的地址,anno_path = './anno/'
    anno_path = 'F:/ZYY/chess/images/test3/txt4corner/anno-transform/'
    # corner txt文件的存储地址,anno_path = './anno/'
    save_path = 'F:/ZYY/chess/images/test3/txt4corner/txt4corner-trans/'
    cal4corner(anno_path, save_path)



if __name__ == '__main__':
    main()