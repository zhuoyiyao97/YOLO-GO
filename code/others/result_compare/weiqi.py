import tkinter as tk
import pyautogui as ag
from PIL import ImageGrab
import os
import time

def drawWeiQi(file_path, file_name, save_dir):
    win = tk.Tk()

    X, Y = ag.size()
    W, H = 800, 800
    winPos = str(W) + "x" + str(H) + "+"
    winPos += str((X - W) // 2) + "+"
    winPos += str((Y - H) // 2)
    print(winPos)
    win.geometry(winPos)
    win.resizable(False, False)
    win.title('桌面分辨率：' + str(X) + "x" + str(Y) + ' ' * 6 + '窗口大小：' + str(W) + "x" + str(H))
    win.update()

    tv = tk.Canvas(win, width=win.winfo_width(), height=win.winfo_height())
    tv.pack(side="top")

    for i in range(18):
        coord = 40, 40, 760, i * 40 + 80
        tv.create_rectangle(coord)
        coord = 40, 40, i * 40 + 80, 760
        tv.create_rectangle(coord)

    coord = 40, 40, 760, 760
    tv.create_rectangle(coord, width=2)

    f = open(file_path, "r")
    lines = f.readlines()
    for i in range(19):
        for j in range(2, len(lines[0])):
            if (lines[i][j] == '1'):
                tv.create_oval(40 * j / 2 - 18, 40 * (i+1) - 18, 40 * j / 2 + 18, 40 * (i+1) + 18, fill='white')
            if (lines[i][j] == '2'):
                tv.create_oval(40 * j / 2 - 18, 40 * (i+1) - 18, 40 * j / 2 + 18, 40 * (i+1) + 18, fill='black')


    tv.mainloop()
    # im = ImageGrab.grab((580,200,1370,1020))
    im = ImageGrab.grab()
    time.sleep(0.5)
    im = ImageGrab.grab()
    im.save(save_dir + file_name + ".jpg", 'jpeg')

if __name__ == '__main__':
    # txt_dir = 'F:/ZYY/chess/images/test3/labels/'
    txt_dir = 'F:/ZYY/chess/images/test3/check/labs/matrix-ensemble/'
    save_dir = 'F:/ZYY/chess/images/test3/check/images-ensemble/'
    for file in os.listdir(txt_dir):
        txt_path = txt_dir + file
        file_name = os.path.splitext(file)[0]
        drawWeiQi(txt_path, file_name, save_dir)