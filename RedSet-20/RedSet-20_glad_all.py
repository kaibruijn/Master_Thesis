import pandas as pd
import os
import sys
import shutil
from random import randrange
import math

def main():


    data = pd.read_excel("Final_RedSet-20Data_Preprocessed.xlsx")
    for i in range(len(data)):
        num = str(i+1)
        text1 = data.loc[[i]]["Text1"].tolist()[0]
        text2 = data.loc[[i]]["Text2"].tolist()[0]
        rel = data.loc[[i]]["Value"].tolist()[0]
        if rel == -1:
            rel = "N"
        else:
            rel = "Y"
        print("DU"+num, text1[0:10], text2[0:10], rel)
        make_dirs("DU"+num, text1, text2, rel)


def make_dirs(dir, known, unknown, relation):
    if len(dir) == 4:
        dir = dir[0:2] + "0" + dir[2:4]
    if len(dir) == 3:
        dir = dir[0:2] + "00" + dir[2]
    dir = "RedSet-20_all/"+dir
    try:
        shutil.rmtree(dir)
    except:
        pass

    os.mkdir(dir)
    text_file = open(dir+"/known01.txt", "w")
    text_file.write(known)
    text_file.close()
    text_file2 = open(dir+"/unknown.txt", "w")
    text_file2.write(unknown)
    text_file2.close()  
    text_file3 = open("RedSet-20_all/truth.txt", "a")
    text_file3.write(dir[11:16] +" "+ relation + "\n" + "")
    text_file3.close()

if __name__ == '__main__':
    main()