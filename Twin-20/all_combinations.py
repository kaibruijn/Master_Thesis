import pandas as pd
import os
import sys
import shutil
from random import randrange
import math

def main():
    try:
        os.remove("types.txt")
    except:
        pass
    try:
        os.remove("type_ids.txt")
    except:
        pass
    du = 1
    data = pd.read_csv("TwinDataIDs.csv")
    data2 = pd.read_csv("TwinIDs.csv")
    for i in range(1,len(data2)+1):
        notlist = []
        twin_1 = data2[data2["Pair_ID"]==i]["Twin1_ID"].tolist()[0]
        try:
            twin_2 = int(data2[data2["Pair_ID"]==i]["Twin2_ID"].tolist()[0])
        except:
            twin_2 = data2[data2["Pair_ID"]==i]["Twin2_ID"].tolist()[0]
        try:
            sibling = int(data2[data2["Pair_ID"]==i]["Sibling_ID"].tolist()[0])
        except:
            sibling = data2[data2["Pair_ID"]==i]["Sibling_ID"].tolist()[0]

        if twin_1 != "-" and twin_2 != "-":
            text_twin_1 = data[data["ID"]==int(twin_1)]["Text"].tolist()[0]
            text_twin_2 = data[data["ID"]==int(twin_2)]["Text"].tolist()[0]
            notlist.append(twin_1)
            notlist.append(twin_2)
            if data[data["ID"]==int(twin_1)]["Type_of_Twin"].tolist()[0] == "I":
                make_dirs("DU"+str(du), text_twin_1, text_twin_2, "T\n")
                print_ids(twin_1, twin_2)
            else:
                make_dirs("DU"+str(du), text_twin_1, text_twin_2, "U\n")
                print_ids(twin_1, twin_2)      
            du += 1
        if twin_1 != "-" and sibling != "-":
            text_sibling = data[data["ID"]==int(sibling)]["Text"].tolist()[0]
            notlist.append(sibling)
            make_dirs("DU"+str(du), text_twin_1, text_sibling, "S\n")
            print_ids(twin_1, sibling)
            du += 1
        if twin_2 != "-" and sibling != "-":
            make_dirs("DU"+str(du), text_twin_2, text_sibling, "S\n")
            print_ids(twin_2, sibling)
            du += 1
        if twin_2 != "-" and sibling != "-":
            r = randrange(1,len(data)+1)
            if r in notlist:
                if r > (len(data)/2):
                    r -= 3
                    print_ids(twin_1, r)
                    print_ids(twin_2, r-1)
                    print_ids(sibling, r-2)
                    text_r1 = data[data["ID"]==int(r)]["Text"].tolist()[0]
                    text_r2 = data[data["ID"]==int(r-1)]["Text"].tolist()[0]
                    text_r3 = data[data["ID"]==int(r-2)]["Text"].tolist()[0]

                else:
                    r += 3
                    print_ids(twin_1, r)
                    print_ids(twin_2, r+1)
                    print_ids(sibling, r+2)
                    text_r1 = data[data["ID"]==int(r)]["Text"].tolist()[0]
                    text_r2 = data[data["ID"]==int(r+1)]["Text"].tolist()[0]
                    text_r3 = data[data["ID"]==int(r+2)]["Text"].tolist()[0]

            else:
                if r > (len(data)/2) and ((r-twin_1)>2):
                    print_ids(twin_1, r)
                    print_ids(twin_2, r-1)
                    print_ids(sibling, r-2)
                    text_r1 = data[data["ID"]==int(r)]["Text"].tolist()[0]
                    text_r2 = data[data["ID"]==int(r-1)]["Text"].tolist()[0]
                    text_r3 = data[data["ID"]==int(r-2)]["Text"].tolist()[0]

                elif r < (len(data)/2) and ((twin_1-r)>2):
                    print_ids(twin_1, r)
                    print_ids(twin_2, r+1)
                    print_ids(sibling, r+2)
                    text_r1 = data[data["ID"]==int(r)]["Text"].tolist()[0]
                    text_r2 = data[data["ID"]==int(r+1)]["Text"].tolist()[0]
                    text_r3 = data[data["ID"]==int(r+2)]["Text"].tolist()[0]

                elif r > (len(data)/2) and ((r-twin_1)<3):
                    r = math.floor(r/2)
                    print_ids(twin_1, r)
                    print_ids(twin_2, r+1)
                    print_ids(sibling, r+2)
                    text_r1 = data[data["ID"]==int(r)]["Text"].tolist()[0]
                    text_r2 = data[data["ID"]==int(r+1)]["Text"].tolist()[0]
                    text_r3 = data[data["ID"]==int(r+2)]["Text"].tolist()[0]

                elif r < (len(data)/2) and ((twin_1-r)<3):
                    r = r*2
                    print_ids(twin_1, r)
                    print_ids(twin_2, r-1)
                    print_ids(sibling, r-2)
                    text_r1 = data[data["ID"]==int(r)]["Text"].tolist()[0]
                    text_r2 = data[data["ID"]==int(r-1)]["Text"].tolist()[0]
                    text_r3 = data[data["ID"]==int(r-2)]["Text"].tolist()[0]

            make_dirs("DU"+str(du), text_twin_1, text_r1, "R\n")
            du += 1
            make_dirs("DU"+str(du), text_twin_2, text_r2, "R\n")
            du += 1
            make_dirs("DU"+str(du), text_sibling, text_r3, "R \n")
            du += 1
        elif twin_2 != "-":
            r = randrange(1,len(data)+1)
            if r in notlist:
                if r > (len(data)/2):
                    r -= 3
                    print_ids(twin_1, r)
                    print_ids(twin_2, r-1)
                    text_r1 = data[data["ID"]==int(r)]["Text"].tolist()[0]
                    text_r2 = data[data["ID"]==int(r-1)]["Text"].tolist()[0]

                else:
                    r += 3
                    print_ids(twin_1, twin_2, r, r+1)
                    text_r1 = data[data["ID"]==int(r)]["Text"].tolist()[0]
                    text_r2 = data[data["ID"]==int(r+1)]["Text"].tolist()[0]

            else:
                if r > (len(data)/2) and ((r-twin_1)>2):
                    print_ids(twin_1, r)
                    print_ids(twin_2, r-1)
                    text_r1 = data[data["ID"]==int(r)]["Text"].tolist()[0]
                    text_r2 = data[data["ID"]==int(r-1)]["Text"].tolist()[0]

                elif r < (len(data)/2) and ((twin_1-r)>2):
                    print_ids(twin_1, r)
                    print_ids(twin_2, r+1)
                    text_r1 = data[data["ID"]==int(r)]["Text"].tolist()[0]
                    text_r2 = data[data["ID"]==int(r+1)]["Text"].tolist()[0]

                elif r > (len(data)/2) and ((r-twin_1)<3):
                    r = math.floor(r/2)
                    print_ids(twin_1, r)
                    print_ids(twin_2, r+1)
                    text_r1 = data[data["ID"]==int(r)]["Text"].tolist()[0]
                    text_r2 = data[data["ID"]==int(r+1)]["Text"].tolist()[0]

                elif r < (len(data)/2) and ((twin_1-r)<3):
                    r = r*2
                    print_ids(twin_1, r)
                    print_ids(twin_2, r-1)
                    text_r1 = data[data["ID"]==int(r)]["Text"].tolist()[0]
                    text_r2 = data[data["ID"]==int(r-1)]["Text"].tolist()[0]

            make_dirs("DU"+str(du), text_twin_1, text_r1, "R\n")
            du += 1
            make_dirs("DU"+str(du), text_twin_2, text_r2, "R\n")
            du += 1
        else:
            text_twin_1 = data[data["ID"]==int(twin_1)]["Text"].tolist()[0]
            notlist.append(twin_1)
            r = randrange(1,len(data)+1)
            if r in notlist:
                if r > (len(data)/2):
                    r -= 1
                    print_ids(twin_1, r)
                    text_r1 = data[data["ID"]==int(r)]["Text"].tolist()[0]

                else:
                    r += 1
                    print_ids(twin_1, r)
                    text_r1 = data[data["ID"]==int(r)]["Text"].tolist()[0]
            print_ids(twin_1, r)
            make_dirs("DU"+str(du), text_twin_1, text_r1, "R\n")
            du += 1


def make_dirs(dir, known, unknown, relation):
    if len(dir) == 4:
        dir = dir[0:2] + "0" + dir[2:4]
    if len(dir) == 3:
        dir = dir[0:2] + "00" + dir[2]
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
    text_file3 = open("types.txt", "a")
    text_file3.write(dir+" "+relation)
    text_file3.close()

def print_ids(id1, id2):
    id1 = str(id1)
    id2 = str(id2)
    print(id1, id2)
    text_file4 = open("type_ids.txt", "a")
    text_file4.write(id1+"-"+id2+"\n")
    text_file4.close()

if __name__ == '__main__':
    main()