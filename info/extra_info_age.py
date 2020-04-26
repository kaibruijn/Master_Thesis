from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd
import csv
from collections import Counter
import sys
import xlsxwriter


def main():
    data = pd.read_csv("../Twin-20/TwinDataIDs.csv")

    preddict = {}
    with open("../answers.txt") as txt:
        for line in txt:
            line = line.strip()
            preddict[line.split()[0][-5:]]=line.split()[1]


    typedict = {}
    with open("../Twin-20/types.txt") as txt:
        for line in txt:
            line = line.strip()
            typedict[line.split()[0][-5:]]=line.split()[1]


    idlist = []
    with open("../Twin-20/type_ids.txt") as txt:
        for line in txt:
            line = line.strip()
            idlist.append(line)


    triplist = []
    for j,i in enumerate(preddict):

        first = idlist[j].split("-")[0]
        second = idlist[j].split("-")[1]
        triplist.append((i, preddict[i], typedict[i], idlist[j], data[data["ID"]==int(first)]["Age"].tolist()[0], data[data["ID"]==int(second)]["Age"].tolist()[0], data[data["ID"]==int(first)]["Gender"].tolist()[0], data[data["ID"]==int(second)]["Gender"].tolist()[0], data[data["ID"]==int(first)]["Text"].tolist()[0][:50], data[data["ID"]==int(second)]["Text"].tolist()[0][:50]))


    allist = []
    for i in triplist:
        allist.append(i[2])

    samelist = []
    for i in triplist:
        if float(i[1]) > 0.5:
            samelist.append(i[2])

    print(Counter(samelist))
    print(Counter(allist))

    Rlist = []
    Slist = []
    Tlist = []
    Ulist = []

    for i in triplist:
        if i[2] == "R" and i[4] != "-" and i[5] != "-" and abs(int(i[4])-int(i[5])) < 10:
            print(i)
            Rlist.append(float(i[1]))
        if i[2] == "S":
            #print(i)
            Slist.append(float(i[1]))
        if i[2] == "T":
            #print(i)
            Tlist.append(float(i[1]))
        if i[2] == "U":
            #print(i)
            Ulist.append(float(i[1]))

    print("T:",(sum(Tlist)/len(Tlist)),min(Tlist), max(Tlist))
    print("S:",(sum(Slist)/len(Slist)),min(Slist), max(Slist))
    print("R:",(sum(Rlist)/len(Rlist)),min(Rlist), max(Rlist))
    print("U:",(sum(Ulist)/len(Ulist)),min(Ulist), max(Ulist))


if __name__ == '__main__':
    main() 