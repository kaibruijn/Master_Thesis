from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd
import csv
from collections import Counter
import sys


def main():
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


    triplist = []
    for i in preddict:
        triplist.append((i, preddict[i], typedict[i]))

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
        print(i)
        if i[2] == "R":
            Rlist.append(float(i[1]))
        if i[2] == "S":
            Slist.append(float(i[1]))
        if i[2] == "T":
            Tlist.append(float(i[1]))
        if i[2] == "U":
            Ulist.append(float(i[1]))

    print("R:",(sum(Rlist)/len(Rlist)),min(Rlist), max(Rlist))
    print("S:",(sum(Slist)/len(Slist)),min(Slist), max(Slist))
    print("T:",(sum(Tlist)/len(Tlist)),min(Tlist), max(Tlist))
    print("U:",(sum(Ulist)/len(Ulist)),min(Ulist), max(Ulist))

    print(len(Tlist))
    print(len(Slist))
    print(len(Rlist))
    print(len(Ulist))

if __name__ == '__main__':
    main() 