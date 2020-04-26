from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, auc
import numpy as np
import pandas as pd
import csv
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn import svm, linear_model

def preprocessor(x):
    return x

def print_n_most_informative_features(coefs, features, n):
    # Prints the n most informative features
    most_informative_feature_list = [(coefs[0][nr],feature) for nr, feature in enumerate(features)]
    sorted_most_informative_feature_list = sorted(most_informative_feature_list, key=lambda tup: abs(tup[0]), reverse=True)
    print("\nMOST INFORMATIVE FEATURES\n#\tvalue\tfeature")
    for nr, most_informative_feature in enumerate(sorted_most_informative_feature_list[:n]):
        print(str(nr+1) + ".","\t%.3f\t%s" % (most_informative_feature[0], most_informative_feature[1]))


def main():
    Xtrain = []
    Ytrain = []
    with open("trainGxG/GxG_News.txt") as txt:
        for line in txt:
            if line[0:8] == "<doc id=":
                Ytrain.append(line.split()[3][8])
                string=[]
            elif line[0:6] == "</doc>":
                Xtrain.append(" ".join(string))
            else:
                string.append(line)

    Xtest = []
    with open("testGxG/GxG_News.txt") as txt:
        for line in txt:
            if line[0:8] == "<doc id=":
                string=[]
            elif "</doc>" in line:
                Xtest.append(" ".join(string))
            else:
                string.append(line)

    Ytest = []
    with open("testGxG/GxG_News_gold.txt") as text:
        for line in text:
            Ytest.append(line.split()[1])


    vec = TfidfVectorizer(preprocessor = preprocessor)


    classifier = Pipeline( [('vec', vec),
                            ('cls', svm.LinearSVC(C=1))] )

    classifier.fit(Xtrain, Ytrain)
    try:
        coefs = classifier.named_steps['cls'].coef_
        features = classifier.named_steps['vec'].get_feature_names()
        print_n_most_informative_features(coefs, features, 10)
        print()
    except:
        pass
    Yguess = classifier.predict(Xtest)

    print(classification_report(Ytest, Yguess))
    print(accuracy_score(Ytest,Yguess))
    print(confusion_matrix(Ytest, Yguess))


if __name__ == '__main__':
    main()