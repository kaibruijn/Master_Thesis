from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, auc
import numpy as np
import pandas as pd
from nltk.tokenize.treebank import TreebankWordDetokenizer
import csv
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from sklearn import svm, linear_model
from nltk.tag import PerceptronTagger
from nltk.corpus import alpino as alp
from nltk.tokenize import TweetTokenizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def preprocessor(x):
#    poslist=[]
#    for i in tagger.tag(x.split()):
#        for j in i:
#            poslist.append(j)
#    x=" ".join(poslist)
    return x

def print_n_most_informative_features(coefs, features, n):
    # Prints the n most informative features
    most_informative_feature_list = [(coefs[0][nr],feature) for nr, feature in enumerate(features)]
    sorted_most_informative_feature_list = sorted(most_informative_feature_list, key=lambda tup: abs(tup[0]), reverse=True)
    print("\nMOST INFORMATIVE FEATURES\n#\tvalue\tfeature")
    for nr, most_informative_feature in enumerate(sorted_most_informative_feature_list[:n]):
        print(str(nr+1) + ".","\t%.3f\t%s" % (most_informative_feature[0], most_informative_feature[1]))

def transform_data(data):
    data = data[["labels", "Text1", "Text2"]]
    data["text"] = data["Text1"].map(str) +"\n\n\n"+ data["Text2"]
    data = data[1:]
    data = data[["labels", "text"]]

    print(data)

    return data

def main():
    data = pd.read_excel("Final_RedSet-20DataTrain.xlsx", names=["labels", "User1", "User2", "Text1", "Text2"])
    dataframetrain = transform_data(data)
    
    data = pd.read_excel("Final_RedSet-20DataTest.xlsx", names=["labels", "User1", "User2", "Text1", "Text2"])
    dataframetest = transform_data(data)

    #dataframetrain = dataframetrain.sample(frac=1)
    Xtrain = dataframetrain['text'].tolist()
    Ytrain = dataframetrain['labels'].tolist()

    #dataframetest = dataframetest.sample(frac=1)
    Xtest = dataframetest['text'].tolist()
    Ytest = dataframetest['labels'].tolist()

    vec = TfidfVectorizer(preprocessor = preprocessor)


    classifier = Pipeline( [('vec', vec),
                            ('cls', linear_model.LinearRegression())] )

    classifier.fit(Xtrain, Ytrain)



    try:
        coefs = classifier.named_steps['cls'].coef_
        features = classifier.named_steps['vec'].get_feature_names()
        print_n_most_informative_features(coefs, features, 10)
        print()
    except:
        pass
    Yguess = classifier.predict(Xtest)
    Ylist = []
    for i in Yguess:
        print(i)
        if i < 0:
            Ylist.append(-1)
        else:
            Ylist.append(1)

    print(confusion_matrix(Ytest, Ylist))
    print(classification_report(Ytest, Ylist))
    print(accuracy_score(Ytest,Ylist))



if __name__ == '__main__':
    main()