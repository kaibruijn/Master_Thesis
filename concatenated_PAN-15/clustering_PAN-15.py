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


def main():

    datatrain = pd.read_excel("../PAN-15/logregexcel_PAN-15trainlargeconcatenated.xlsx", names=["Folder", "labels", "Text1", "Text2"])
    dataframetrain = transform_data(datatrain)
    #dataframetrain = dataframetrain.sample(frac=1)
    Xtrain = dataframetrain['text'].tolist()
    Ytrain = dataframetrain['labels'].tolist()

    datatest = pd.read_excel("../PAN-15/logregexcel_PAN-15testlargeconcatenated.xlsx", names=["Folder", "labels", "Text1", "Text2"])
    dataframetest = transform_data(datatest)
    #dataframetest = dataframetest.sample(frac=1)
    Xtest = dataframetest['text'].tolist()
    Ytest = dataframetest['labels'].tolist()

    vec = TfidfVectorizer(preprocessor = preprocessor)


    classifier = Pipeline( [('vec', vec),
                            ('cls', KMeans(n_clusters=2))] )

    classifier.fit(Xtrain, Ytrain)

    try:
        X_prep = vec.fit_transform(Xtest).toarray()
        labels = classifier.fit_predict(Xtest)
        pca = PCA(n_components=2).fit(X_prep)
        coords = pca.transform(X_prep)
        label_colors=["red", "blue", "green", "yellow", "black", "purple", "cyan"]
        colors = [label_colors[i] for i in labels]
        plt.scatter(coords[:, 0], coords[:, 1], c=colors)
        centroids = classifier.named_steps['cls'].cluster_centers_
        centroid_coords = pca.transform(centroids)
        plt.scatter(centroid_coords[:, 0], centroid_coords[:, 1], marker="X", s=200, linewidth=2, c="#444d61")
        plt.show()
    except:
        pass

    try:
        coefs = classifier.named_steps['cls'].coef_
        print(coefs)
        features = classifier.named_steps['vec'].get_feature_names()
        print_n_most_informative_features(coefs, features, 10)
        print()
    except:
        pass
    Yguess = classifier.predict(Xtest)
    Ylist = []
    for i in Yguess:
        if i < 0.5:
            Ylist.append(0)
        else:
            Ylist.append(1)

    print(classification_report(Ytest, Ylist))
    print(accuracy_score(Ytest,Ylist))


def transform_data(data):
    data = data[["labels", "Text1", "Text2"]]
    data["text"] = data["Text1"].map(str) +"\n\n\n"+ data["Text2"]
    data = data[1:]
    data = data[["labels", "text"]]
    transformer_dict = {-1: 0, 1: 1}
    data['labels'] = data['labels'].apply(lambda x: transformer_dict[x])

    return data


if __name__ == '__main__':
    main()