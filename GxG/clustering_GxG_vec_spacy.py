import spacy
import numpy
import nltk
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import cluster
from sklearn import metrics
from sklearn.decomposition import PCA
from scipy.cluster import hierarchy
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from collections import Counter
from nltk.tag import PerceptronTagger
from nltk.corpus import alpino as alp

def preprocess(x):
    #poslist=[]
    #for i in tagger.tag(x.split()):
    #    for j in i:
    #        poslist.append(j)
    #poslist.append(str(len(poslist)))  
    #x=" ".join(poslist)
    return x


def main():

    training_corpus = list(alp.tagged_sents())
    global tagger
    tagger = PerceptronTagger()
    tagger.train(training_corpus)
    num = 2138
    dic = {}

    Xtrain = []
    Ytrain = []
    with open("trainGxG/GxG_News.txt") as txt:
        for line in txt:
            if line[0:8] == "<doc id=":
                Ytrain.append(line.split()[3][8])
                string=[line.split('\"')[1]]
                dic[line.split('\"')[1]] = line.split()[3][8]
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

    sentences = []
    for i in Xtrain[:num]:
        sentences.append(preprocess(i))


    nlp = spacy.load('nl_core_news_sm')
    veclist = []

    for sentence in sentences:
        doc = nlp(sentence)
        vec = doc.vector 
        veclist.append(vec)

    X = np.array(veclist)

    clf = KMeans(n_clusters=2, init='k-means++', n_init=10, max_iter=300, tol=0.0001, precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=None)
    labels = clf.fit_predict(X)
    pca = PCA(n_components=2).fit(X)
    coords = pca.transform(X)

    lst = []

    for index, sentence in enumerate(sentences):
        plt.text(coords[index].tolist()[0],coords[index].tolist()[1], str(dic[sentence.split()[0]]) + str(labels[index]) + ":" + str(sentence)[0:10], fontsize=4)
        lst.append(str(dic[sentence.split()[0]]) + str(labels[index]))

    label_colors=["red", "blue", "green", "yellow", "black", "purple", "cyan"]
    colors = [label_colors[i] for i in labels]
    plt.scatter(coords[:, 0], coords[:, 1], c=colors)
    centroids = clf.cluster_centers_
    centroid_coords = pca.transform(centroids)
    plt.scatter(centroid_coords[:, 0], centroid_coords[:, 1], marker="X", s=200, linewidth=2, c="#444d61")

    print(Counter(labels))

    genders = []
    for i,j in enumerate(sentences):
        if i < num:
            genders.append(dic[j.split()[0]])
    print(Counter(genders))
    print(Counter(lst))
    plt.show()

if __name__ == '__main__':
    main()