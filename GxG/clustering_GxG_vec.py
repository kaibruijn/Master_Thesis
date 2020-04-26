from gensim.models import Word2Vec
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


def vectorizer(sent, m):
    vex = []
    numw = 0
    for w in sent:
        try:
            if numw == 0:
                vec = m[w]
            else:
                vex = np.add(vec, m[w])
            numw+=1
        except:
            pass

    return np.asarray(vec)/numw

def main():
    num = 100
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

    sentences = Xtrain[:num]

    tok_sentences = []
    for sentence in sentences:
        tok_sentence = nltk.word_tokenize(sentence)
        tok_sentences.append(tok_sentence)

    sentences = tok_sentences

    m = Word2Vec(sentences, size=50, min_count=1, sg=1)


    l = []
    for i in sentences:
        l.append(vectorizer(i,m))

    X = np.array(l)

    clf = KMeans(n_clusters=2, init='k-means++', n_init=10, max_iter=300, tol=0.0001, precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=None)
    labels = clf.fit_predict(X)
    pca = PCA(n_components=2).fit(X)
    coords = pca.transform(X)

    for index, sentence in enumerate(sentences):
        print(str(labels[index]) + ":" + str(" ".join(sentence))[0:10]+": "+str(coords[index].tolist()[0])[0:6],",", str(coords[index].tolist()[1])[0:6])
        plt.text(coords[index].tolist()[0],coords[index].tolist()[1], str(dic[sentence[0]]) + str(labels[index]) + ":" + str(" ".join(sentence))[0:10], fontsize=7)

    label_colors=["red", "blue", "green", "yellow", "black", "purple", "cyan"]
    colors = [label_colors[i] for i in labels]
    plt.scatter(coords[:, 0], coords[:, 1], c=colors)
    centroids = clf.cluster_centers_
    centroid_coords = pca.transform(centroids)
    plt.scatter(centroid_coords[:, 0], centroid_coords[:, 1], marker="X", s=200, linewidth=2, c="#444d61")

    print(Counter(labels))

    genders = []
    for i,j in enumerate(dic):
        if i < (num+1):
            genders.append(dic[j])
    print(Counter(genders))

    plt.show()


if __name__ == '__main__':
    main()