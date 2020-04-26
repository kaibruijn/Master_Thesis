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


def main():
    data = pd.read_csv("../Twin-20/TwinDataIDs.csv")
    data2 = pd.read_csv("../Twin-20/TwinIDs.csv")

    sentences= data["Text"].tolist()
    sentences2 = sentences

    senlen = []
    for i in sentences:
        print(len(i))
        senlen.append(len(i))

    print("average char:",sum(senlen)/len(senlen),"min char:", min(senlen),"max char", max(senlen))
    ids = data["ID"].tolist()
    dictje = {}

    for num, idtje in enumerate(ids):
        dictje[sentences[num]] = idtje

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

    n_clusters = 2
    clf = KMeans(n_clusters = n_clusters, max_iter=500, init="k-means++", n_init=1)
    labels = clf.fit_predict(X)
    pca = PCA(n_components=2).fit(X)
    coords = pca.transform(X)

    for index, sentence in enumerate(sentences):
        print(str(labels[index]) + ":" + str(" ".join(sentence))[0:10]+": "+str(coords[index].tolist()[0])[0:6],",", str(coords[index].tolist()[1])[0:6])

    label_colors = ["#2AB0E9", "#2BAF74", "#D7665E", "#CCCCCC", "#D2CA0D", "#522A64", "#A3DB05", "#FC6514"]
    colors = [label_colors[i] for i in labels]
    plt.scatter(coords[:, 0], coords[:, 1])
    centroids = clf.cluster_centers_
    centroid_coords = pca.transform(centroids)
    plt.scatter(centroid_coords[:, 0], centroid_coords[:, 1], marker="X", s=200, linewidth=2, c="#444d61")

    for index, sentence in enumerate(sentences):
        for i in sentences2:
            if nltk.word_tokenize(i) == sentence:
                idtje = dictje[i]
                try:
                    twin_id = data2.loc[data2["Twin1_ID"]==idtje]["Twin2_ID"].tolist()[0]
                    plt.text(coords[index].tolist()[0],coords[index].tolist()[1], str(labels[index]) + ":" + str(" ".join(sentence))[0:10]+ " " + str(idtje) + "+" + str(twin_id), fontsize=7)
                except:
                    try:
                        twin_id = data2.loc[data2["Twin2_ID"]==str(idtje)]["Twin1_ID"].tolist()[0]
                        plt.text(coords[index].tolist()[0],coords[index].tolist()[1], str(labels[index]) + ":" + str(" ".join(sentence))[0:10]+ " " + str(idtje) + "+" + str(twin_id), fontsize=7)
                    except:
                        plt.text(coords[index].tolist()[0],coords[index].tolist()[1], str(labels[index]) + ":" + str(" ".join(sentence))[0:10]+ " " + str(idtje), fontsize=7)

    plt.show()

    print(Counter(labels))

    lenlist = []
    len2list = []

    for i in tok_sentences:
        lenlist.append(len(i))
        len2list.append(len(list(set(i))))

    print("average tokens:",sum(lenlist)/len(lenlist),"min tokens:", min(lenlist),"max tokens", max(lenlist))
    print("average types:",sum(len2list)/len(len2list),"min types:", min(len2list),"max types", max(len2list))


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


if __name__ == '__main__':
    main()