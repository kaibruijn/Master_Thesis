#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basic clustering functionality with K-Means.
File provided for the assignment on clustering (IR course 2018/19)
"""

from nltk.tokenize import word_tokenize
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
import numpy as np
from os import listdir # to read files
from os.path import isfile, join # to read files
import sys
from scipy.special import comb #for rand index
from sklearn.cluster import AgglomerativeClustering #for hierarchical clustering
from nltk.corpus import stopwords #for ignoring stopwords
from nltk.stem.snowball import SnowballStemmer #for stemming

def tokenize(text):
    return word_tokenize(text)
    

def prepare_data(X, n_features=1000):
    vectorizer = TfidfVectorizer(tokenizer=tokenize, max_features=n_features)
    X_prep = vectorizer.fit_transform(X)
    return X_prep, vectorizer

    
def kmeans(X_prep, y, n_k):
    print("\n##### Running K-Means...")
    km = KMeans(init="random", n_clusters=n_k, verbose=False)
    km.fit_predict(X_prep)
    return km
    

def evaluate(model, y):
    print("\n##### Evaluating...")
    print("Contingency matrix")
    contingency_matrix = metrics.cluster.contingency_matrix(y, model.labels_)
    print (contingency_matrix)

    purity = np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)
    print("Purity %.3f" % purity)

    print("Adjusted rand-index: %.3f"
      % metrics.adjusted_rand_score(y, model.labels_))

def myComb(a,b):
    return comb(a,b,exact=True)

vComb = np.vectorize(myComb)
# TODO calculate the rand index
# Use the cluster IDs (model.labels_) and the categories (y)
def rand_index(model, y):
    print("\n##### Calculating rand index...")
    contingency_matrix = metrics.cluster.contingency_matrix(y, model.labels_)
    tp_plus_fp = vComb(contingency_matrix.sum(0, dtype=int),2).sum()
    tp_plus_fn = vComb(contingency_matrix.sum(1, dtype=int),2).sum()
    tp = vComb(contingency_matrix.astype(int), 2).sum()
    fp = tp_plus_fp - tp
    fn = tp_plus_fn - tp
    tn = comb(contingency_matrix.sum(), 2) - tp - fp - fn
    print((tp+tn)/(tp+fp+tn+fn))

#Code inspired by and altered from https://stackoverflow.com/questions/49586742/rand-index-function-clustering-performance-evaluation
    
# TODO complete this function
def top_terms_per_cluster(model, vectorizer, n_k, n_terms=10):
    print("\n##### Top terms per cluster...")
    sorted_centroids = model.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names()
    for i in range(len(sorted_centroids)):
        print("Cluster", i+1)
        for ind in sorted_centroids[i, :10]:
            print(terms[ind], end=" ")
            print()

    # TODO iterate over each cluster
    # then get the ids of the 10 top terms in the cluster from sorted_centroids
    # then get the terms that correspond to the term ids (from terms)


# TODO complete this function
def hierarchical_clustering(X, k=2):
    print("\n##### Running hierarchical clustering...")
    cluster = AgglomerativeClustering(n_clusters=k, linkage='average')  
    return cluster.fit(X.toarray())
    # TODO run hierarchical clustering and return the resulting model

def rss(model, y):
    print("\n##### Calculating RSS...")
    print(model.inertia_)


def main():
    num = 1832
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


    X = Xtrain[:num]
    categories = ["M", "F"]

    y = Ytrain[:num]
    X_prep, vectorizer = prepare_data(X)

    model = kmeans(X_prep, y, 2)
    evaluate(model, y)
    rand_index(model, y)
    top_terms_per_cluster(model, vectorizer, y)


if __name__ == '__main__':
    main()
