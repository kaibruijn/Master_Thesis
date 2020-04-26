from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import pandas as pd

def main():

    data = pd.read_csv("../Twin-20/TwinDataIDs.csv")

    sentences= data["Text"].tolist()

    documents = sentences[10:]

    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(documents)

    true_k = 2
    model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
    model.fit(X)

    print("Top terms per cluster:")
    order_centroids = model.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names()
    for i in range(true_k):
        print("Cluster %d:" % i),
        for ind in order_centroids[i, :10]:
            print(' %s' % terms[ind]),
        print

    print("\n")
    print("Prediction")

    for i in range(10):

        Y = vectorizer.transform([sentences[i]])
        prediction = model.predict(Y)
        print(prediction)


if __name__ == '__main__':
    main()