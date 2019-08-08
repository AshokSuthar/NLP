import pandas as pd
import numpy as np
from gensim.models.word2vec import Word2Vec
from sklearn.cluster import KMeans
from sklearn.neighbors import KDTree
import multiprocessing;
from wordcloud import WordCloud, ImageColorGenerator
import matplotlib.pyplot as plt
from itertools import cycle
import sys

def clustering_on_wordvecs(word_vectors, num_clusters):
    # Initalize a k-means object and use it to extract centroids
    kmeans_clustering = KMeans(n_clusters = num_clusters, init='k-means++')
    idx = kmeans_clustering.fit_predict(word_vectors)
    
    return kmeans_clustering.cluster_centers_, idx

def get_top_words(index2word, k, centers, wordvecs):
    tree = KDTree(wordvecs)

    #Closest points for each Cluster center is used to query the closest 20 points to it.
    closest_points = [tree.query(np.reshape(x, (1, -1)), k=k) for x in centers]
    closest_words_idxs = [x[1] for x in closest_points]

    #Word Index is queried for each position in the above array, and added to a Dictionary.
    closest_words = {}
    for i in range(0, len(closest_words_idxs)):
        closest_words['Cluster #' + str(i+1).zfill(2)] = [index2word[j] for j in closest_words_idxs[i][0]]

    #A DataFrame is generated from the dictionary.
    df = pd.DataFrame(closest_words)
    df.index = df.index+1

    return df

def display_cloud(cluster_num, cmap):
    wc = WordCloud(background_color="black", max_words=2000, max_font_size=80, colormap=cmap)
    wordcloud = wc.generate(' '.join([word for word in top_words['Cluster #' + str(cluster_num).zfill(2)]]))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.savefig('cluster_' + str(cluster_num), bbox_inches='tight')


if __name__ == "__main__":
    model_path = "Models\intentModelArxiveWord2Vec"
    w2v_model= Word2Vec.load(model_path + ".model")
    Z = w2v_model.wv.syn0
    print(Z[0].shape)
    print(Z[0])
    centers, clusters = clustering_on_wordvecs(Z, 10)
    print(centers)
    centroid_map = dict(zip(w2v_model.wv.index2word, clusters))
    print()
    
    top_words = get_top_words(w2v_model.wv.index2word, 300, centers, Z)
    print("HEYYYYYYYYY")

    cmaps = cycle([
            'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
            'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg', 'hsv',
            'gist_rainbow', 'rainbow', 'jet', 'nipy_spectral', 'gist_ncar'])

    for i in range(10):
        col = next(cmaps)
        display_cloud(i+1, col)
