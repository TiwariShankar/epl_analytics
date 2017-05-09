#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
@author: shankartiwari
"""

from utility import *
import numpy as np


def perform_kmeans(X_reduced, df, mergedf, feature_name, seq):
    matplotlib.style.use('ggplot')
    logging.info(
        "\n--------------------------------Started KMeans-------------------------------------------")
    start_time_kmeans = time.time()

    # for determining k
    if X_reduced.shape[0] > 10000:
        plot_kmeans, x_data = calscore(X_reduced, 7)
    elif X_reduced.shape[0] < 500:
        plot_kmeans, x_data = calscore(X_reduced, 10)
    else:
        plot_kmeans, x_data = calscore(X_reduced, 10)

    # Gets the max sil value and corresponding cluster no.
    max_value = max(plot_kmeans)
    max_index = plot_kmeans.index(max_value) + 5

    if seq == 1:
        plotsil(x_data, plot_kmeans, 'kmeans_1.png')
    if seq == 2:
        plotsil(x_data, plot_kmeans, 'kmeans_2.png')

    # fit the model and calculates score
    if X_reduced.shape[0] > 10000:
        # for more than 10000 rows
        df, labels, centroids, kmeans_model = fitkmeans(df, X_reduced, max_index, 7, 1000)
    elif X_reduced.shape[0] < 500:
        # for less than 500 rows
        df, labels, centroids, kmeans_model  = fitkmeans(df, X_reduced, max_index, 10, 0)
    else:
        # for less than 5000 rows
        df, labels, centroids, kmeans_model  = fitkmeans(df, X_reduced, max_index, 10, 0)

    mergedf['cluster_id'] = labels
    mergedf.to_csv("combine_df.csv", encoding='utf-8')

    # removing duplicate screen names under same cluster id
    cluster_screenname = {
        k: v["screen_name"].tolist() for k,
        v in mergedf.groupby("cluster_id")}
    cluster_screenname = {k: list(set(v))
                          for k, v in cluster_screenname.items()}

    for k, v in cluster_screenname.items():
        logging.info("Length of cluster {}: {}".format(k, len(v)))

    # saves screen name and cluster id in csv file
    pd.DataFrame.from_dict(
        cluster_screenname,
        orient='index').T.to_csv(
        'ClusterID_ScreenName.csv',
        index=False,
        encoding='utf-8')
    logging.info('\n\nTop terms per cluster:\n')
    # np.linspace: Return evenly spaced numbers over a specified interval.
    color = iter(cm.rainbow(np.linspace(0, 1, max_index)))
    colors = [next(color) for i in range(max_index)]
    # sort cluster centers by proximity to centroid
    order_centroids = kmeans_model.cluster_centers_.argsort()[:, ::-1]
    # The dict() constructor builds dictionaries directly from sequences of key-value pairs
    col_map = dict(zip(set(labels), colors)) # key as labels and value as colors
    label_color = [col_map[l] for l in labels]

    if seq == 1:
        plotkmeans(X_reduced, label_color, centroids, 'clusters_1.png', max_index)
        # Top 30 words in each cluster for wordcloud
        generatewordcloud(max_index, order_centroids, feature_name, seq)

    if seq == 2:
        plotkmeans(X_reduced, label_color, centroids, 'clusters_2.png', max_index)
        # Top 30 words in each cluster for wordcloud
        generatewordcloud(max_index, order_centroids, feature_name, seq)

    seconds = (time.time() - start_time_kmeans)
    logging.info("KMeans Time:{}".format(str(timedelta(seconds=seconds))))

    return mergedf
