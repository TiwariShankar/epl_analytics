#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
@author: shankartiwari
"""

from utility import *
import numpy as np


def perform_kmeans(X_reduced, df, mergedf, feature_name):
    matplotlib.style.use('ggplot')
    logging.info(
        "\n--------------------------------Started KMeans-------------------------------------------")
    start_time_kmeans = time.time()

    # for determining k
    if X_reduced.shape[0] > 10000:
        plot_kmeans, x_data = calscore(X_reduced, 7)
    elif X_reduced.shape[0] < 500:
        plot_kmeans, x_data, max_index = calscore(X_reduced, 10)
    else:
        plot_kmeans, x_data, max_index = calscore(X_reduced, 10)

    # plotsil(x_data, plot_kmeans, 'kmeans.png')
    # fit the model and calculates score
    if X_reduced.shape[0] > 10000:
        # for more than 10000 rows
        df, labels, centroids, kmeans_model = fitkmeans(df, X_reduced, max_index, 7, 1000)
    elif X_reduced.shape[0] < 500:
        # for less than 500 rows
        df, labels, centroids, kmeans_model = fitkmeans(df, X_reduced, max_index, 10, 0)
    else:
        # for less than 5000 rows
        df, labels, centroids, kmeans_model = fitkmeans(df, X_reduced, max_index, 10, 0)

    # df['cluster_id'] = labels
    mergedf['cluster_id'] = labels
    # saves entire result in csv
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
    col_map = dict(zip(set(labels), colors))  # key as labels and value as colors
    label_color = [col_map[l] for l in labels]

    # plotkmeans(X_reduced, label_color, centroids, 'clusters.png', max_index, labels, col_map)
    # Top 30 words in each cluster for wordcloud
    # generatewordcloud(max_index, order_centroids, feature_name)

    seconds = (time.time() - start_time_kmeans)
    logging.info("KMeans Time:{}".format(str(timedelta(seconds=seconds))))

    return mergedf
