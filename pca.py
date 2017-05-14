#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
@author: shankartiwari
"""

from utility import *
from sklearn.decomposition import PCA
import numpy as np


def perform_pca(df):
        matplotlib.style.use('ggplot')
        logging.info("\n--------------------Performing dimension reduction of feature vector----------------")
        start_time_pca = time.time()
        pca = PCA().fit(df)

        exp_var = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4) * 100)
        logging.debug("\nExplained variance ratio:{}".format(exp_var))
        plotpca(exp_var, 'pca.png')
        clf = PCA(0.90)  # keep 90% of variance
        x_trans = clf.fit_transform(df)

        # Components are determined from above plot diagram
        # pca.n_components = 2
        # x_trans = pca.fit_transform(df)
        # print "\nReduced shape:"
        logging.info("\nReduced Shape:{}".format(x_trans.shape))
        logging.debug("\nReduced Shape:{}".format(x_trans.shape))
        col = list(range(0, x_trans.shape[1]))
        # Dump components relations with features:
        pca_df = pd.DataFrame(clf.components_, columns=df.columns, index=col)
        # pca_df = pd.DataFrame(pca.components_, columns=df.columns, index=col)
        pca_df.to_csv("pca_df.csv", encoding='utf-8')

        seconds = (time.time() - start_time_pca)
        logging.info("\nPCA time:{}".format(str(timedelta(seconds=seconds))))

        return x_trans, df
