#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
@author: shankartiwari
"""

import configparser
import pymongo

from utility import *
from db import *
from pca import *
from kmeans import *
from match_results import *


class Model:
    def __init__(self):
        self._config = None

    def _configure(self):
        self._config = configparser.ConfigParser()
        self._config.read('config.ini')

        client = pymongo.MongoClient('localhost', 27017)
        self._db = client['test-database']
        self._posts = self._db.posts
        self._match = ''
        self._result = 0

    def _compute(self, startDate, endDate, model):
        # gettweets(self._posts, self._match, startDate, endDate, seq)
        index_id, documents, follower_count, retweet_count, hashtag_count, mention_count, sentiment_pol, favorite_count, \
                          verify_count, screen_name, description = fetchdb(self._posts, self._match, startDate, endDate)
        # gettophashtags(documents, self._match)
        # getstopwords(self._match, self._posts, startDate, endDate)
        df, mergedf, feature_name = get_features(index_id, documents, follower_count, retweet_count,
                                                 hashtag_count, mention_count, sentiment_pol, favorite_count,
                                                 verify_count, screen_name, description)
        if df.empty:
            return
        x_reduced, df = perform_pca(df)
        mergedf = perform_kmeans(x_reduced, df, mergedf, feature_name)
        saveResults(mergedf, self._match, self._result, model)

    def _testData(self):
        match_date = '05-14-2017'
        self._match = '#TOTMUN'
        logging.info("\033[92m")
        logging.info("\nComputing tweet count of match: {}".format(self._match))
        logging.debug("\nComputing tweet count of match: {}".format(self._match))
        logging.info("\033[0m")
        yy = match_date.split('-')[2]
        dd = match_date.split('-')[1]
        mm = match_date.split('-')[0]
        before_endDate = datetime(int(yy), int(mm), int(dd))
        before_startDate = before_endDate - timedelta(days=7)
        before_endDate = before_endDate + timedelta(days=2)
        logging.info("\nStart Date: {} \nEnd Date: {}".format(before_startDate, before_endDate))
        logging.debug("\nStart Date: {} \nEnd Date: {}".format(before_startDate, before_endDate))
        self._compute(before_startDate, before_endDate, "test")     

    def main(self):
        self._configure()
        # self._testData()
        rfmodel()
        # getMatches(self._posts)
        # getallmatchtweets(self._posts)
        # found = False
        # for line in open('pl.txt'):
        #     plmatch = line.strip().split(',')[0]
        #     plmatch = plmatch[1:-1]  # removes quotes
        #     self._match = plmatch
        #     if len(line.strip().split(',')) == 2:
        #             continue
        #     logging.info("\033[92m")
        #     logging.info("\nComputing tweet count of match: {}".format(self._match))
        #     logging.debug("\nComputing tweet count of match: {}".format(self._match))
        #     logging.info("\033[0m")
        #     match_date = line.strip().split(',')[1]
        #     self._result = line.strip().split(',')[2]
        #     yy = match_date.split('-')[2]
        #     dd = match_date.split('-')[1]
        #     mm = match_date.split('-')[0]
        #     before_endDate = datetime(int(yy), int(mm), int(dd))
        #     found = True
        #     if not found:
        #         logging.info("\nNo such match found.")
        #         logging.debug("\nNo such match found.")
        #         sys.exit(1)
        #     if os.path.exists('data.csv'):
        #        pd_df_train = pd.read_csv('data.csv', sep='\t', header=None)
        #        lst_matches = pd_df_train.iloc[:, 0].tolist()
        #        if any(self._match in s for s in lst_matches):
        #             logging.info("\nResults are already evaluated for the match: {}".format(self._match))
        #             continue
        #     before_startDate = before_endDate - timedelta(days=7)
        #     before_endDate = before_endDate + timedelta(days=2)
        #     logging.info("\nStart Date: {} \nEnd Date: {}".format(before_startDate, before_endDate))
        #     logging.debug("\nStart Date: {} \nEnd Date: {}".format(before_startDate, before_endDate))
        #     self._compute(before_startDate, before_endDate, "train")            
        # self._testData()


if __name__ == '__main__':
    app = Model()
    sys.exit(app.main())
