#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
@author: shankartiwari
"""

import gc
import configparser
import pymongo

from db import *
from pca import *
from kmeans import *
from match_results import *
from utility import *


class Model:
    def __init__(self):
        self._config = None

    def _configure(self):
        self._config = configparser.ConfigParser()
        self._config.read('config.ini')

        client = pymongo.MongoClient('localhost', 27017)
        self._db = client['test-database']
        self._posts = self._db.posts
        self._match = '#ARSMUN'

    def _compute(self, startDate, endDate, seq):
        gettweets(self._posts, self._match, startDate, endDate, seq)
        index_id, documents, follower_count, retweet_count, hashtag_count, mention_count, \
        sentiment_pol, favorite_count, verify_count, screen_name, description = fetchdb(self._posts, self._match, startDate, endDate)
        gettophashtags(documents, self._match, seq)
        getstopwords(self._match, self._posts, startDate, endDate, seq)
        df, mergedf, feature_name = get_features(index_id, documents, follower_count, retweet_count,
                                                 hashtag_count, mention_count, sentiment_pol, favorite_count,
                                                 verify_count, screen_name, description)
        x_reduced, df = perform_pca(df, seq)
        mergedf = perform_kmeans(x_reduced, df, mergedf, feature_name, seq)
        getResults(mergedf, self._match)

        del df
        del mergedf
        gc.collect()  
        del gc.garbage[:]

    def main(self):
        self._configure()
        # getMatches(self._posts)
        logging.info("\nComputing tweet count per match........")
        logging.debug("\nComputing tweet count per match.......")
        getallmatchtweets(self._posts)
        # Before the match
        logging.info("\033[92m")
        logging.info("\nCalculating metrics: Before the match")
        logging.debug("\nCalculating metrics: Before the match")
        logging.info("\033[0m")
        found = False
        for line in open('pl.txt'):
            plmatch = line.strip().split(',')[0]
            plmatch = plmatch[1:-1]  # removes quotes
            if plmatch == self._match:
                match_date = line.strip().split(',')[1]
                yy = match_date.split('-')[2]
                dd = match_date.split('-')[1]
                mm = match_date.split('-')[0]
                before_endDate = datetime(int(yy), int(mm), int(dd))
                found = True
                break
        if not found:
            logging.info("\nNo such match found.")
            logging.debug("\nNo such match found.")
            sys.exit(1)
        before_startDate = before_endDate - timedelta(days=7)
        logging.info("\nStart Date: {} \nEnd Date: {}".format(before_startDate, before_endDate))
        logging.debug("\nStart Date: {} \nEnd Date: {}".format(before_startDate, before_endDate))
        self._compute(before_startDate, before_endDate, 1)
        # During the day of match
        logging.info("\033[92m")
        logging.info("\nCalculating metrics: During the match")
        logging.debug("\nCalculating metrics: During the match")
        logging.info("\033[0m")
        after_endDate = before_endDate + timedelta(days=2)
        # after_startDate = after_endDate - timedelta(days=1)
        logging.info("\nStart Date: {} \nEnd Date: {}".format(before_endDate, after_endDate))
        logging.debug("\nStart Date: {} \nEnd Date: {}".format(before_endDate, after_endDate))
        self._compute(before_endDate, after_endDate, 2)


if __name__ == '__main__':
    app = Model()
    sys.exit(app.main())
