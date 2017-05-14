#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
@author: shankartiwari
"""

import logging
import os
import seaborn as sns
import sys
import time
import pytz
from datetime import timedelta
from datetime import datetime
import matplotlib.dates as md
import pandas as pd
import json
import string
import re
import operator
import matplotlib
import matplotlib.cm as cm
from matplotlib.ticker import FuncFormatter, MultipleLocator
import calendar
from scipy.sparse import hstack
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler, normalize, MinMaxScaler
from sklearn.metrics.pairwise import euclidean_distances
from replacers import RegexpReplacer, RepeatReplacer
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.cluster import KMeans
from sklearn import metrics
from wordcloud import WordCloud, STOPWORDS
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import matplotlib.pyplot as plt
from collections import defaultdict
import collections

sns.set(color_codes=True)
sns.despine()
sns.set_style("ticks")
sns.set_palette(sns.color_palette("Blues_r"))

log_dir = '/Cluster_Epl/Log/'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
filename = 'model.log'
log_path = os.path.join(log_dir, filename)

# set up logging to file - see previous section for more details
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename=log_path,
                    filemode='w+')

console = logging.StreamHandler()
console.setLevel(logging.INFO)

formatter = logging.Formatter('%(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

bowmatrix_count = []
followercount_features = []
retweetcount_features = []
hashtagcount_features = []
mentioncount_features = []
pol_count = []
fav_count = []
ver_count = []

matplotlib.style.use('ggplot')


def getMatches(posts):
    cur = posts.find({}, {'match': 1, '_id': 0})
    records = list(cur)
    matches = []
    for row in records:
        if row['match'] not in matches:
            matches.append(row['match'])
    logging.info("\nMatches: {}".format(matches))
    logging.debug("\nMatches: {}".format(matches))


def getallmatchtweets(posts):
    start_time = time.time()
    plot_match_tweets = {}
    for line in open('pl.txt'):
        match = line.strip().split(',')[0]
        match = match[1:-1]  # removes quotes
        count = posts.find({'match': match}).count()
        plot_match_tweets[match] = count
    seconds = (time.time() - start_time)
    logging.info("Time:{}".format(str(timedelta(seconds=seconds))))
    dict_len = len(plot_match_tweets)
    fig, ax = plt.subplots()
    # range(len(plot_match_tweets) -> returns range
    # plot_match_tweets.values() -> returns height
    x_pos = range(len(plot_match_tweets))
    plt.bar(x_pos, plot_match_tweets.values())
    plt.xticks(x_pos, list(plot_match_tweets.keys()), rotation=70)
    ax.set_xlim(0, dict_len)
    plt.setp(ax.get_xticklabels(), fontsize=2)
    plt.ylabel('Count')
    plt.xlabel('Matches')
    plt.title('Tweets count for every match', y=1.03)
    plt.savefig('all_tweet_count.png', dpi=800)
    plt.close()


def gettweets(posts, match, startDate, endDate):
    # cursor = posts.find({'match': match}, {'_id': 0, 'date': 1})
    cursor = posts.find({'$and': [{'match': match},
                                  {'date': {'$gte': startDate, '$lte': endDate}}]},
                                  {'date': 1, '_id': 0})

    if cursor.count() == 0:
        logging.info("Tweets doesn't exist for the given match")
        sys.exit(1)
    dict_date = {}
    for rec in cursor:
        x = calendar.timegm(rec['date'].utctimetuple())
        if x not in dict_date:
            dict_date[x] = 0
        else:
            dict_date[x] += 1
    df_object = pd.DataFrame(dict_date.items())
    df_object.columns = ['date', 'count']
    df_object['date'] = pd.to_datetime(df_object['date'], unit='s')
    df_object = df_object.sort_values(by='date')

    plt.xticks(rotation=25)
    f, ax = plt.subplots()
    ax.plot(df_object['date'], df_object['count'])
    plt.ylabel('Count')
    plt.xlabel('Date')
    plt.savefig('Tweets Count.png', dpi=800)
    plt.close()


def getrecords(posts, startDate, endDate, match):
    cursor_records = posts.find({'$and': [{'match': match},
                                         {"date": {"$gte": startDate,
                                                   "$lte": endDate}}]},
                               {'doc': 1,
                                'hashtag_count': 1,
                                'mention_count': 1,
                                'sent_pol': 1,
                                'retweetCount': 1,
                                'verified': 1,
                                'screen_name': 1,
                                'description': 1,
                                'followers_count': 1,
                                'favorite_count': 1,
                                'date': 1,
                                '_id': 1})
    # cursor_before = posts.find({'$and': [{'$or': [{'match': match},
    #                                               {'match': '#MUFC'},
    #                                               {'match': '#AFC'}]},
    #                                      {"date": {"$gte": startDate,
    #                                                "$lt": endDate}}]},
    #                            {'doc': 1,
    #                             'hashtag_count': 1,
    #                             'mention_count': 1,
    #                             'sent_pol': 1,
    #                             'retweetCount': 1,
    #                             'verified': 1,
    #                             'screen_name': 1,
    #                             'description': 1,
    #                             'followers_count': 1,
    #                             'favorite_count': 1,
    #                             'date': 1,
    #                             '_id': 1})
    # print len(list(cursor_records))
    # sys.exit(1)
    return cursor_records


def fetchdb(posts, match, startDate, endDate):
    index_id = []
    documents = []
    hashtag_count = []
    mention_count = []
    sentiment_pol = []
    retweet_count = []
    follower_count = []
    favorite_count = []
    verify_count = []
    screen_name = []
    description = []
    cursor_records = getrecords(posts, startDate, endDate, match)
    records = list(cursor_records)

    # keeps unique row and sequence
    done = set()
    result = []
    for d in records:
        if d['doc'] not in done:
            done.add(d['doc'])
            result.append(d)

    exp = {
        "ain't": "am not",
        "aren't": "are not",
        "can't": "cannot",
        "can't've": "cannot have",
        "'cause": "because",
        "c'mon": "common",
        "could've": "could have",
        "couldn't": "could not",
        "couldn't've": "could not have",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "hadn't": "had not",
        "hadn't've": "had not have",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he would",
        "he'd've": "he would have",
        "he'll": "he will",
        "he'll've": "he will have",
        "he's": "he is",
        "how'd": "how did",
        "how'd'y": "how do you",
        "how'll": "how will",
        "how's": "how is",
        "i'd": "i would",
        "i'd've": "i would have",
        "i'll": "i will",
        "i'll've": "i will have",
        "i'm": "i am",
        "i've": "i have",
        "isn't": "is not",
        "it'd": "it would",
        "it'd've": "it would have",
        "it'll": "it will",
        "it'll've": "it will have",
        "it's": "it is",
        "let's": "let us",
        "ma'am": "madam",
        "mayn't": "may not",
        "might've": "might have",
        "mightn't": "might not",
        "mightn't've": "might not have",
        "must've": "must have",
        "mustn't": "must not",
        "mustn't've": "must not have",
        "needn't": "need not",
        "needn't've": "need not have",
        "o'clock": "of the clock",
        "oughtn't": "ought not",
        "oughtn't've": "ought not have",
        "shan't": "shall not",
        "sha'n't": "shall not",
        "shan't've": "shall not have",
        "she'd": "she would",
        "she'd've": "she would have",
        "she'll": "she will",
        "she'll've": "she will have",
        "she's": "she is",
        "should've": "should have",
        "shouldn't": "should not",
        "shouldn't've": "should not have",
        "so've": "so have",
        "so's": "so as",
        "that'd": "that would",
        "that'd've": "that would have",
        "that's": "that is",
        "there'd": "there would",
        "there'd've": "there would have",
        "there's": "there is",
        "they'd": "they would",
        "they'd've": "they would have",
        "they'll": "they will",
        "they'll've": "they will have",
        "they're": "they are",
        "they've": "they have",
        "to've": "to have",
        "wasn't": "was not",
        "we'd": "we would",
        "we'd've": "we would have",
        "we'll": "we will",
        "we'll've": "we will have",
        "we're": "we are",
        "we've": "we have",
        "weren't": "were not",
        "what'll": "what will",
        "what'll've": "what will have",
        "what're": "what are",
        "what's": "what has",
        "what've": "what have",
        "when's": "when is",
        "when've": "when have",
        "where'd": "where did",
        "where's": "where is",
        "where've": "where have",
        "who'll": "who will",
        "who'll've": "who will have",
        "who's": "who is",
        "who've": "who have",
        "why's": "why is",
        "why've": "why have",
        "will've": "will have",
        "won't": "will not",
        "won't've": "will not have",
        "would've": "would have",
        "wouldn't": "would not",
        "wouldn't've": "would not have",
        "y'all": "you all",
        "y'all'd": "you all would",
        "y'all'd've": "you all would have",
        "y'all're": "you all are",
        "y'all've": "you all have",
        "you'd": "you would",
        "you'd've": "you would have",
        "you'll": "you will",
        "you'll've": "you will have",
        "you're": "you are",
        "you've": "you have"
    }

    for rec in result:
        index_id.append(rec['_id'])
        documents.append(rec['doc'])
        hashtag_count.append(rec['hashtag_count'])
        mention_count.append(rec['mention_count'])
        sentiment_pol.append(rec['sent_pol'])
        follower_count.append(rec['followers_count'])
        retweet_count.append(rec['retweetCount'])
        favorite_count.append(rec['favorite_count'])
        verify_count.append(rec['verified'])
        screen_name.append(rec['screen_name'])
        description.append(rec['description'])

    mod_doc = []
    for sent in documents:
        for word in sent.split():
            if word in exp:
                sent = sent.replace(word, exp[word])
        # sent.replace('amp', '')
        mod_doc.append(sent)

    return index_id, mod_doc, follower_count, retweet_count, hashtag_count, \
        mention_count, sentiment_pol, favorite_count, verify_count, \
        screen_name, description


def gettophashtags(documents, match):
    toptags = {}
    match = match.lower().strip()
    if len(documents) > 0:
        for tweet in documents:
            tweet = tweet.split()
            for word in tweet:
                word = word.strip()
                if word != match and word.startswith("#"):
                    if word in toptags:
                        toptags[word] += 1
                    else:
                        toptags[word] = 0
        dict_filt_zeros = {k: v for k, v in toptags.items() if v != 0}
        dict_top = dict(
            sorted(
                dict_filt_zeros.iteritems(),
                key=operator.itemgetter(1),
                reverse=True)[
                :30])
        dict_len = len(dict_top)
        fig, ax = plt.subplots()
        plt.bar(range(len(dict_top)), dict_top.values())
        plt.xticks(
            range(
                len(dict_top)), list(
                dict_top.keys()), rotation='vertical')
        ax.set_xlim(0, dict_len)
        plt.setp(ax.get_xticklabels(), fontsize=4)
        plt.title("Top tags", y=1.03)
        plt.savefig('Top_tags.png', dpi=800)
        plt.close()
        logging.debug(
            "\n--------------------------------Top tags:-----------------------------------------")
        logging.debug("\nTop tags: {}".format(toptags))


def getstopwords(match, posts, startDate, endDate):
    cursor_records = posts.find({'$and': [{'match': match},
                                          {"date": {"$gte": startDate,
                                                    "$lt": endDate}}]},
                                {'tweetText': 1,
                                 '_id': 0})
    # cursor_records = posts.find({'$and': [{'$or': [{'match': match},
    #                                                {'match': '#MUFC'},
    #                                                {'match': '#AFC'}]},
    #                                       {"date": {"$gte": startDate,
    #                                                 "$lt": endDate}}]},
    #                             {'tweetText': 1,
    #                              '_id': 0})
    done = set()
    tweets = []
    for d in cursor_records:
        if isinstance(d['tweetText'], list):
            d['tweetText'] = " ".join(d['tweetText'])
        if d['tweetText'] not in done:
            done.add(d['tweetText'])
            tweets.append(d['tweetText'])
    stopset = set(stopwords.words('english'))
    prefixes = ('rt', '//', 'http', 'https', '\\', '#rt', ':')
    stopwords_list = {}
    for tweet in tweets:
        token_split = tweet.split(' ')
        # removes prefixes
        tweet_tokenize = [token for token in token_split if not
                          token.startswith(prefixes) and len(token) != 1]
        # removes digit
        tweet_tokenize = [x for x in tweet_tokenize if not any(
            c.isdigit() for c in x)]
        # Removes punctuation
        tweet_tokenize = [
            i for i in tweet_tokenize if i not in string.punctuation]
        lmtzr = WordNetLemmatizer()
        tweet_tokenize = [lmtzr.lemmatize(token) for token in tweet_tokenize]
        for token in tweet_tokenize:
            if token in stopset:
                if token in stopwords_list:
                    stopwords_list[token] += 1
                else:
                    stopwords_list[token] = 0
    # Set up the axes and figure
    stp_len = len(stopwords_list)
    fig, ax = plt.subplots()
    plt.bar(
        range(
            len(stopwords_list)),
        stopwords_list.values())
    plt.xticks(
        range(
            len(stopwords_list)),
        list(stopwords_list.keys()),
        rotation='vertical')
    # Set the x-axis limit
    ax.set_xlim(0, stp_len)
    plt.setp(ax.get_xticklabels(), fontsize=4)
    plt.title("Stop words", y=1.03)
    plt.savefig('Stop_words.png', dpi=800)
    plt.close()


def dictvecfeature(feature, featureName):
    # This transformer turns lists of mappings (dict-like objects)
    # of feature names to feature values into Numpy arrays
    for count in feature:
        if featureName == "bow_matrix_count":
            addit_feature = {featureName: count}
            bowmatrix_count.append(addit_feature)
        elif featureName == "follower_count":
            addit_feature = {featureName: count}
            followercount_features.append(addit_feature)
        elif featureName == "retweet_count":
            addit_feature = {featureName: count}
            retweetcount_features.append(addit_feature)
        elif featureName == "hashtag_count":
            addit_feature = {featureName: count}
            hashtagcount_features.append(addit_feature)
        elif featureName == "mention_count":
            addit_feature = {featureName: count}
            mentioncount_features.append(addit_feature)
        elif featureName == "sentiment_pol":
            addit_feature = {featureName: count}
            pol_count.append(addit_feature)
        elif featureName == "favorite_count":
            addit_feature = {featureName: count}
            fav_count.append(addit_feature)
        elif featureName == "verify_count":
            addit_feature = {featureName: count}
            ver_count.append(addit_feature)


def get_features(index_id, documents, follower_count, retweet_count, hashtag_count, mention_count, sentiment_pol, favorite_count, verify_count, screen_name, description):
    logging.info(
        "\n--------------------------------Feature Info:---------------------------------------")
    # max_features: If not None, build a vocabulary that only consider
    # the top max_features ordered by term
    # frequency across the corpus.

    # min_df it is ignored. 0.2 here indicates 20 % . i.e. if a feature does
    # exist not in 20 % of the document, it will be discarded.
    if len(documents) == 0 or len(documents) <= 50:
        logging.info("No such match present or few data for match")
        return pd.DataFrame(), "", ""
    elif len(documents) > 5000:
        # max_features = 1000
        countVec = CountVectorizer(min_df=0.005, stop_words='english')
    elif len(documents) < 500:
        # max_features = 50
        countVec = CountVectorizer(min_df=0.015, stop_words='english')
    else:
        # max_features = 300
        countVec = CountVectorizer(min_df=0.005, stop_words='english')
    # countVec = CountVectorizer(max_features=max_features, stop_words='english')
    bow_matrix = countVec.fit_transform(documents)
    bow_matrix_count = bow_matrix.toarray().sum(axis=1)  # row wise sum

    logging.debug("\nfeatures:{}".format(countVec.get_feature_names()))

    dict_vec = DictVectorizer()

    dictvecfeature(bow_matrix_count, "bow_matrix_count")
    dictvecfeature(follower_count, "follower_count")
    dictvecfeature(retweet_count, "retweet_count")
    dictvecfeature(hashtag_count, "hashtag_count")
    dictvecfeature(mention_count, "mention_count")
    dictvecfeature(sentiment_pol, "sentiment_pol")
    dictvecfeature(favorite_count, "favorite_count")
    dictvecfeature(verify_count, "verify_count")

    bowmatrixFM = dict_vec.fit_transform(bowmatrix_count)
    logging.debug(
        "\nmentioncountFeatureMatrix shape:{}".format(
            bowmatrixFM.shape))

    followercountFM = dict_vec.fit_transform(followercount_features)
    logging.debug(
        "\nmentioncountFeatureMatrix shape:{}".format(
            followercountFM.shape))

    retweetcountFM = dict_vec.fit_transform(retweetcount_features)
    logging.debug(
        "\nmentioncountFeatureMatrix shape:{}".format(
            retweetcountFM.shape))

    hashtagcountFM = dict_vec.fit_transform(hashtagcount_features)
    logging.debug(
        "\nmentioncountFeatureMatrix shape:{}".format(
            hashtagcountFM.shape))

    mentioncountFM = dict_vec.fit_transform(mentioncount_features)
    logging.debug(
        "\nmentioncountFeatureMatrix shape:{}".format(
            mentioncountFM.shape))

    sentipolarityFM = dict_vec.fit_transform(pol_count)
    logging.debug(
        "\nmentioncountFeatureMatrix shape:{}".format(
            sentipolarityFM.shape))

    favcountFM = dict_vec.fit_transform(fav_count)
    logging.debug(
        "\nmentioncountFeatureMatrix shape:{}".format(
            favcountFM.shape))

    vrcountFM = dict_vec.fit_transform(ver_count)
    logging.debug(
        "\nmentioncountFeatureMatrix shape:{}".format(
            vrcountFM.shape))

    # Horizontally stack matrices (column-wise) to build one large feature
    # matrix
    sp_matrix = hstack([bow_matrix,
                        bowmatrixFM,
                        followercountFM,
                        retweetcountFM,
                        hashtagcountFM,
                        mentioncountFM,
                        sentipolarityFM,
                        favcountFM,
                        vrcountFM])
    x_columns = countVec.get_feature_names() + ['bow_matrix_count'] + \
        ['follower_count'] + ['retweet_count'] + ['hashtag_count'] + \
        ['mention_count'] + ['sentiment_pol'] + ['favorite_count'] + \
        ['verify_count']

    logging.info("\nMatrix Shape:{}".format(sp_matrix.shape))

    x_train = sp_matrix.toarray()
    # matplotlib.style.use('ggplot')
    # df_bfr = pd.DataFrame(data=x_train)
    # df_bfr.columns = x_columns
    # plt.hist(df_bfr['retweet_count'], bins=50)
    # plt.show()
    # plt.close()
    # sys.exit(1)

    # scaling and normalization of features
    # Scale input vectors individually to unit norm
    normalize_xtrain = normalize(x_train)
    std_scale_xtrain = StandardScaler().fit_transform(
        normalize_xtrain)  # mean = 0 and variance = 1
    scaler_xtrain = MinMaxScaler().fit_transform(
        std_scale_xtrain)  # min and max range

    df = pd.DataFrame(data=scaler_xtrain, index=index_id)
    df.columns = x_columns
    mergedf = pd.DataFrame(index=index_id)
    mergedf['tweets'] = documents
    mergedf['verify'] = verify_count
    mergedf['screen_name'] = screen_name
    mergedf['description'] = description

    logging.debug("\n\ndf info: {}".format(df.info()))
    logging.debug("\ndf head: {}".format(df.head()))
    logging.debug(
        "\nAny null values in data frame: {}".format(
            df.isnull().values.any()))
    logging.info(
        "\nAny null values in data frame: {}".format(
            df.isnull().values.any()))

    logging.debug("\n\neuclidean_distances: {}".format(x_train))
    logging.debug("---------------------------------------------------------")
    logging.debug(euclidean_distances(StandardScaler().fit_transform(x_train)))

    logging.debug("\n\nmax and min of bow_matrix_count: {} {}".format(
        df['bow_matrix_count'].max(), df['bow_matrix_count'].min()))
    logging.debug("\n\nmax and min of bow_matrix_count: {} {}".format(
        df['follower_count'].max(), df['follower_count'].min()))
    logging.debug("\n\nmax and min of bow_matrix_count: {} {}".format(
        df['retweet_count'].max(), df['retweet_count'].min()))
    logging.debug("\n\nmax and min of bow_matrix_count: {} {}".format(
        df['hashtag_count'].max(), df['hashtag_count'].min()))
    logging.debug("\n\nmax and min of bow_matrix_count: {} {}".format(
        df['mention_count'].max(), df['mention_count'].min()))
    logging.debug("\n\nmax and min of bow_matrix_count: {} {}".format(
        df['sentiment_pol'].max(), df['sentiment_pol'].min()))
    logging.debug("\n\nmax and min of bow_matrix_count: {} {}".format(
        df['favorite_count'].max(), df['favorite_count'].min()))

    analyser = countVec.build_analyzer()
    logging.debug(
        "\nHow this tweet was tokenized: {}".format(
            json.dumps(
                analyser(
                    documents[0]), analyser(
                    documents[1]), analyser(
                        documents[2]), analyser(
                            documents[3]), analyser(
                                documents[4]), indent=4)))
    logging.debug("\nCorresponding feature vector: {}".format(
        x_train[0], x_train[1], x_train[2], x_train[3], x_train[4]))
    logging.debug(
        "\n\nNames of additional features: {}".format(
            dict_vec.get_feature_names()))

    del bowmatrix_count[:]
    del followercount_features[:]
    del retweetcount_features[:]
    del hashtagcount_features[:]
    del mentioncount_features[:]
    del pol_count[:]
    del fav_count[:]
    del ver_count[:]

    return df, mergedf, countVec.get_feature_names()


def plotpca(exp_var, figname):
    plt.plot(exp_var)
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.title("PCA", y=1.03)
    plt.savefig(figname, dpi=800)
    plt.close()


def plotsil(x_data, plot_kmeans, figname):
    f, ax = plt.subplots()
    ax.plot(x_data, plot_kmeans)
    plt.ylabel("Silouette")
    plt.xlabel("k")
    plt.title("Silouette for K-means cell's behaviour", y=1.03)
    plt.savefig(figname, dpi=800)
    plt.close()


def calscore(X_reduced, random_state):
    plot_kmeans = []
    x_data = []
    for num in range(5, 16):
        kmeans = KMeans(
            init='k-means++',
            n_clusters=num,
            random_state=random_state)
        kmeans.fit(X_reduced)

        labels = kmeans.labels_
        plot_kmeans.append(metrics.silhouette_score(X_reduced, labels))
        x_data.append(num)
        logging.info("cluster no:{}, silhouetter score:{}".format(num, metrics.silhouette_score(X_reduced, labels)))
    # Gets the max sil value and corresponding cluster no.
    max_value = max(plot_kmeans)
    max_index = plot_kmeans.index(max_value) + 5

    return plot_kmeans, x_data, max_index


def fitkmeans(df, X_reduced, n_clusters, random_state, sample_size):
    # This initializes the centroids to be (generally) distant from each other, leading to provably better results than
    # random initialization, as shown in the reference.
    kmeans_model = KMeans(
        init='k-means++',
        n_clusters=n_clusters,
        random_state=random_state).fit(X_reduced)
    labels = kmeans_model.labels_
    centroids = kmeans_model.cluster_centers_
    logging.info("\033[92m")
    if sample_size != 0:
        logging.info(
            "\nSilhouette Coefficient:{}".format(
                metrics.silhouette_score(
                    X_reduced,
                    labels,
                    sample_size=sample_size)))
    else:
        logging.info(
            "\nSilhouette Coefficient:{}".format(
                metrics.silhouette_score(
                    X_reduced,
                    labels)))
    logging.info("\033[0m")
    return df, labels, centroids, kmeans_model


def plotkmeans(X_reduced, label_color, centroids, figname, num_of_cluster, labels, col_map):
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=label_color)
    plt.hold(True)
    # plt.legend()
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=100.0, color='black')
    plt.title('K-means clusters the dataset into (%d) clusters' % num_of_cluster, y=1.03)
    plt.savefig(figname, dpi=800)
    plt.close()


def generatewordcloud(n_clusters, order_centroids, feature_name):
    for i in range(n_clusters):
        word_cluster = []
        logging.info("Cluster {}:".format(i))
        logging.debug("Cluster {}:".format(i))
        for ind in order_centroids[i, :50]:
            logging.debug(feature_name[ind])
            logging.info(feature_name[ind])
            word_cluster.append(feature_name[ind])
        logging.info("\n")
        words = ' '.join(word_cluster)
        wordcloud = WordCloud(
            stopwords=STOPWORDS,
            background_color='black',
            margin=10,
        ).generate(words)

        base_dir_1 = '/Cluster_Epl/wordcloud/1/'
        base_dir_2 = '/Cluster_Epl/wordcloud/2/'
        plt.imshow(wordcloud)
        plt.axis('off')
        imgname = str(i) + '.png'
        file_path = os.path.join(base_dir_1, imgname)
        plt.title("Cluster: " + str(i), y=1.03)
        plt.savefig(file_path, dpi=800)
        plt.close()


def getclustercount(cid_tweets, clusterno, home, away):
    nc_h = nc_a = 0
    for k, v in cid_tweets.items():
        if k == clusterno:
            for sent in v:
                sent = sent.split()
                if len([word for word in home if word in sent]) > 0:
                    # logging.info("home tweet: {}".format(sent))
                    # logging.info("words found in home tweet: {}".format([word for word in home if word in sent]))
                    # logging.debug("home tweet: {}".format(sent))
                    # logging.debug("words found in home tweet: {}".format([word for word in home if word in sent]))
                    nc_h += 1
                if len([word for word in away if word in sent]) > 0:
                    # logging.info("home tweet: {}".format(sent))
                    # logging.info("words found in home tweet: {}".format([word for word in home if word in sent]))
                    # logging.debug("away tweet: {}".format(sent))
                    # logging.debug("words found in away tweet: {}".format([word for word in away if word in sent]))
                    nc_a += 1
                    # print "\n"
    return nc_h, nc_a


def savepreprocessdb(tweet_tokenize):
    hashtag_count = 0
    mention_count = 0

    # regex_replacer = RegexpReplacer()
    # replacer = RepeatReplacer()
    stopset = set(stopwords.words('english'))
    prefixes = ('rt', '//', 'http', '\\', '#rt')

    # rec[0] = replacer.replace(rec[0])

    tweet_tokenize = ("".join([" " + i for i in tweet_tokenize])).strip()
    analysis = TextBlob(tweet_tokenize)
    # get the sentiment polarity no
    sent_pol = analysis.sentiment.polarity
    """
     replacer.replace("can't is a contraction")
     cannot is a contraction'
    """
    # tweet_tokenize = regex_replacer.replace(tweet_tokenize)
    tweet_tokenize = tweet_tokenize.split(' ')
    # removes prefixes
    tweet_tokenize = [token for token in tweet_tokenize if not token.startswith(
        prefixes) and len(token) != 1]
    # removes stopwords
    tweet_tokenize = [
        token for token in tweet_tokenize if token not in stopset]
    # removes digit
    tweet_tokenize = [
        x for x in tweet_tokenize if not any(
            c.isdigit() for c in x)]
    # counts hashtag and mention
    for item in tweet_tokenize:
        if item.startswith('#'):
            hashtag_count += 1
        if item.startswith('@'):
            mention_count += 1

    # u"hello\xffworld".encode("ascii", "ignore")
    # b'helloworld'
    doc = "".join([" " + i.encode('ascii', 'ignore').decode('ascii') if not i.startswith(
        "'") and i not in string.punctuation else i.encode('ascii', 'ignore').decode('ascii')
        for i in tweet_tokenize]).strip()

    return doc, sent_pol, hashtag_count, mention_count
