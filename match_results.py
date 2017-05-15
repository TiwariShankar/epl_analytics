#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
@author: shankartiwari
"""

from utility import *
from tags import *
from collections import OrderedDict
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score


def saveResults(finaldf, match, result, model):
    logging.info(
        "\n--------------------------------Evaluating results------------------------------------------")
    cid_tweets = {k: v["tweets"].tolist()
                  for k, v in finaldf.groupby("cluster_id")}
    if match == '#JuveFCB':
        mod_mat = match.split('#')
        home_team = mod_mat[1][0:4].strip()
        away_team = mod_mat[1][4:].strip()
    else:
        mod_mat = match.split('#')
        home_team = mod_mat[1][0:3].strip()
        away_team = mod_mat[1][3:].strip()
    data_dict = OrderedDict()
    home_tag, away_tag = taglst(home_team, away_team)
    if len(home_tag) != 0 and len(away_tag) != 0:
        data_dict['match'] = match
        data_dict['home_team'] = home_team
        data_dict['away_team'] = away_team
        home_tag = [word.strip() for word in home_tag]
        away_tag = [word.strip() for word in away_tag]
        for k, v in cid_tweets.items():
            col_name_home = "nc"+str(k)+"_h"
            col_name_away = "nc"+str(k)+"_a"
            data_dict[col_name_home], data_dict[col_name_away] = \
                getclustercount(cid_tweets, k, home_tag, away_tag)
        if model == "train":
            data_dict['result'] = result
            pd.DataFrame.from_dict(data_dict, orient='index').T.to_csv(
                              'data.csv', header=False, index=False, encoding='utf-8',
                               mode='a')
        else:
            pd.DataFrame.from_dict(data_dict, orient='index').T.to_csv(
                              'data.csv', header=False, index=False, encoding='utf-8',
                               mode='a')
    else:
        logging.info("\n Team info is not present")


def rfmodel():
    rf = RandomForestRegressor(random_state=10, n_estimators=100)
    df_train = pd.read_csv('data.csv', error_bad_lines=False, header=None)
    len_col = len(df_train.columns)
    len_row = df_train.shape[0]
    df_train = df_train.iloc[:, 3:]
    df_train = df_train.fillna(df_train.mean())
    # std_scale_xtrain = StandardScaler().fit_transform(
    #     df_train)  # mean = 0 and variance = 1
    # scaler_xtrain = MinMaxScaler().fit_transform(
    #     std_scale_xtrain)  # min and max range
    # scaled_df = pd.DataFrame(data=scaler_xtrain)

    len_col_sdf = len(df_train.columns)
    len_row_sdf = df_train.shape[0]
    # training the data frame
    xTrain = df_train.iloc[:len_row_sdf - 1, 0:len_col_sdf - 1]
    yTrain = df_train.iloc[:len_row_sdf - 1, len_col_sdf - 1:]
    estimator = rf.fit(xTrain, yTrain.values.ravel())
    # predciting test values
    xTest = df_train.tail(1).iloc[:len_row_sdf - 1, 0:len_col_sdf - 1]
    y_predict = estimator.predict(xTest)

    print "Prediction: (0-draw, 1-home_win, 2-away_win)", round(y_predict)
    score = cross_val_score(estimator, xTrain, yTrain.values.ravel(), cv=10).mean()
    print("Score = %.2f" % score)
    sys.exit(1)
