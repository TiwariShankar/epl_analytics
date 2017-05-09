#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
@author: shankartiwari
"""

from utility import *


def getResults(finaldf, match):
    logging.info(
        "\n--------------------------------Evaluating results------------------------------------------")
    cid_tweets = {k: v["tweets"].tolist()
                  for k, v in finaldf.groupby("cluster_id")}
    match = match.split('#')
    home_team = match[1][0:3].strip()
    away_team = match[1][3:].strip()
    home_tag = away_tag = []
    if home_team == 'CHE':
        home_tag = [
            '#cfc',
            '#chelsea',
            '#willian',
            '#hazard',
            '#chelseaindiasupportersclub',
            '#chelseafc',
            '#edenhazard',
            '#stamfordbridge',
            '#chelseaday',
            '#chelseafans',
            '#cfcindo',
            '#londonisblue',
            '#theblues',
            '#chelseanews',
            '#ctid',
            '#antonioconte',
            '#chelseatillidie',
            '#blueisthecolour',
            '#blueday',
            '#moda',
            '#londononly',
            '#londonforyo',
            '#blues',
            '#ktbffh',
            '#conte',
            '#coyb',
            '#chelseaworld',
            '#blueteam',
            '#foreverblue',
            '#comeonchelsea',
            '#prideoflondon',
            '#ktbffh',
            '#ifitsnotblueitwillbe',
            '#cfcarmy',
            '#theblues',
            '#comeonyoublues',
            '#wearechelsea']
    if away_team == 'CHE':
        away_tag = [
            '#cfc',
            '#chelsea',
            '#willian',
            '#hazard',
            '#chelseaindiasupportersclub',
            '#chelseafc',
            '#edenhazard',
            '#stamfordbridge',
            '#chelseaday',
            '#chelseafans',
            '#cfcindo',
            '#londonisblue',
            '#theblues',
            '#chelseanews',
            '#ctid',
            '#antonioconte',
            '#chelseatillidie',
            '#blueisthecolour',
            '#blueday',
            '#moda',
            '#londononly',
            '#londonforyo',
            '#blues',
            '#ktbffh',
            '#conte',
            '#coyb',
            '#chelseaworld',
            '#blueteam',
            '#foreverblue',
            '#comeonchelsea',
            '#prideoflondon',
            '#ktbffh',
            '#ifitsnotblueitwillbe',
            '#cfcarmy',
            '#theblues',
            '#comeonyoublues',
            '#wearechelsea']
    if home_team == 'MUN':
        home_tag = [
            '#mun',
            '#manchesterunited',
            '#mufc',
            '#ggmu',
            '#mourinho',
            '#davesaves',
            '#reddevils',
            '#red_devils',
            '#oldtrafford',
            '#manu',
            '#manchesterisred',
            '#manunited',
            '#busby',
            '#red_devils',
            '#mufcfamily',
            '#unitedtillidie',
            '#mufcfans',
            '#unitedfans',
            '#mufcfanpics',
            '#manutdfans',
            '#utd',
            '#redordead']
    if away_team == 'MUN':
        away_tag = [
            '#mun',
            '#manchesterunited',
            '#mufc',
            '#ggmu',
            '#mourinho',
            '#davesaves',
            '#reddevils',
            '#red_devils',
            '#oldtrafford',
            '#manu',
            '#manchesterisred',
            '#manunited',
            '#busby',
            '#red_devils',
            '#mufcfamily',
            '#unitedtillidie',
            '#mufcfans',
            '#unitedfans',
            '#mufcfanpics',
            '#manutdfans',
            '#utd',
            '#redordead']
    if home_team == 'MCI':
        home_tag = [
            '#mci',
            '#mancity',
            '#mcfc',
            '#manchester',
            '#manchestercity',
            '#mcfcofficial',
            '#manchestercityfc',
            '#mancityfc',
            '#guardiola',
            '#pepguardiola',
            '#bluemoon',
            '#fnh',
            '#ctid',
            '#mcwfc',
            '#blue']
    if away_team == 'MCI':
        away_tag = [
            '#mci',
            '#mancity',
            '#mcfc',
            '#manchester',
            '#manchestercity',
            '#mcfcofficial',
            '#manchestercityfc',
            '#mancityfc',
            '#guardiola',
            '#pepguardiola',
            '#bluemoon',
            '#fnh',
            '#ctid',
            '#mcwfc',
            '#blue']
    if home_team == 'LIV':
        home_tag = [
            '#lfc',
            '#ynwa',
            '#liverpool',
            '#nutmeg',
            '#reds',
            '#liverpoolproud',
            '#anfield',
            '#fort',
            '#ynwa',
            '#firmino',
            '#robertofirmino',
            '#lfcfamily',
            '#thereds',
            '#theredslfc',
            '#football',
            '#lfcfans',
            '#kloppsarmy',
            '#wegoagain',
            '#makeusdream',
            '#lovemyclub',
            '#yellowtshirt',
            '#thisisanfield',
            '#anfieldstadium',
            '#lallana',
            '#redmen',
            '#liverpoolfcfanclub',
            '#stevengerrard',
            '#youllneverwalkalone',
            '#jurgenklopp',
            '#liverpoolfans',
            '#ywnwa',
            '#gerrard',
            '#red',
            '#sakho',
            '#liverpoolfamily']
    if away_team == 'LIV':
        away_tag = [
            '#lfc',
            '#ynwa',
            '#liverpool',
            '#nutmeg',
            '#reds',
            '#liverpoolproud',
            '#anfield',
            '#fort',
            '#ynwa',
            '#firmino',
            '#robertofirmino',
            '#lfcfamily',
            '#thereds',
            '#theredslfc',
            '#football',
            '#lfcfans',
            '#kloppsarmy',
            '#wegoagain',
            '#makeusdream',
            '#lovemyclub',
            '#yellowtshirt',
            '#thisisanfield',
            '#anfieldstadium',
            '#lallana',
            '#redmen',
            '#liverpoolfcfanclub',
            '#stevengerrard',
            '#youllneverwalkalone',
            '#jurgenklopp',
            '#liverpoolfans',
            '#ywnwa',
            '#gerrard',
            '#red',
            '#sakho',
            '#liverpoolfamily']
    if home_team == 'TOT':
        home_tag = [
            '#tottenhamhotspur',
            '#coyg',
            '#tottenham',
            '#spurs',
            '#coys',
            '#ttid',
            '#thfc',
            '#yidarmy',
            '#tottenhamfc',
            '#totw',
            '#stadiumoflight',
            '#whitehartlane',
            '#tottenhamtillidie',
            '#comeonyouspurs',
            '#spursarmy',
            '#togetherthfc',
            '#spursfamily',
            '#spurs']
    if away_team == 'TOT':
        away_tag = [
            '#tottenhamhotspur',
            '#coyg',
            '#tottenham',
            '#spurs',
            '#coys',
            '#ttid',
            '#thfc',
            '#yidarmy',
            '#tottenhamfc',
            '#totw',
            '#stadiumoflight',
            '#whitehartlane',
            '#tottenhamtillidie',
            '#comeonyouspurs',
            '#spursarmy',
            '#togetherthfc',
            '#spursfamily',
            '#spurs']
    if home_team == 'EVE':
        home_tag = [
            '#coyb',
            '#toffees',
            '#upthetoffees',
            '#toonarmy',
            '#steventaylor',
            '#upontyne',
            '#stjamespark',
            '#stjames',
            '#everton',
            '#evertonfc',
            '#eagles',
            '#neverforgotten',
            '#seahouses',
            '#gerarddeulofeu',
            '#efc',
            '#brentford',
            '#evertonthe',
            '#morganschneiderlin',
            '#calcio',
            '#lukaku']
    if away_team == 'EVE':
        away_tag = [
            '#coyb',
            '#toffees',
            '#upthetoffees',
            '#toonarmy',
            '#steventaylor',
            '#upontyne',
            '#stjamespark',
            '#stjames',
            '#everton',
            '#evertonfc',
            '#eagles',
            '#neverforgotten',
            '#seahouses',
            '#gerarddeulofeu',
            '#efc',
            '#brentford',
            '#evertonthe',
            '#morganschneiderlin',
            '#calcio',
            '#lukaku']
    if home_team == 'ARS':
        home_tag = [
            '#gooner',
            '#ozil',
            '#gunners',
            '#westminster',
            '#arsenalfc',
            '#wengerout',
            '#coyg',
            '#afc',
            '#arsenal',
            '#gunners',
            '#arsenalfc',
            '#alexis',
            '#highbury',
            '#prideoflondon',
            '#emiratesstadium',
            '#wehatetottenham',
            '#arsene',
            '#wenger',
            '#londonisred',
            '#goonerette',
            '#yagoonersya',
            '#arsenalfantv']
    if away_team == 'ARS':
        away_tag = [
            '#gooner',
            '#ozil',
            '#gunners',
            '#westminster',
            '#arsenalfc',
            '#wengerout',
            '#coyg',
            '#afc',
            '#arsenal',
            '#gunners',
            '#arsenalfc',
            '#alexis',
            '#highbury',
            '#prideoflondon',
            '#emiratesstadium',
            '#wehatetottenham',
            '#arsene',
            '#wenger',
            '#londonisred',
            '#goonerette',
            '#yagoonersya',
            '#arsenalfantv']
    if home_team == 'LEI':
        home_tag = [
            '#lcfc',
            '#coyf',
            '#leicester',
            '#leicestercity',
            '#fuchs',
            '#foxes',
            '#foxesneverquit',
            '#lei',
            '#fearlessfoxes',
            '#mahrez',
            '#vardy',
            '#leicestertilidie',
            '#gofoxes',
            '#ranieri',
            '#bird',
            '#coys',
            '#comeonyoufoxes']
    if away_team == 'LEI':
        away_tag = [
            '#lcfc',
            '#coyf',
            '#leicester',
            '#leicestercity',
            '#fuchs',
            '#foxes',
            '#foxesneverquit',
            '#lei',
            '#fearlessfoxes',
            '#mahrez',
            '#vardy',
            '#leicestertilidie',
            '#gofoxes',
            '#ranieri',
            '#bird',
            '#coys',
            '#comeonyoufoxes']
    if home_team == 'WBA':
        home_tag = [
            '#wba',
            '#baggies',
            '#wbafc',
            '#westbrom',
            '#westbromwichalbionfc',
            '#ynwa',
            '#westbromfc',
            '#coyi',
            '#hawthorns',
            '#throstles']
    if away_team == 'WBA':
        away_tag = [
            '#wba',
            '#baggies',
            '#wbafc',
            '#westbrom',
            '#westbromwichalbionfc',
            '#ynwa',
            '#westbromfc',
            '#coyi',
            '#hawthorns',
            '#throstles']
    if home_team == 'SOU':
        home_tag = [
            '#saintsfc ',
            '#wemarchon',
            '#southamptonfc',
            '#saints',
            '#upthesaints',
            '#marchasone',
            '#ronaldkoeman',
            '#southampton',
            '#sfc',
            '#stmarys',
            '#koeman',
            '#redandwhite',
            '#wearesouthampton',
            '#josefonte',
            '#fonte',
            '#coyr',
            '#soares',
            '#redarmy',
            '#virgilvandijk']
    if away_team == 'SOU':
        away_tag = [
            '#saintsfc ',
            '#wemarchon',
            '#southamptonfc',
            '#saints',
            '#upthesaints',
            '#marchasone',
            '#ronaldkoeman',
            '#southampton',
            '#sfc',
            '#stmarys',
            '#koeman',
            '#redandwhite',
            '#wearesouthampton',
            '#josefonte',
            '#fonte',
            '#coyr',
            '#soares',
            '#redarmy',
            '#virgilvandijk']
    if home_team == 'SWA':
        home_tag = [
            '#swans',
            '#swanseacity',
            '#swansea',
            '#broadsideswansea',
            '#scfc',
            '#swanseacityfc',
            '#southwales',
            '#swan',
            '#sfc',
            '#safc',
            '#liberty',
            '#yjb',
            '#jackarmy']
    if away_team == 'SWA':
        away_tag = [
            '#swans',
            '#swanseacity',
            '#swansea',
            '#broadsideswansea',
            '#scfc',
            '#swanseacityfc',
            '#southwales',
            '#swan',
            '#sfc',
            '#safc',
            '#liberty',
            '#yjb',
            '#jackarmy']
    if home_team == 'CRY':
        home_tag = [
            '#cpfc',
            '#eagles ',
            '#crystalpalace',
            '#crystalpalacefc',
            '#coys',
            '#coyp',
            '#palace',
            '#calcio',
            '#southlondonandproud',
            '#official_cpfc ',
            '#bigsam',
            '#sam',
            '#saha',
            '#selhurst',
            '#glaziers']
    if away_team == 'CRY':
        away_tag = [
            '#cpfc',
            '#eagles ',
            '#crystalpalace',
            '#crystalpalacefc',
            '#coys',
            '#coyp',
            '#palace',
            '#calcio',
            '#southlondonandproud',
            '#official_cpfc ',
            '#bigsam',
            '#sam',
            '#saha',
            '#selhurst',
            '#glaziers']
    if home_team == 'SUN':
        home_tag = [
            '#safc',
            '#stadiumoflight',
            '#sunderland',
            '#sunderlandfc',
            '#defoe',
            '#blackcats',
            '#eastcoast',
            '#sun',
            '#light',
            '#moyes']
    if away_team == 'SUN':
        away_tag = [
            '#safc',
            '#stadiumoflight',
            '#sunderland',
            '#sunderlandfc',
            '#defoe',
            '#blackcats',
            '#eastcoast',
            '#sun',
            '#light',
            '#moyes']
    if home_team == 'STK':
        home_tag = [
            '#scfc',
            '#potters',
            '#bet365',
            '#stokecity',
            '#coymp',
            '#crouch',
            '#shaqiri',
            '#berahino',
            '#hughes']
    if away_team == 'STK':
        away_tag = [
            '#scfc',
            '#potters',
            '#bet365',
            '#stokecity',
            '#coymp',
            '#crouch',
            '#shaqiri',
            '#berahino',
            '#hughes']
    if len(home_tag) != 0 and len(away_tag) != 0:
        df_final_results = pd.DataFrame()
        df_final_results['match'] = match
        df_final_results['home_team'] = home_team
        df_final_results['away_team'] = away_team
        home_tag = [word.strip() for word in home_tag]
        away_tag = [word.strip() for word in away_tag]
        nc0_h, nc0_a = getclustercount(cid_tweets, 0, home_tag, away_tag)
        nc1_h, nc1_a = getclustercount(cid_tweets, 1, home_tag, away_tag)
        nc2_h, nc2_a = getclustercount(cid_tweets, 2, home_tag, away_tag)
        nc3_h, nc3_a = getclustercount(cid_tweets, 3, home_tag, away_tag)
        nc4_h, nc4_a = getclustercount(cid_tweets, 4, home_tag, away_tag)
        df_final_results["nc0_h"] = nc0_h
        df_final_results["nc0_a"] = nc0_a
        df_final_results["nc1_h"] = nc1_h
        df_final_results["nc1_a"] = nc1_a
        df_final_results["nc2_h"] = nc2_h
        df_final_results["nc2_a"] = nc2_a
        df_final_results["nc3_h"] = nc3_h
        df_final_results["nc3_a"] = nc3_a
        df_final_results["nc4_h"] = nc4_h
        df_final_results["nc4_a"] = nc4_a
        x_columns = ['match'] + ['home_team'] + ['away_team'] + ['nc0_h'] + ['nc0_a'] + \
                    ['nc1_h'] + ['nc1_a'] + \
                    ['nc2_h'] + ['nc2_a'] + ['nc3_h'] + ['nc3_a'] + ['nc4_h'] + ['nc4_a']
        df_final_results.columns = x_columns
        if not os.path.exists('results.csv'):
            df_final_results.to_csv('results.csv',  mode='w', header=True, encoding='utf-8', index=False)
        else:
            df_final_results.to_csv('results.csv',  mode='a', header=False, encoding='utf-8', index=False)
        logging.info(
            "\n Home Team: {} vs Away Team: {} \n Result for each cluster is shown below".format(home_team, away_team))
        logging.info(
            " c0_h: {} \tc0_a: {} \t| c1_h: {} \tc1_a: {} \t| c2_h: {} \tc2_a: {} \t| c3_h: {} \tc3_a: {} \t| c4_h: {} \tc4_a: {} ".format(
                nc0_h,
                nc0_a,
                nc1_h,
                nc1_a,
                nc2_h,
                nc2_a,
                nc3_h,
                nc3_a,
                nc4_h,
                nc4_a))
    else:
        logging.info("\n Team info is not present")
