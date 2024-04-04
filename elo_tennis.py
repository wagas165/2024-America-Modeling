import csv
import math
import pickle
import pandas
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import style
from pandas import DataFrame
from pandas import Series
from datetime import datetime, timedelta


# define k factor assumptions
def k_factor(matches_played):
    K = 250
    offset = 5
    shape = 0.4
    return K / (matches_played + offset) ** shape


# define a function for calculating the expected score of player_A
# expected score of player_B = 1 - expected score of player
def calc_exp_score(playerA_rating, playerB_rating):
    exp_score = 1 / (1 + (10 ** ((playerB_rating - playerA_rating) / 400)))
    return exp_score


# define a function for calculating new elo
def update_elo(old_elo, k, actual_score, expected_score):
    new_elo = old_elo + k * (actual_score - expected_score)
    return new_elo


# winning a match regardless the number of sets played = 1
score = 1
####获取选手信息
with open('atp_players.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    col_index = [0, 1, 2, 5]
    all_players = []
    for row in readCSV:
        player_info = []
        player_info.append(row[0])
        name = str(row[1]) + ' ' + str(row[2])
        player_info.append(name)
        all_players.append(player_info)

# Column headers for player dataframe
player_col_header = ['player_id', 'name']

# Create a dataframe for keeping track of player info
# every player starts with an elo rating of 1500
players = DataFrame(all_players, columns=player_col_header)
players.drop(0, axis=0, inplace=True)
###当前elo先全部更新为1500 后续开始有比赛再做调整

players['current_elo'] = Series(1500, index=players.index)
players['last_tourney_date'] = Series('N/A', index=players.index)
players['matches_played'] = Series(0, index=players.index)
# Convert objects within dataframe to numeric
players = players.apply(pd.to_numeric, errors='ignore')

####要输出的timeseries文件
names=players['name'].tolist()
elo_timeseries = DataFrame(columns=names)
# start_date = '2011-01-01'
# end_date = '2023-12-31'
# date_range = pd.date_range(start=start_date, end=end_date)
# elo_timeseries = elo_timeseries.reindex(date_range)

###开始统计每个人的elo成绩
for current_year in range(1968, 2024):

    print(f'开始处理 {current_year} 年的比赛数据...')

    current_year_file_name = 'atp_matches_' + str(current_year) + '.csv'
    matches = pd.read_csv(current_year_file_name)

    # Sort matches dataframe by tourney_date and then by round

    sorter = ['RR', 'R128', 'R64', 'R32', 'R16', 'QF', 'SF', 'F']
    matches['round'] = matches['round'].astype('category')
    matches['round'] = matches['round'].cat.set_categories(sorter)
    matches = matches.sort_values(by=['tourney_date', 'round'], ascending=[True, True])

    for index, row in matches.iterrows():

        winner_id = row['winner_id']
        loser_id = row['loser_id']
        tourney_date = row['tourney_date']

        winner_name = row['winner_name']
        loser_name = row['loser_name']

        index_winner = players[players['player_id'] == winner_id].index.tolist()
        index_loser = players[players['player_id'] == loser_id].index.tolist()

        old_elo_winner = players.loc[index_winner[0], 'current_elo']
        old_elo_loser = players.loc[index_loser[0], 'current_elo']

        exp_score_winner = calc_exp_score(old_elo_winner, old_elo_loser)
        exp_score_loser = 1 - exp_score_winner

        matches_played_winner = players.loc[index_winner[0], 'matches_played']
        matches_played_loser = players.loc[index_loser[0], 'matches_played']

        new_elo_winner = update_elo(old_elo_winner, k_factor(matches_played_winner), score, exp_score_winner)
        new_elo_loser = update_elo(old_elo_loser, k_factor(matches_played_loser), score - 1, exp_score_loser)

        players.loc[index_winner[0], 'current_elo'] = new_elo_winner
        players.loc[index_winner[0], 'last_tourney_date'] = tourney_date
        players.loc[index_winner[0], 'matches_played'] = players.loc[index_winner[0], 'matches_played'] + 1
        players.loc[index_loser[0], 'current_elo'] = new_elo_loser
        players.loc[index_loser[0], 'last_tourney_date'] = tourney_date
        players.loc[index_loser[0], 'matches_played'] = players.loc[index_loser[0], 'matches_played'] + 1


        tourney_date_timestamp = pandas.to_datetime(tourney_date, format='%Y%m%d')

        if tourney_date_timestamp not in elo_timeseries.index:
            elo_timeseries.loc[tourney_date_timestamp] = None

        ##补齐了日期

        if (winner_name in elo_timeseries.columns) and (loser_name in elo_timeseries.columns):  ###都在表中
            elo_timeseries.loc[tourney_date_timestamp, winner_name] = new_elo_winner
            elo_timeseries.loc[tourney_date_timestamp, loser_name] = new_elo_loser

        elif winner_name in elo_timeseries.columns:   ###loser不在表中
            elo_timeseries.loc[tourney_date_timestamp, winner_name] = new_elo_winner
            elo_timeseries[loser_name]=None
            elo_timeseries.loc[tourney_date_timestamp,loser_name]=new_elo_loser

        elif loser_name in elo_timeseries.columns:  ###winner不在表中
            elo_timeseries.loc[tourney_date_timestamp, loser_name] = new_elo_loser
            elo_timeseries[winner_name] = None
            elo_timeseries.loc[tourney_date_timestamp, winner_name] = new_elo_winner

        else:
            elo_timeseries[winner_name] = None
            elo_timeseries.loc[tourney_date_timestamp, winner_name] = new_elo_winner
            elo_timeseries[loser_name] = None
            elo_timeseries.loc[tourney_date_timestamp, loser_name] = new_elo_loser


    ##Uncomment to output year end elo_rankings for every year between 1968 and 2015
    # output_file_name = str(current_year) + '_yr_end_elo_ranking.csv'
    # players.to_csv(output_file_name)

elo_timeseries=elo_timeseries.T
elo_timeseries.to_csv('历史玩家ELO.csv')